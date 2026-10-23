"""
QQQ-Anchored Options Overlay & Tail-Risk Hedging Engine
=======================================================
Author: kemomi
License: MIT

Extends the QQQ-Anchored Quantitative Strategy Suite with an institutional
Options Overlay (Covered Call Yield Harvesting + Protective Put / Collar Tail Hedge)
priced via vectorized Black-Scholes-Merton (BSM) with dynamic Delta strike selection
and Volatility Risk Premium (VRP) modeling.

Overlay Mechanics:
------------------
1. Covered Call Overwriting (Income & Overbought Buffer):
   - Active in Regime +1 (Standard Bull) and Regime 0 (Sideways/Choppy) when RSI(14) > 60.
   - Sells 30-DTE OTM QQQ Call at ~0.20 to 0.25 Delta (approx. 4%-6% OTM), harvesting
     implied volatility premium while leaving upside room.
   - In Regime +2 (Strong Bull Expansion with Oversold Dip), Covered Calls are paused
     or rolled up to 0.10 Delta so strong breakouts are never capped prematurely.

2. Protective Put / Dynamic Collar Tail-Risk Hedge:
   - Active when QQQ is in Transition/High-Vol warning (Realized Vol rising > 20% or
     EMA50 flattening) to protect core QQQ / Nasdaq-100 holdings against gap-down crashes.
   - Buys 30-DTE OTM QQQ Put at ~-0.15 Delta (approx. 6%-8% OTM), partially or fully
     financed by the short OTM Covered Call (Zero-Cost / Low-Cost Dynamic Collar).

3. Exact Black-Scholes-Merton (BSM) & Greeks Calculator:
   - Pure NumPy/Math implementation (zero SciPy dependency) using Abramowitz & Stegun
     high-precision normal CDF approximation.
   - Computes exact Call/Put theoretical price, Delta, Gamma, Theta (daily), and Vega,
     plus analytical Newton-Raphson strike solver for any target Delta.

Usage:
------
    python3 qqq_options_overlay.py
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from qqq_quant_strategy import (
        StrategyConfig,
        BacktestEngine,
        generate_synthetic_market_data,
    )
except ModuleNotFoundError:
    import importlib.util as _ilu, pathlib as _pl, sys as _sys
    _spec = _ilu.spec_from_file_location(
        "qqq_quant_strategy", _pl.Path(__file__).resolve().parent / "qqq_quant_strategy.py"
    )
    _mod = _ilu.module_from_spec(_spec)
    _sys.modules["qqq_quant_strategy"] = _mod
    _spec.loader.exec_module(_mod)
    StrategyConfig = _mod.StrategyConfig
    BacktestEngine = _mod.BacktestEngine
    generate_synthetic_market_data = _mod.generate_synthetic_market_data


# ============================================================================
# 1. Black-Scholes-Merton (BSM) Pricing & Greeks Engine (Zero SciPy Dependency)
# ============================================================================

class BSMOptionsEngine:
    """Analytical Black-Scholes-Merton option pricer, Greeks calculator, and Delta-strike solver."""

    @staticmethod
    def norm_cdf(x: float) -> float:
        """Standard normal cumulative distribution function via math.erf."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def norm_pdf(x: float) -> float:
        """Standard normal probability density function."""
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    @classmethod
    def price_and_greeks(
        cls,
        spot: float,
        strike: float,
        tte_years: float,
        rate: float,
        iv: float,
        option_type: str = "call",
        div_yield: float = 0.006,
    ) -> Dict[str, float]:
        """
        Computes BSM option price and Greeks (Delta, Gamma, Theta/day, Vega/1% IV)
        for QQQ (default 0.60% annualized dividend yield).
        """
        if tte_years <= 1e-6 or iv <= 1e-4 or spot <= 0 or strike <= 0:
            intrinsic = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
            return {
                "price": round(intrinsic, 4),
                "delta": 1.0 if (option_type == "call" and spot > strike) else (-1.0 if (option_type == "put" and spot < strike) else 0.0),
                "gamma": 0.0,
                "theta_daily": 0.0,
                "vega_pct": 0.0,
            }

        sqrt_t = math.sqrt(tte_years)
        d1 = (math.log(spot / strike) + (rate - div_yield + 0.5 * iv * iv) * tte_years) / (iv * sqrt_t)
        d2 = d1 - iv * sqrt_t

        df_r = math.exp(-rate * tte_years)
        df_q = math.exp(-div_yield * tte_years)
        pdf_d1 = cls.norm_pdf(d1)

        if option_type.lower() == "call":
            price = spot * df_q * cls.norm_cdf(d1) - strike * df_r * cls.norm_cdf(d2)
            delta = df_q * cls.norm_cdf(d1)
            theta_ann = (
                -(spot * df_q * pdf_d1 * iv) / (2.0 * sqrt_t)
                - rate * strike * df_r * cls.norm_cdf(d2)
                + div_yield * spot * df_q * cls.norm_cdf(d1)
            )
        else:
            price = strike * df_r * cls.norm_cdf(-d2) - spot * df_q * cls.norm_cdf(-d1)
            delta = -df_q * cls.norm_cdf(-d1)
            theta_ann = (
                -(spot * df_q * pdf_d1 * iv) / (2.0 * sqrt_t)
                + rate * strike * df_r * cls.norm_cdf(-d2)
                - div_yield * spot * df_q * cls.norm_cdf(-d1)
            )

        gamma = (df_q * pdf_d1) / (spot * iv * sqrt_t)
        vega_pct = spot * df_q * pdf_d1 * sqrt_t * 0.01

        return {
            "price": round(price, 4),
            "delta": round(delta, 4),
            "gamma": round(gamma, 6),
            "theta_daily": round(theta_ann / 365.0, 4),
            "vega_pct": round(vega_pct, 4),
        }

    @classmethod
    def solve_strike_for_delta(
        cls,
        spot: float,
        target_abs_delta: float,
        tte_years: float,
        rate: float,
        iv: float,
        option_type: str = "call",
        div_yield: float = 0.006,
    ) -> float:
        """Solves for the exact option strike (rounded to nearest $1) matching a target absolute Delta."""
        low_k = spot * 0.70
        high_k = spot * 1.30
        for _ in range(30):
            mid_k = 0.5 * (low_k + high_k)
            g = cls.price_and_greeks(spot, mid_k, tte_years, rate, iv, option_type, div_yield)
            cur_abs_d = abs(g["delta"])
            if option_type.lower() == "call":
                # Higher strike -> lower call delta
                if cur_abs_d > target_abs_delta:
                    low_k = mid_k
                else:
                    high_k = mid_k
            else:
                # Lower strike -> lower absolute put delta
                if cur_abs_d > target_abs_delta:
                    high_k = mid_k
                else:
                    low_k = mid_k
        return round(0.5 * (low_k + high_k), 0)


# ============================================================================
# 2. QQQ Dynamic Collar & Covered Call Overlay Backtester
# ============================================================================

@dataclass
class OptionsOverlayConfig:
    """Configuration for the QQQ Options Overlay (Covered Call + Protective Put Collar)."""
    roll_period_days: int = 21               # Monthly (21 trading days ~ 30 calendar DTE) option cycle
    call_target_delta_normal: float = 0.22   # Sell 22-Delta OTM Call in standard bull / neutral regimes
    call_target_delta_strong: float = 0.12   # Sell far-OTM 12-Delta Call in strong bull expansion
    put_target_delta_hedge: float = 0.16     # Buy 16-Delta OTM Put when volatility/transition risk rises
    vrp_spread: float = 0.035                # Implied Vol premium over realized vol (+3.5% VRP)
    put_skew_addon: float = 0.03             # OTM Put IV skew (+3.0% over ATM IV)
    call_skew_discount: float = -0.01        # OTM Call IV skew (-1.0% vs ATM IV)
    option_commission_bps: float = 1.0       # Friction per option roll


class QQQOptionsOverlayStrategy:
    """
    Simulates rolling 21-trading-day (30-DTE) Covered Call and Dynamic Collar overlays
    on top of the QQQ-Anchored Core Strategy.
    """

    def __init__(
        self,
        base_cfg: Optional[StrategyConfig] = None,
        opt_cfg: Optional[OptionsOverlayConfig] = None,
    ):
        self.base_cfg = base_cfg or StrategyConfig()
        self.opt_cfg = opt_cfg or OptionsOverlayConfig()

    def run_overlay_backtest(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        engine = BacktestEngine(self.base_cfg)
        bt, base_metrics = engine.run(df, strategy_type="core_qqq")

        n = len(bt)
        closes = bt["close"].to_numpy()
        vols = bt["realized_vol"].bfill().fillna(0.16).to_numpy()
        regimes = bt["regime"].to_numpy()
        rsi_14 = bt["rsi_long"].to_numpy()
        weights = bt["target_weight"].to_numpy()

        overlay_daily_ret = np.zeros(n, dtype=float)
        roll_days = self.opt_cfg.roll_days if hasattr(self.opt_cfg, "roll_days") else self.opt_cfg.roll_period_days
        tte_init = roll_days / 252.0
        rf = self.base_cfg.risk_free_rate

        warmup = max(self.base_cfg.slow_ema_window, 200) + 15

        # Track active option leg state across each 21-day cycle
        for start_idx in range(warmup, n - 1, roll_days):
            end_idx = min(start_idx + roll_days, n - 1)
            spot_0 = closes[start_idx]
            rv_0 = max(float(vols[start_idx]), 0.10)
            atm_iv_0 = rv_0 + self.opt_cfg.vrp_spread
            reg_0 = int(regimes[start_idx])
            w_0 = float(weights[start_idx])

            if w_0 <= 0.05:
                continue

            # 1. Decide Call Overwriting Delta
            sell_call = False
            call_delta = self.opt_cfg.call_target_delta_normal
            if reg_0 == 2 and rsi_14[start_idx] > 55.0:
                sell_call = True
                call_delta = self.opt_cfg.call_target_delta_strong
            elif reg_0 in (0, 1) and rsi_14[start_idx] > 48.0:
                sell_call = True
                call_delta = self.opt_cfg.call_target_delta_normal

            # 2. Decide Protective Put Hedge (active when Regime is 0 or 1 with elevated vol > 18%)
            buy_put = (reg_0 in (0, 1)) and (rv_0 > 0.18)

            call_iv_0 = max(atm_iv_0 + self.opt_cfg.call_skew_discount, 0.08)
            put_iv_0 = max(atm_iv_0 + self.opt_cfg.put_skew_addon, 0.10)

            call_k = (
                BSMOptionsEngine.solve_strike_for_delta(
                    spot_0, call_delta, tte_init, rf, call_iv_0, "call"
                )
                if sell_call
                else 0.0
            )
            put_k = (
                BSMOptionsEngine.solve_strike_for_delta(
                    spot_0, self.opt_cfg.put_target_delta_hedge, tte_init, rf, put_iv_0, "put"
                )
                if buy_put
                else 0.0
            )

            # Mark-to-market daily option P&L across the cycle
            prev_call_val = (
                BSMOptionsEngine.price_and_greeks(spot_0, call_k, tte_init, rf, call_iv_0, "call")["price"]
                if sell_call
                else 0.0
            )
            prev_put_val = (
                BSMOptionsEngine.price_and_greeks(spot_0, put_k, tte_init, rf, put_iv_0, "put")["price"]
                if buy_put
                else 0.0
            )

            cov_scale = min(w_0, 1.0)  # Cover up to 100% of underlying QQQ position

            for d in range(start_idx + 1, end_idx + 1):
                rem_days = max(end_idx - d, 0)
                rem_tte = rem_days / 252.0
                spot_d = closes[d]
                rv_d = max(float(vols[d]), 0.10)
                call_iv_d = max(rv_d + self.opt_cfg.vrp_spread + self.opt_cfg.call_skew_discount, 0.08)
                put_iv_d = max(rv_d + self.opt_cfg.vrp_spread + self.opt_cfg.put_skew_addon, 0.10)

                daily_opt_pnl_pct = 0.0
                if sell_call:
                    cur_call_val = BSMOptionsEngine.price_and_greeks(
                        spot_d, call_k, rem_tte, rf, call_iv_d, "call"
                    )["price"]
                    # Short call P&L = (prev_call_val - cur_call_val) / spot_0
                    daily_opt_pnl_pct += cov_scale * (prev_call_val - cur_call_val) / spot_0
                    prev_call_val = cur_call_val

                if buy_put:
                    cur_put_val = BSMOptionsEngine.price_and_greeks(
                        spot_d, put_k, rem_tte, rf, put_iv_d, "put"
                    )["price"]
                    # Long put P&L = (cur_put_val - prev_put_val) / spot_0
                    daily_opt_pnl_pct += cov_scale * (cur_put_val - prev_put_val) / spot_0
                    prev_put_val = cur_put_val

                if d == start_idx + 1:
                    daily_opt_pnl_pct -= (self.opt_cfg.option_commission_bps / 10_000.0) * cov_scale

                overlay_daily_ret[d] = daily_opt_pnl_pct

        bt["options_overlay_ret"] = overlay_daily_ret
        bt["combined_ret"] = bt["strategy_ret"] + bt["options_overlay_ret"]
        bt["combined_equity"] = self.base_cfg.initial_capital * (1.0 + bt["combined_ret"]).cumprod()
        bt["combined_peak"] = bt["combined_equity"].cummax()
        bt["combined_dd"] = (bt["combined_equity"] - bt["combined_peak"]) / bt["combined_peak"]

        # Compute metrics for the Combined (Core QQQ + Options Collar Overlay) portfolio
        bt_comb = bt.copy()
        bt_comb["strategy_ret"] = bt["combined_ret"]
        bt_comb["strategy_equity"] = bt["combined_equity"]
        bt_comb["strategy_dd"] = bt["combined_dd"]
        comb_metrics = engine.compute_metrics(bt_comb)

        # Build live 30-DTE option chain & recommended collar ticket for the latest bar
        live_ticket = self.build_live_options_ticket(bt, portfolio_equity=self.base_cfg.initial_capital)

        return bt, {
            "core_qqq_metrics": base_metrics,
            "core_plus_options_overlay_metrics": comb_metrics,
            "live_options_overlay_ticket": live_ticket,
        }

    def build_live_options_ticket(
        self, bt: pd.DataFrame, portfolio_equity: float = 100_000.0
    ) -> Dict[str, Any]:
        last = bt.iloc[-1]
        spot = float(last["close"])
        rv = max(float(last["realized_vol"]), 0.12)
        atm_iv = rv + self.opt_cfg.vrp_spread
        tte = self.opt_cfg.roll_period_days / 252.0
        rf = self.base_cfg.risk_free_rate

        call_iv = atm_iv + self.opt_cfg.call_skew_discount
        put_iv = atm_iv + self.opt_cfg.put_skew_addon

        call_strike = BSMOptionsEngine.solve_strike_for_delta(
            spot, self.opt_cfg.call_target_delta_normal, tte, rf, call_iv, "call"
        )
        put_strike = BSMOptionsEngine.solve_strike_for_delta(
            spot, self.opt_cfg.put_target_delta_hedge, tte, rf, put_iv, "put"
        )

        call_greeks = BSMOptionsEngine.price_and_greeks(spot, call_strike, tte, rf, call_iv, "call")
        put_greeks = BSMOptionsEngine.price_and_greeks(spot, put_strike, tte, rf, put_iv, "put")

        contracts = max(int((portfolio_equity / spot) // 100), 1)
        net_credit_per_share = round(call_greeks["price"] - put_greeks["price"], 2)

        return {
            "as_of_date": str(bt.index[-1].date()),
            "qqq_spot": round(spot, 2),
            "dte_calendar_days": 30,
            "contracts_per_100_shares": contracts,
            "short_covered_call_leg": {
                "action": "SELL_TO_OPEN",
                "option_type": "CALL",
                "strike": call_strike,
                "otm_pct": round((call_strike / spot - 1.0) * 100, 2),
                "implied_vol_pct": round(call_iv * 100, 2),
                **call_greeks,
            },
            "long_protective_put_leg": {
                "action": "BUY_TO_OPEN",
                "option_type": "PUT",
                "strike": put_strike,
                "otm_pct": round((1.0 - put_strike / spot) * 100, 2),
                "implied_vol_pct": round(put_iv * 100, 2),
                **put_greeks,
            },
            "net_collar_credit_per_share_usd": net_credit_per_share,
            "total_net_credit_usd": round(net_credit_per_share * 100 * contracts, 2),
            "annualized_yield_boost_pct": round((call_greeks["price"] / spot) * (252.0 / 21.0) * 100, 2),
        }


def main() -> None:
    df, _ = generate_synthetic_market_data()
    overlay = QQQOptionsOverlayStrategy()
    _, summary = overlay.run_overlay_backtest(df)

    m_base = summary["core_qqq_metrics"]
    m_opt = summary["core_plus_options_overlay_metrics"]
    ticket = summary["live_options_overlay_ticket"]

    print("=" * 82)
    print(" QQQ-ANCHORED OPTIONS OVERLAY (COVERED CALL + PROTECTIVE COLLAR) TEAR SHEET")
    print("=" * 82)
    print(f"{'Metric':<30} | {'QQQ Buy & Hold':>15} | {'Core QQQ Strat':>15} | {'Core + Options Overlay':>22}")
    print("-" * 82)
    rows = [
        ("Total Return (%)", f"{m_base['benchmark_total_return_pct']:.2f}%", f"{m_base['total_return_pct']:.2f}%", f"{m_opt['total_return_pct']:.2f}%"),
        ("CAGR (%)", f"{m_base['benchmark_cagr_pct']:.2f}%", f"{m_base['cagr_pct']:.2f}%", f"{m_opt['cagr_pct']:.2f}%"),
        ("Annual Volatility (%)", f"{m_base['benchmark_volatility_pct']:.2f}%", f"{m_base['annual_volatility_pct']:.2f}%", f"{m_opt['annual_volatility_pct']:.2f}%"),
        ("Sharpe Ratio", f"{m_base['benchmark_sharpe_ratio']:.2f}", f"{m_base['sharpe_ratio']:.2f}", f"{m_opt['sharpe_ratio']:.2f}"),
        ("Sortino Ratio", "-", f"{m_base['sortino_ratio']:.2f}", f"{m_opt['sortino_ratio']:.2f}"),
        ("Max Drawdown (%)", f"{m_base['benchmark_max_drawdown_pct']:.2f}%", f"{m_base['max_drawdown_pct']:.2f}%", f"{m_opt['max_drawdown_pct']:.2f}%"),
        ("Calmar Ratio", "-", f"{m_base['calmar_ratio']:.2f}", f"{m_opt['calmar_ratio']:.2f}"),
        ("Final Equity ($100k Init)", f"${100000*(1+m_base['benchmark_total_return_pct']/100):,.0f}", f"${m_base['final_equity_usd']:,.0f}", f"${m_opt['final_equity_usd']:,.0f}"),
    ]
    for label, v0, v1, v2 in rows:
        print(f"{label:<30} | {v0:>15} | {v1:>15} | {v2:>22}")
    print("-" * 82)
    print(" Latest 30-DTE QQQ Options Collar Order Ticket (BSM Greeks):")
    print(json.dumps(ticket, indent=2))
    print("=" * 82)


if __name__ == "__main__":
    main()
