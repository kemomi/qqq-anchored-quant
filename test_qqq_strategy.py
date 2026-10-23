"""
Automated Unit & Integration Test Suite for QQQ-Anchored Quant Strategy
=======================================================================
Author: kemomi
License: MIT

Run with:
    python3 test_qqq_strategy.py
"""

from __future__ import annotations

import unittest
import numpy as np

from qqq_quant_strategy import (
    StrategyConfig,
    QuantIndicators,
    QQQAnchoredStrategy,
    BacktestEngine,
    generate_synthetic_market_data,
)
try:
    from live_trader_execution import ExecutionConfig, PreTradeRiskGate, BrokerExecutionClient
except ModuleNotFoundError:
    import importlib.util as _ilu, pathlib as _pl, sys as _sys
    _spec = _ilu.spec_from_file_location("live_trader_execution", _pl.Path(__file__).resolve().parent / "live_trader_execution.py")
    _mod = _ilu.module_from_spec(_spec)
    _sys.modules["live_trader_execution"] = _mod
    _spec.loader.exec_module(_mod)
    ExecutionConfig, PreTradeRiskGate, BrokerExecutionClient = _mod.ExecutionConfig, _mod.PreTradeRiskGate, _mod.BrokerExecutionClient


class TestQQQQuantStrategySuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.qqq_df, cls.const_df = generate_synthetic_market_data(
            start_date="2020-01-01", end_date="2025-12-31", seed=123
        )
        cls.cfg = StrategyConfig()

    def test_01_indicators_bounds_and_integrity(self) -> None:
        """Verify RSI, Bollinger Bands, ATR, and ADX stay within mathematical bounds."""
        c = self.qqq_df["close"]
        rsi_2 = QuantIndicators.rsi(c, window=2)
        rsi_14 = QuantIndicators.rsi(c, window=14)
        self.assertTrue(((rsi_2 >= 0.0) & (rsi_2 <= 100.0)).all())
        self.assertTrue(((rsi_14 >= 0.0) & (rsi_14 <= 100.0)).all())

        mid, upper, lower, pct_b, bw = QuantIndicators.bollinger_bands(c, window=20, num_std=2.0)
        valid_idx = mid.dropna().index
        self.assertTrue((upper.loc[valid_idx] >= mid.loc[valid_idx]).all())
        self.assertTrue((mid.loc[valid_idx] >= lower.loc[valid_idx]).all())

        atr = QuantIndicators.atr(self.qqq_df, window=14).dropna()
        self.assertTrue((atr > 0.0).all())

        adx, plus_di, minus_di = QuantIndicators.adx(self.qqq_df, window=14)
        self.assertTrue(((adx >= 0.0) & (adx <= 100.0)).all())

    def test_02_zero_lookahead_bias(self) -> None:
        """Strict causality test: mutating future prices never alters past weights."""
        strat = QQQAnchoredStrategy(self.cfg)
        base_out = strat.generate_signals_and_weights(self.qqq_df)

        mutated_df = self.qqq_df.copy()
        cutoff = len(mutated_df) - 100
        mutated_df.iloc[cutoff:, mutated_df.columns.get_loc("close")] *= 0.50
        mutated_df.iloc[cutoff:, mutated_df.columns.get_loc("high")] *= 0.50
        mutated_df.iloc[cutoff:, mutated_df.columns.get_loc("low")] *= 0.50

        mutated_out = strat.generate_signals_and_weights(mutated_df)

        np.testing.assert_allclose(
            base_out["target_weight"].iloc[: cutoff + 1].to_numpy(),
            mutated_out["target_weight"].iloc[: cutoff + 1].to_numpy(),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_03_exposure_and_leverage_bounds(self) -> None:
        """Verify portfolio weights respect min_exposure, max_leverage, and unit sum."""
        strat = QQQAnchoredStrategy(self.cfg)
        out = strat.generate_signals_and_weights(self.qqq_df)

        self.assertTrue((out["target_weight"] >= self.cfg.min_exposure - 1e-9).all())
        self.assertTrue((out["target_weight"] <= self.cfg.max_leverage + 1e-9).all())

        total_etf_alloc = out["w_tqqq"] + out["w_qqq"] + out["w_sqqq"] + out["w_cash"]
        np.testing.assert_allclose(total_etf_alloc.to_numpy(), np.ones(len(out)), rtol=1e-6, atol=1e-6)

    def test_04_all_three_backtest_modes_execute(self) -> None:
        """Verify all 3 strategy modes execute cleanly and output complete metrics."""
        engine = BacktestEngine(self.cfg)
        for mode in ["core_qqq", "leveraged_etf", "constituent_rotation"]:
            bt_df, metrics = engine.run(
                self.qqq_df, constituents_df=self.const_df, strategy_type=mode
            )
            self.assertGreater(len(bt_df), 500)
            self.assertIn("sharpe_ratio", metrics)
            self.assertGreater(metrics["final_equity_usd"], 0.0)

    def test_05_pre_trade_risk_gate_and_dry_run_execution(self) -> None:
        """Verify PreTradeRiskGate caps oversized orders and orders SELLs before BUYs."""
        exec_cfg = ExecutionConfig(
            broker="dry_run",
            max_single_order_usd=25_000.0,
            min_drift_threshold=0.01,
        )
        gate = PreTradeRiskGate(exec_cfg)
        raw_orders = [
            {"symbol": "TQQQ", "side": "BUY", "notional_usd": 40_000.0, "target_weight": 0.40},
            {"symbol": "QQQ", "side": "SELL", "notional_usd": 30_000.0, "target_weight": 0.60},
        ]
        prices = {"TQQQ": 80.0, "QQQ": 500.0}
        validated = gate.filter_and_validate_orders(raw_orders, 100_000.0, prices)

        self.assertEqual(len(validated), 2)
        self.assertEqual(validated[0]["side"], "SELL")
        self.assertEqual(validated[1]["side"], "BUY")
        self.assertLessEqual(validated[0]["estimated_notional_usd"], 25_000.0)
        self.assertLessEqual(validated[1]["estimated_notional_usd"], 25_000.0)

        client = BrokerExecutionClient(exec_cfg)
        receipts = client.execute_orders(validated)
        self.assertEqual(len(receipts), 2)
        self.assertEqual(receipts[0]["status"], "DRY_RUN_SIMULATED_FILL")

    def test_06_bsm_put_call_parity_and_options_overlay(self) -> None:
        """Verify Black-Scholes-Merton put-call parity and Options Collar Overlay execution."""
        import importlib.util as _ilu, pathlib as _pl, math as _math
        _spec = _ilu.spec_from_file_location("qqq_options_overlay", _pl.Path(__file__).resolve().parent / "qqq_options_overlay.py")
        import sys as _sys
        _mod = _ilu.module_from_spec(_spec)
        _sys.modules["qqq_options_overlay"] = _mod
        _spec.loader.exec_module(_mod)
        BSM = _mod.BSMOptionsEngine
        spot, strike, tte, r, iv, q = 480.0, 500.0, 30.0 / 365.0, 0.04, 0.22, 0.006
        c = BSM.price_and_greeks(spot, strike, tte, r, iv, "call", q)
        p = BSM.price_and_greeks(spot, strike, tte, r, iv, "put", q)
        # Put-Call Parity: C - P = S * exp(-q*T) - K * exp(-r*T)
        lhs = c["price"] - p["price"]
        rhs = spot * _math.exp(-q * tte) - strike * _math.exp(-r * tte)
        self.assertAlmostEqual(lhs, rhs, places=3)

        overlay = _mod.QQQOptionsOverlayStrategy()
        _, summary = overlay.run_overlay_backtest(self.qqq_df)
        self.assertIn("core_plus_options_overlay_metrics", summary)
        self.assertGreater(summary["core_plus_options_overlay_metrics"]["sharpe_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
