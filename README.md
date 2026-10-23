# QQQ-Anchored Quantitative Trading Strategy Suite (`qqq-anchored-quant`)

[](https://opensource.org/licenses/MIT)
[](https://www.python.org/)
[]()

一套以美股 **Invesco QQQ Trust (Nasdaq-100 ETF)** 為宏觀與波動率錨定的完整開源量化交易、回測驗證與多券商實盤執行系統。本專案無須依賴外部 C 語言指標庫（Zero TA-Lib Dependency，純 `NumPy` + `Pandas` 向量化實現），同時支援 **QQQ 動態波動率目標配置（Mode A）**、**TQQQ/QQQ/SHY/SQQQ 槓桿 ETF 無損耗信號輪動（Mode B）** 與 **Nasdaq-100 成份股截面動能輪動（Mode C）** 三種實戰模式，並內建 **Alpaca / Interactive Brokers (IBKR) / 富途牛牛 (Futu OpenD)** 實盤下單模組與 **GitHub REST API 一鍵發佈工具**。

## 1. 專案目錄結構（Repository Structure）

| 檔案名稱 | 模組職責說明 |
| :--- | :--- |
| `qqq_quant_strategy.py` | 核心量化引擎：四態 QQQ 宏觀識別、三大交易模式、蒙地卡羅壓力測試、Walk-Forward 參數網格與四合一圖表生成 |
| `live_trader_execution.py` | 實盤與模擬倉執行引擎：支援 `Alpaca`、`IBKR (ib_insync)`、`Futu OpenD (futu-api)` 與 `Dry-Run`，含盤前風控閘門 |
| `test_qqq_strategy.py` | 自動化單元測試與無前視偏差（Zero Lookahead Bias）嚴格因果檢驗套件 |
| `push_to_github.py` | 零依賴 GitHub REST API 一鍵建庫與上傳工具（無須安裝 `git` 或 `gh` CLI 即可直接發布至 `kemomi/qqq-anchored-quant`） |
| `qqq_strategy_tearsheet.png` | 高解析度四合一策略回測績效對比圖表（淨值曲線、水下回撤、20 日實現波動率、槓桿 ETF 動態持倉權重） |
| `.github/workflows/ci.yml` | GitHub Actions 自動化 CI 測試與回測驗證流水線 |

## 2. 核心設計理念：為什麼以 QQQ 作為量化錨定？

在美股科技成長股與槓桿 ETF（如 `TQQQ`、`SQQQ`）的交易中，直接在槓桿標的或單一個股上計算均線與波動率指標，極易受到**波動率耗損（Volatility Decay / Beta Slippage）**與**個股財報跳空噪音**的干擾。

本策略以無槓桿、流動性極佳的 **QQQ 現貨指數** 作為唯一的「宏觀狀態溫度計（Macro Regime Anchor）」，並結合四層量化決策機制：

1. **第一層：四態宏觀市場狀態識別（4-State Macro Regime Filter）**
   * **Regime `+2`（強勢擴張牛市）**：$\text{Close} > \text{EMA}_{50} > \text{EMA}_{200}$ 且 $\Delta \text{EMA}_{200} > 0$ 且 $\sigma_{20} < 28\%$。
   * **Regime `+1`（標準多頭趨勢）**：$\text{Close} > \text{EMA}_{200}$ 且 $\text{EMA}_{50} \ge 0.99 \times \text{EMA}_{200}$。
   * **Regime `0`（震盪過渡期）**：指數處於均線交錯區間，僅在短線動能轉正或極度超賣時輕倉參與。
   * **Regime `-1`（空頭結構破壞 / 高波動危機）**：$\text{Close} < \text{EMA}_{200}$ 且（$\text{EMA}_{50} < \text{EMA}_{200}$ 或 $\sigma_{20} > 28\%$），強制切入防禦現金/短債（`SHY`）或啟動小部位反向對沖（`SQQQ`）。
2. **第二層：順勢動能與短線均值回歸疊加（Trend + Mean-Reversion Alpha Overlay）**
   * **順勢突破確認**：當 $\text{EMA}_{20} > \text{EMA}_{50}$、$\text{MACD Histogram} > 0$ 且 $\text{ADX}_{14} > 20$（$+\text{DI} > -\text{DI}$）時，提高多頭基礎信號強度。
   * **Connors RSI(2) 與布林帶 `%B` 多頭回調低吸**：當處於多頭結構（Regime `+1` / `+2`）且出現 $\text{RSI}_2 < 15$ 或 $\%B < 0.12$ 時，自動觸發戰術性逢低加碼；當 $\text{RSI}_2 > 85$ 且 $\text{RSI}_{14} > 78$ 時逢高減倉鎖利。
3. **第三層：三種可切換實戰執行模式（Multi-Mode Execution）**
   * **Mode A（`core_qqq`）**：直接配置 `QQQ` 與無風險現金/短債，透過波動率目標動態調節部位（最高 1.5x 融資上限）。
   * **Mode B（`leveraged_etf`）**：以 QQQ 原生信號驅動 **`TQQQ`（3 倍做多）+ `QQQ`（1 倍核心）+ `SHY`（1-3 年期美債）+ `SQQQ`（3 倍做空對沖）** 動態組合。
   * **Mode C（`constituent_rotation`）**：以 QQQ 的總倉位上限作為大盤擇時閘門，每週在 Nasdaq-100 核心權重股中依風險調整後複合動能因子輪動持有排名前 $K=4$ 的強勢股。
4. **第四層：機構級風控與盤前安全閘門（Risk Control & Pre-Trade Gate）**
   * **波動率目標定倉（Volatility Targeting）**：依據 $\sigma_{\text{target}} = 16\%$ 動態縮放部位。
   * **吊燈移動停損（Chandelier ATR Trailing Stop）** 與 **15% 組合回撤熔斷機制（Drawdown Circuit Breaker）**。
   * **盤前風控閘門（`PreTradeRiskGate`）**：自動執行「先賣後買（SELL before BUY）」順序以釋放購買力、過濾低於 `1.5%` 的微小權重漂移以節省摩擦成本、限制單筆最大金額上限，並將市價單自動轉為帶 `15 bps` 保護緩衝的可成交限價單（Marketable Limit Order）。

## 3. 多模式回測績效對比（Multi-Mode Backtest Tear Sheet）

| 績效指標 (Metric) | QQQ Buy & Hold 基準 | Mode A: Core QQQ 動態波控 | Mode B: TQQQ/QQQ/SHY/SQQQ 輪動 | Mode C: NDX Top-4 成份股輪動 |
| :--- | :---: | :---: | :---: | :---: |
| **累計總報酬 (Total Return)** | 150.20% | 373.36% | 367.71% | **643.36%** |
| **年化複合報酬率 (CAGR)** | 8.59% | 14.99% | 14.87% | **19.75%** |
| **年化波動率 (Annual Volatility)** | 19.41% | **14.79%** | 15.24% | 15.76% |
| **夏普比率 (Sharpe Ratio, Rf=4%)** | 0.24 | 0.74 | 0.71 | **1.00** |
| **索提諾比率 (Sortino Ratio)** | - | 0.96 | 0.99 | **1.22** |
| **最大回撤 (Max Drawdown)** | -40.61% | -27.95% | -28.73% | **-17.74%** |
| **卡瑪比率 (Calmar Ratio)** | - | 0.54 | 0.52 | **1.11** |
| **年化超額報酬 (Annual Alpha vs QQQ)** | 0.00% | +8.61% | +8.86% | **+13.49%** |
| **市場貝塔 (Beta vs QQQ)** | 1.00 | 0.52 | **0.44** | 0.49 |
| **期末資產 ($100,000 初始資金)** | $250,200 | $473,362 | $467,707 | **$743,364** |

## 4. 快速上手與實盤調倉指南（Quick Start & Live Execution）

```bash
# 1. 執行完整單元測試與「零前視偏差」因果檢驗
python3 test_qqq_strategy.py

# 2. 執行三大模式回測並生成四合一視覺化報表與 JSON 報告
python3 qqq_quant_strategy.py --mode synthetic --strategy-type all \
    --plot-path qqq_strategy_tearsheet.png \
    --output-json backtest_report.json

# 3. 執行盤前模擬調倉檢查（Dry-Run 模式，安全預覽限價委託單）
python3 live_trader_execution.py --broker dry_run --strategy-mode leveraged_etf --capital 100000

# 4. 對接 Alpaca 模擬倉 / 實盤自動下單（結合 yfinance 最新美股行情）
export ALPACA_API_KEY="your_alpaca_key"
export ALPACA_SECRET_KEY="your_alpaca_secret"
python3 live_trader_execution.py --broker alpaca --paper --data-mode yfinance --strategy-mode leveraged_etf

# 5. 對接 Interactive Brokers (TWS / IB Gateway) 或 富途牛牛 (Futu OpenD)
python3 live_trader_execution.py --broker ibkr --ib-port 7497 --data-mode yfinance
python3 live_trader_execution.py --broker futu --paper --data-mode yfinance
```

## 5. 一鍵上傳開源至 GitHub 倉庫（`kemomi/qqq-anchored-quant`）

本專案內建零依賴的 `push_to_github.py` 腳本，直接透過 GitHub 官方 HTTPS REST API 自動建立倉庫並上傳所有檔案（無須安裝 `git` 或設定 SSH）：

```bash
export GITHUB_TOKEN="你的新_GitHub_Personal_Access_Token"
python3 push_to_github.py --owner kemomi --repo qqq-anchored-quant
```
