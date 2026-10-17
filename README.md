# S&P 500 ML Investigation

I built an ML pipeline to predict next-week price direction for S&P 500 stocks, tested it thoroughly, and concluded it doesn't work. The negative result is the finding — technical-indicator ML on daily S&P 500 data has been arbitraged away.

## Headline numbers

| Strategy | 8-year total return | CAGR | Sharpe | Max DD |
|---|---|---|---|---|
| SPY buy-and-hold | +204% | ~15% | ~0.75 | -24% |
| Best ML variant (single window) | +52% | +28.7% | 1.91 | -9.6% |
| ML walk-forward mean (3 folds) | +14% | +4.8% | 0.70 | -31.4% |
| PEAD alternative strategy | +47% | +4.8% | 0.31 | -52.5% |

Every variant underperformed buy-and-hold. The single-window "+28% CAGR" dissolved to 0.70 Sharpe under walk-forward validation.

## What's in here

- 16 Python scripts covering data pipeline, feature engineering, regime detection, model training, backtesting, walk-forward validation, paper trading, and one alternative strategy (PEAD)
- Full project writeup in `PROJECT_SYNTHESIS.md`
- Results tables in `RESULTS_SUMMARY.md`
- Technical glossary in `GLOSSARY.md`

## Pipeline order

```
scripts/
├── Step 1b — Data scraping (S&P 500 OHLCV + sectors)
├── Step 1d — SEC Form 4 insider transactions (alt data)
├── Step 2b — Technical indicators (fixed, vectorized)
├── Step 2c — Enhanced features (sector momentum, VIX, cross-sectional ranks)
├── Step 3b — HMM regime clustering
├── Step 4b — Purged train/test split with embargo
├── Step 5b — LightGBM training (calibrated, threshold-optimized)
├── Step 6b — Backtest (per-regime threshold)
├── Step 6c — Backtest diagnostics (SPY benchmark, cost sensitivity)
├── Step 6d — Backtest (flexible exit)
├── Step 6e — Backtest (confluence filter + flexible exit)
├── Step 7  — Walk-forward validation (base)
├── Step 7d — Walk-forward validation (flexible exit)
├── Step 7e — Walk-forward validation (confluence filter)
├── Step 8  — Paper trading daily runner (Alpaca API)
└── Step 9  — PEAD strategy (event-driven alternative)
```

Run in order. First run of Step 1b takes ~15 minutes (data download). Walk-forward steps take 30-60 minutes.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.11, tested on macOS.

## Key bugs I found and fixed

- Stochastic %D was computed without `groupby('Ticker')` — cross-ticker data leakage inflating CV AUC by ~3-5 pp
- Target used today's open as reference while features included today's close — subtle lookahead
- Row-based train/test split (`iloc[:split]`) could split a date across train and test
- Download start date was hardcoded, pipeline silently broke over time

After fixes, honest CV AUC is ~0.51. Adding more features (sector, VIX, ranks, insider data) couldn't push it past ~0.53. That ceiling is informational, not architectural.

## What I'd use instead

For retail capital at this scale, buy-and-hold VTI with dollar-cost averaging beats every active variant I tested. The capacity-limited active strategies that still work (volatility selling on quality stocks, small-cap event-driven) require different infrastructure than this project.

## License

MIT. Build on it however you want.

— MJ Lee (leemingjun12@gmail.com)
