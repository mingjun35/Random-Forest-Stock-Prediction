# Results Summary

Headline numbers for every variant tested. See `PROJECT_SYNTHESIS.md` for context.

## Single-window backtest (Aug 2024 — Apr 2026)

| Variant | Trades | Hit rate | CAGR | Sharpe | Max DD | Alpha vs SPY |
|---|---|---|---|---|---|---|
| SPY buy-and-hold | — | — | +17.6% | 0.72 | -18.8% | 0 |
| Step 6b (fixed 7-day) | 1,814 | 59.2% | +27.6% | 1.42 | -12.2% | +10% |
| Step 6b (threshold 0.75) | 1,140 | 62.6% | +32.1% | 1.74 | -12.9% | +14.5% |
| Step 6b (threshold 0.80) | 474 | 66.0% | +59.1% | 2.35 | -12.9% | +41.5% |
| Step 6d (flexible exit) | 1,818 | 52.4% | +23.3% | 1.60 | -9.4% | +5.7% |
| Step 6d (tuned stops) | 1,905 | 50.2% | +28.7% | 1.91 | -9.6% | +11.1% |
| Step 6e (confluence, 5 filters) | 1,084 | 44.8% | +12.9% | 1.23 | -7.6% | -4.7% |
| Step 6e (confluence, 2 filters) | 1,189 | 47.3% | +14.6% | 1.24 | -7.3% | -3.0% |

Single-window results look promising for several variants. The problem is that this specific window was a strong bull market.

## Walk-forward across 3 historical folds

Folds:
- Fold 1: train 2018-01 to 2021-06, test 2021-07 to 2022-12 (bear market period)
- Fold 2: train 2019-01 to 2022-12, test 2023-01 to 2024-06 (bull recovery)
- Fold 3: train 2020-01 to 2023-12, test 2024-01 to 2024-08 (bull continuation)

| Variant | Mean return | Mean CAGR | Mean Sharpe | Worst fold | Folds positive |
|---|---|---|---|---|---|
| Step 7 (base) | +8.6% | +10.2% | 0.55 | -13.1% | 2/3 |
| Step 7d (flexible exit) | +14.2% | +4.8% | 0.70 | -2.4% | 2/3 |
| Step 7e (confluence filter) | +1.7% | -4.6% | 0.70 | -21.8% | 2/3 |

Confluence filters made things dramatically worse in the bear market fold. The filter kept selecting bull-trap setups during 2021-22.

## PEAD alternative strategy (Step 9)

| Metric | Value |
|---|---|
| Hit rate | 70.1% |
| Avg win | +16.5% |
| Avg loss | -8.4% |
| Profit factor | 1.30 |
| Total return (8 years) | +47.5% |
| CAGR | +4.8% |
| Sharpe | 0.31 |
| Max drawdown | -52.5% |
| SPY same window | +204.6% |
| Alpha vs SPY | -157.2% |

PEAD walk-forward: Fold 1 -16.6%, Fold 2 +3.4%, Fold 3 +6.5%. Mean Sharpe 0.04.

The 70% hit rate shows the PEAD signal is real. The opportunity cost of being out of the market during a bull run was devastating.

## CV AUC progression

| Features | CV AUC |
|---|---|
| Pre-bug-fix (contaminated) | 0.60 (fake) |
| Honest baseline (15 technical indicators) | 0.51 |
| + Regime conditioning | 0.52 |
| + Sector/VIX/rank features | 0.53 |
| + SEC Form 4 insider counts | 0.53 |

The ceiling of ~0.53 is informational, not architectural. No model class — RF, LightGBM, deep learning — would push this meaningfully higher on this feature set.

## Transaction cost sensitivity

| Variant | 0 bps | 5 bps | 10 bps | 20 bps |
|---|---|---|---|---|
| Step 6d (tuned) | +63.7% | +57.7% | +51.8% | +39.9% |
| Step 6e (confluence) | +25.1% | +23.4% | +21.7% | +18.4% |
| PEAD (long holds) | +50.0% | +49.0% | +48.0% | +46.1% |

Longer holding periods (PEAD) are more cost-robust. High-turnover daily strategies lose 10-25% of edge at realistic broker costs.

## Bottom line

| Question | Answer |
|---|---|
| Does any variant beat SPY on walk-forward? | No |
| Does any variant beat SPY on cherry-picked windows? | Yes, but that caveat matters |
| Is CV AUC of 0.53 enough to overcome costs? | No |
| Would more features help? | Probably not (ceiling is informational) |
| Would deep learning help? | Marginally if at all |
| What's the honest conclusion? | Buy VTI, add factor tilts, use capacity-limited active strategies if you want to be active |

## Reproducibility

All results reproducible by running scripts in order starting with Step 1b. Pipeline is seeded (`random_state=42`) for determinism. Exact numbers may vary slightly based on yfinance data refresh dates and library versions.
