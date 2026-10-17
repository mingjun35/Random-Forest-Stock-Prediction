# Results Summary

Headline numbers for every variant tested. See `PROJECT_SYNTHESIS.md` for context.

## 1. Single-window backtest (Aug 2024 — Apr 2026)

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

### What each variant did

**SPY buy-and-hold** — Do nothing. Buy the index, sit on it. +17.6%/yr. This is the thing to beat.

**Step 6b (fixed 7-day hold)** — Model says buy → hold exactly 7 days → sell no matter what. Mechanical. 1,814 trades, won 59% of the time.

**Step 6b (threshold 0.75)** — Same as above but only trade when model is at least 75% confident. Pickier = fewer trades, better hit rate.

**Step 6b (threshold 0.80)** — Even pickier (≥80% confidence). Only 474 trades, +59% per year. Looks amazing but it's basically 3 lucky mega-winners — walk-forward later confirmed this.

**Step 6d (flexible exit)** — Dumped the fixed 7-day rule. Instead exits when any of: stop-loss hits, take-profit hits, trailing stop hits, model changes its mind, or 14-day timer runs out. Returns dipped slightly but drawdown halved.

**Step 6d (tuned stops)** — Same flexible exit but with tighter stop-losses and wider trailing stops. Best risk-adjusted result (Sharpe 1.91).

**Step 6e (confluence, 5 filters)** — Only trade when the model AND market trend AND stock trend AND volatility AND RSI all agree. Backfired. The filter kept selecting false signals.

**Step 6e (confluence, 2 filters)** — Simplified to 2 filters (market + stock trend). Still worse. The confluence idea itself was flawed, not the number of filters.

**The catch:** this window was a straight-up bull market. Any trend-following strategy was going to look good. That's why Section 2 matters.

---

## 2. Walk-forward across 3 historical folds

Instead of one time window, test across 3 different historical periods including a bear market. If a strategy only works in bull markets, this exposes it.

Folds:
- Fold 1: train 2018-01 to 2021-06, test 2021-07 to 2022-12 (bear market period)
- Fold 2: train 2019-01 to 2022-12, test 2023-01 to 2024-06 (bull recovery)
- Fold 3: train 2020-01 to 2023-12, test 2024-01 to 2024-08 (bull continuation)

| Variant | Mean return | Mean CAGR | Mean Sharpe | Worst fold | Folds positive |
|---|---|---|---|---|---|
| Step 7 (base) | +8.6% | +10.2% | 0.55 | -13.1% | 2/3 |
| Step 7d (flexible exit) | +14.2% | +4.8% | 0.70 | -2.4% | 2/3 |
| Step 7e (confluence filter) | +1.7% | -4.6% | 0.70 | -21.8% | 2/3 |

### What each variant did

**Step 7 (base)** — The 7-day fixed-hold strategy from Section 1, retested properly across 3 periods. Mean Sharpe 0.55, worse than SPY's 0.75. The single-window outstanding results disappeared.

**Step 7d (flexible exit)** — The best single-window performer (Step 6d), retested properly. Mean Sharpe 0.70 — still worse than SPY. Its one virtue: worst fold only lost 2.4%, so good risk control even when returns stink.

**Step 7e (confluence filter)** — Mean Sharpe 0.70 looks okay, but worst fold lost 21.8%. The filter made the bear market worse. During 2021-22, "all signals agree" setups were bull-trap patterns that got crushed.

**In plain English:** every strategy that looked great in Section 1 fell apart when tested across different market conditions. The single-window numbers were influenced by the bull-market, not a real edge.

---

## 3. PEAD alternative strategy (Step 9)

PEAD = "Post-Earnings Announcement Drift." Different idea entirely: when a company reports surprisingly good earnings, the stock tends to keep drifting up for days/weeks afterward (slow investor reaction). Trade that drift instead of trying to predict daily moves from indicators.

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

### What this means

The 70% hit rate proves the PEAD signal is real — 7 out of 10 trades win, with wins averaging twice the size of losses.

So why did it still lose to SPY by 157%? **Opportunity cost.** PEAD only trades right after earnings (a few trades per stock per year). The rest of the time, your capital sits in cash. During an 8-year bull market, sitting in cash is devastating even if your actual trades win.

The lesson: a strategy can "work" in the sense of being right most of the time, and still lose badly if being out of the market during a bull run. It shows opportunity costs far outweighs clever gains. Time in the market is always better than timing the market.

---

## 4. CV AUC progression

AUC (Area under the Curve) = how well the model ranks winners over losers. 0.50 is a coin flip. 0.60 is strong. For stocks, anything above 0.55 is considered real signal.

| Features | CV AUC |
|---|---|
| Pre-bug-fix (contaminated) | 0.60 (fake) |
| Honest baseline (15 technical indicators) | 0.51 |
| + Regime conditioning | 0.52 |
| + Sector/VIX/rank features | 0.53 |
| + SEC Form 4 insider counts | 0.53 |

### What this means

- **0.60 (fake)** — The original "promising" result. Turned out to be the Stochastic %D leakage bug (one indicator leaking data across tickers). The entire 0.60 was artificial.
- **0.51** — Honest baseline after bug fixes. Barely above a coin flip.
- **0.52** — Added market regime awareness. Tiny bump.
- **0.53** — Added sector momentum, VIX, cross-sectional ranks. Tiny bump.
- **0.53** — Added SEC Form 4 insider transactions. No bump at all.

The ceiling at ~0.53 isn't a modeling problem — switching to deep learning won't push it meaningfully higher. It's an information problem. The features themselves just don't contain much predictive signal. The information has been arbitraged away by professionals who got to it first.

---

## 5. Transaction cost sensitivity

How much do broker fees eat into returns? Realistic retail cost is 5-10 basis points per trade.

| Variant | 0 bps | 5 bps | 10 bps | 20 bps |
|---|---|---|---|---|
| Step 6d (tuned) | +63.7% | +57.7% | +51.8% | +39.9% |
| Step 6e (confluence) | +25.1% | +23.4% | +21.7% | +18.4% |
| PEAD (long holds) | +50.0% | +49.0% | +48.0% | +46.1% |

### What this means

- **Step 6d (daily trading):** +63.7% at zero cost → +39.9% at 20 bps. Loses 25% of its edge to costs. High turnover gets killed by fees.
- **PEAD (longer holds):** +50% → +46.1%. Barely affected by fees. Longer holding periods are naturally cost-robust because you're not paying the toll as often.

Takeaway: daily trading strategies lose a lot to friction. Longer-horizon strategies survive it. High-turnover daily strategies lose 10-25% of edge at realistic broker costs.

---

## Bottom line

| Question | Answer |
|---|---|
| Does any variant beat SPY on walk-forward? | No |
| Does any variant beat SPY on cherry-picked windows? | Yes, but that caveat matters |
| Is CV AUC of 0.53 enough to overcome costs? | No |
| Would more features help? | Probably not (ceiling is informational) |
| Would deep learning help? | Marginally if at all |
| What's the honest conclusion? | Buy ETFs, add factor tilts, use capacity-limited active strategies if you want to be active |

## Reproducibility

All results reproducible by running scripts in order starting with Step 1b. Pipeline is seeded (`random_state=42`) for determinism. Exact numbers may vary slightly based on yfinance data refresh dates and library versions.
