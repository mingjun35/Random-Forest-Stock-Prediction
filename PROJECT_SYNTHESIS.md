# Project Synthesis

The full story of what I built, what broke, and what I learned.

## What I set out to do

Build a Random Forest classifier that predicts next-week price direction for S&P 500 stocks using standard technical indicators (RSI, MACD, ATR, Bollinger, Stochastic). My expectation going in was that with careful feature engineering, I'd land on something that beats SPY by a few percent per year.

That's not what happened.

## How the project evolved

I went through roughly five rounds of iteration.

**Round 1 — the original build.** Scraped S&P 500 prices, computed 15 technical indicators, clustered tickers into groups, trained a Random Forest per cluster with GridSearchCV, backtested the top-10 predictions with a fixed 7-day hold. Hit rates in the 65-75% range on the test set. Looked great. I nearly stopped here.

**Round 2 — the rebuild and bug hunt.** Rewriting everything with proper time-series methodology turned up multiple problems. The Stochastic %D indicator was being computed without a `groupby('Ticker')` call, which meant rolling means leaked across ticker boundaries — the last five rows of AAPL polluted the first five of MSFT. Subtle but real contamination inflating CV AUC. The target formula compared future close to today's open, but my indicators used today's close, which is a small implicit lookahead. The train/test split used `iloc[:split_index]`, which can cut through the middle of a date with 500 stocks per day. The data scraper had `start="2026-04-01"` hardcoded. I fixed all of these, switched to LightGBM with proper calibration, added purged splits with a 6-day embargo matching the target horizon, and built walk-forward validation across three historical periods.

The honest CV AUC after fixes was 0.51 — essentially random. Walk-forward mean Sharpe was 1.06 but with huge variance across folds.

**Round 3 — feature engineering.** Added sector-relative momentum, VIX, and cross-sectional ranks. CV AUC moved from 0.51 to 0.53. VIX dominated feature importance but didn't move actual predictive power meaningfully. The information was partially redundant with what the regime model already captured.

**Round 4 — risk management.** Instead of trying to extract more alpha from the model, I shifted to execution. Added ATR-based stop-loss, take-profit, trailing stop, time stop, and lost-conviction exits. Single-window Sharpe jumped from 1.42 to 1.91, max drawdown halved. Then tried confluence filters (only trade when multiple signals agree) — and in walk-forward this made the bear market fold go from -2.4% to -22%. The filter kept selecting bull-trap setups during the 2021-22 bear.

**Round 5 — alternative approaches.** Scraped SEC EDGAR Form 4 insider transactions as alternative data. CV AUC stayed at 0.53. Tried Post-Earnings Announcement Drift (PEAD) as a completely different strategy — 70% hit rate but overall Sharpe of 0.31 because of opportunity cost versus a bull market SPY was crushing. Walk-forward mean Sharpe for PEAD was 0.04.

At this point the conclusion was unambiguous.

## Technical lessons

The two most impactful bugs weren't dramatic — they were silent. They inflated CV AUC by about 9 percentage points. If I'd deployed the pre-fix model to real money, I'd have been trading on phantom signals. In financial ML, assume every impressive result is leakage until you've systematically audited for it.

Walk-forward validation is non-negotiable. The single test window showed +28% CAGR with Sharpe 1.91. The walk-forward across three historical periods showed mean Sharpe 0.70 with one fold losing 22%. Any backtest result from one time window is almost worthless.

Model choice matters less than I thought. Random Forest to LightGBM was about 1 pp AUC improvement. LightGBM to deep learning is documented to be another 1 pp. The ceiling is informational, not architectural.

Calibration matters when you threshold on probabilities. Raw probabilities from tree models aren't really probabilities. If you want "p(up) > 0.70 means 70% historical hit rate," you need isotonic regression. Without it, your threshold tuning is bogus.

Class imbalance needs explicit handling. Bull markets have more up days than down days. Without `scale_pos_weight` or `class_weight='balanced'`, models silently learn "always predict up."

## Market structure lessons

Market efficiency is real in a practical sense, not just an academic theory. After spending months trying to find alpha and failing, you internalize that the market is efficient enough that retail-accessible features on liquid stocks don't contain extractable signal.

Alpha decays as information spreads. The strategies that worked in academic papers from the 1990s have decayed to near-zero excess returns in the 2020s. Once published, they're arbitraged. Factor ETFs commoditized the easy wins.

The surviving edges live in capacity-limited corners big funds can't fit into: small/mid caps, illiquid names, short-dated specific options, specific event windows. These remain accessible to retail by nature of being too small for institutions.

Opportunity cost of not being long is often higher than you think. The 2018-2026 window was one of the strongest 8-year equity returns in history. Any active strategy that keeps capital in cash periodically gives up massive opportunity cost that no amount of alpha-generation makes up for.

## What I'd do differently

Test for signal before building infrastructure. A 5-line correlation test between features and forward returns would have shown the ceiling was ~0.52 in an afternoon. Instead I spent weeks building pipeline, then discovered the ceiling.

Start with logistic regression as baseline. It's a one-liner, trains in seconds, and tells you immediately if there's signal above random.

Benchmark against SPY from day one. I didn't start comparing to buy-and-hold until Step 6c. If I had, I'd have realized by Step 2 that even my best variants were underperforming the index.

Use purged time-split and walk-forward from the start. I used naive splitting initially. By the time I rewrote everything with proper validation, I had to throw out weeks of apparently-promising results.

Pivot to a different problem sooner. I spent too long trying to fix the directional prediction problem. The right pivot was earlier: cross-sectional ranking instead of binary classification, or shift to small/mid cap universe, or move to vol-selling entirely.

## Where retail alpha actually lives

Based on everything learned, my honest tier list from most to least promising for retail:

**Volatility risk premium (selling cash-secured puts on quality stocks).** Insurance premiums are structural. Realistic Sharpe 1.0-1.3. Scales to retail account sizes.

**Index investing with factor tilts.** Not alpha but captures risk premia at near-zero cost. Realistic edge over market 1-3% per year. Unlimited capacity.

**Sector rotation (ETF momentum).** Slow institutional flows plus behavioral under/over-reaction. Realistic Sharpe 0.7-0.9.

**Short squeeze hunting on Russell 2000 small caps.** Mechanical short-covering is real. Capacity-limited to small caps. High variance.

**Post-earnings drift on small/mid caps.** Behavioral underreaction, less arbitraged than large caps. PEAD on S&P 500 didn't work in my tests but small-cap PEAD may still have edge.

**Insider cluster buying.** Insiders know things. Signal is weak but real.

Things to probably avoid: options spreads (positive expected value doesn't help if a tail event wipes you out), crypto momentum (regulatory risk + extreme volatility), and the thing I spent months on — technical indicator ML on daily S&P 500 equities, which is empirically dead.

## Personal takeaway

Six things I now believe that I didn't before this project:

The hardest skill in quant isn't modeling — it's falsifying your own hypothesis. I spent weeks believing I had a +28% CAGR strategy. Walk-forward told me it was mostly luck. Most retail traders never do walk-forward and never find out.

Negative results are more valuable than positive ones when properly validated. A rigorously validated "this doesn't work" is a permanent protection against future wasted effort. A positive result from one backtest is a trap.

The simplest strategy — buy VTI, hold, contribute monthly — beats 95% of what I can build. This isn't defeat. It's realism. The market is efficient enough that trying to outsmart it is a losing proposition for retail.

Information edge beats modeling edge. If 10,000 people have the same features and the same models, nobody has edge. The surviving edges live in information other people don't have.

Capacity constraints are features, not bugs. The reason small-cap PEAD might still work is that it's too small for big funds. If a strategy is both easy to find AND scales to billions, it's already arbitraged.

The best thing I built isn't a model — it's an experimental method. The pipeline, the validation discipline, the walk-forward tests, the willingness to accept negative results. The specific model is just one application of them.

## Closing

If someone reads this project and takes away "machine learning on stocks doesn't work," they've missed the point.

The point is: ML on this specific problem (daily directional prediction of S&P 500 stocks from technical indicators) doesn't work, because the information has been arbitraged away. ML works brilliantly in domains where the information hasn't been arbitraged away.

Going forward I'm putting the bulk of my capital in VTI with monthly contributions, learning options income strategies for an income sleeve, keeping a small speculation sleeve for ongoing experimentation, and carrying the methodological discipline from this project into any future quantitative work.

The project was never going to make me rich. It was always going to make me wiser about what's possible. On that measure, it succeeded.

— MJ Lee, 2026
