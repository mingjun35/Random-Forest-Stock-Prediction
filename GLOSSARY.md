# Glossary

Quick definitions for everything used in this project.

## Models

| Term | What it is |
|---|---|
| Random Forest (RF) | Many independent decision trees averaged together |
| LightGBM | Gradient boosting — each tree corrects the previous tree's errors. Faster and more accurate than RF on tabular data |
| XGBoost | Older cousin of LightGBM, similar performance, slower |
| Logistic Regression | Simplest ML baseline. If LightGBM barely beats it, features are weak |
| HMM (Hidden Markov Model) | Assumes the market is in one of K hidden states (bull/range/bear) with transitions. Captures regime persistence |
| GMM (Gaussian Mixture Model) | Clusters data as Gaussian blobs. No time-awareness |
| K-Means | Fastest clustering, assumes spherical clusters |

## Technical Indicators

| Term | What it is |
|---|---|
| OHLCV | Open, High, Low, Close, Volume |
| SMA | Simple moving average over N days |
| EMA | Exponential moving average, weighted toward recent prices |
| SMA_ratio | Short SMA ÷ long SMA. >1 = uptrend |
| RSI | 0-100 momentum oscillator. >70 overbought, <30 oversold |
| MACD | Short EMA minus long EMA. Trend direction |
| ATR | Average true range — typical daily movement. Used for stops |
| ADX | Trend strength (0-100). >25 = strong trend |
| Bollinger Bands | Price ± 2 standard deviations |
| %B | Position within Bollinger bands (0 to 1) |
| Stochastic | Current price vs recent high/low range |
| %K / %D | Fast/slow stochastic lines |
| Wilder's Smoothing | EMA with alpha = 1/N, used in RSI/ADX/ATR |

## Prediction Target Terms

| Term | What it is |
|---|---|
| Target variable | What the model predicts (here: price up/down in 6 days) |
| Binary classification | Target has 2 classes |
| Target leakage | Target info sneaks into features |
| Lookahead bias | Features include info unavailable at prediction time |
| Horizon | How far ahead you're predicting |
| Class imbalance | More of one class than another |

## Model Parameters

| Parameter | What it controls |
|---|---|
| n_estimators | Number of trees |
| max_depth | How deep each tree grows |
| num_leaves | LightGBM equivalent of max_depth |
| min_samples_leaf | Minimum rows in a leaf (RF) |
| min_child_samples | Same for LightGBM |
| learning_rate | How much each boosting tree contributes |
| max_features | Features considered per split |
| subsample / bagging_fraction | % of rows per tree |
| colsample_bytree / feature_fraction | % of columns per tree |
| reg_alpha / reg_lambda | L1 / L2 regularization strength |
| scale_pos_weight | LightGBM class imbalance weight |
| class_weight='balanced' | RF equivalent |

## Validation

| Term | What it is |
|---|---|
| Cross-validation (CV) | Split data into folds, train on some, test on others |
| TimeSeriesSplit | CV where test data is always after train data |
| Purged split | Drop training rows whose target depends on test period |
| Embargo | Gap between train and test matching target horizon |
| Walk-forward | Train on rolling window, test on next window, advance, repeat |
| Fold | One train/test window pair |

## Model Metrics

| Term | What it is | Good value |
|---|---|---|
| Accuracy | % correct | > baseline rate |
| AUC (ROC-AUC) | Can model rank positives above negatives? 0.5 = random | > 0.55 shows real signal |
| Precision | Of predicted positives, how many were correct? | > 0.60 for trading |
| Recall | Of actual positives, how many did we catch? | depends on use case |
| F1 | Harmonic mean of precision + recall | > 0.5 |
| Hit rate | % of trades profitable | > 55% is good |

## Trading Metrics

| Term | What it is |
|---|---|
| PnL | Profit and loss in dollars |
| Total return | Final equity ÷ starting equity - 1 |
| CAGR | Compound annual growth rate |
| Sharpe | Return / volatility. >1 decent, >1.5 good, >2 excellent |
| Max drawdown | Biggest peak-to-trough loss |
| Calmar | CAGR / \|max DD\| |
| Profit factor | Total wins / total losses. >1.5 healthy, >2 great |
| Win rate | Same as hit rate |
| Alpha | Excess return vs benchmark |
| Beta | Sensitivity to market moves |

## Entry / Exit

| Term | What it is |
|---|---|
| Signal | Model says "trade this" |
| Entry | Opening a position |
| Exit | Closing a position |
| Stop-loss | Auto-exit if price drops below threshold |
| Take-profit | Auto-exit if price rises above threshold |
| Trailing stop | Stop that follows price up |
| Time stop | Exit after N days regardless of price |
| ATR multiple | Stop expressed in units of ATR |
| Position sizing | How much $ per trade |
| Threshold | Minimum probability to trigger a trade |
| Calibration | Mapping raw model scores to true probabilities |

## Risk Management

| Term | What it is |
|---|---|
| Diversification | Spreading risk across positions |
| Concurrency cap | Max open positions per ticker/overall |
| Capacity | How much $ a strategy can handle before moving the market |
| Transaction cost | Broker fees + slippage, in basis points (bps) |
| Basis point (bp) | 0.01%. 5 bps = 0.05% |
| Slippage | Difference between expected and actual fill price |
| Hedging | Taking offsetting positions |
| Leverage | Using borrowed money |

## Market Jargon

| Term | What it is |
|---|---|
| Bull market | Rising prices |
| Bear market | -20% from peak |
| Correction | -10% from peak |
| Volatility | How much prices swing |
| VIX | S&P 500 implied volatility index |
| Momentum | Winners keep winning |
| Mean reversion | Extreme moves reverse |
| Market cap | Share price × shares outstanding |
| Large / mid / small cap | >$10B / $2-10B / <$2B |
| Russell 2000 | Index of US small caps |
| S&P 500 | ~500 largest US companies |
| GICS | Global Industry Classification Standard — 11 sectors |

## Market Efficiency

| Term | What it is |
|---|---|
| Alpha | Return above what the market gives you for the risk |
| EMH | Efficient Market Hypothesis |
| Arbitraged away | Strategy that worked until too many people did it |
| Capacity-limited | Stops working at large size — stays available to retail |
| Factor | Systematic driver of returns (value, momentum, quality) |
| PEAD | Post-Earnings Announcement Drift |
| Short squeeze | Shorts forced to buy as price rises |
| Volatility risk premium | Insurance premium from selling options |
| Carry | Earning yield difference |

## Big Firm Types

| Type | What they do | Examples |
|---|---|---|
| Investment banks | M&A, IPOs, market-making, wealth management | Goldman Sachs, JP Morgan |
| Market makers | Earn bid-ask spread on huge volumes | Citadel Securities, Jane Street |
| HFT firms | Microsecond trading | Jump, Virtu |
| Multi-strat hedge funds | Many pods, many strategies | Citadel, Millennium, Point72 |
| Factor managers | Systematic factor exposure | AQR, Dimensional, Bridgewater |
| Fundamental hedge funds | Deep company research | Tiger Global, Lone Pine |
| Quant research | Secretive, data-heavy | Renaissance (Medallion) |

## Alternative Data

| Source | What it is | Cost |
|---|---|---|
| SEC Form 4 | Insider stock transactions | Free (EDGAR) |
| SEC 13F | Quarterly institutional holdings | Free |
| SEC 8-K | Material corporate events | Free |
| Earnings calendar | Upcoming earnings + consensus | Free (yfinance) |
| FINRA short interest | Reported shorts, bi-monthly | Free |
| News sentiment | NLP on news headlines | Free-ish to $$$ |
| Options flow | Unusual options activity | Paid |
| Satellite imagery | Parking lot counts, oil tankers | Institutional only |

## Python / Code

| Term | What it means |
|---|---|
| DataFrame | Pandas 2D table |
| Index | Row labels (often dates) |
| groupby | Split, apply, combine |
| transform | Returns same shape as input |
| joblib | Fast serialization for sklearn models |
| yfinance | Free stock data library |
| sklearn | scikit-learn |
| TimeSeriesSplit | sklearn's time-respecting CV |
| GridSearchCV | Try every hyperparameter combo |
| RandomizedSearchCV | Try random hyperparameter samples |
| predict_proba | Returns probabilities instead of class labels |
| ewm | Pandas exponentially weighted window |
| pct_change | % change from previous period |
| rolling | Rolling window operation |

## Reference Numbers

| Number | Meaning |
|---|---|
| 0.50 AUC | Coin-flip |
| 0.55 AUC | Meaningful signal for finance |
| 0.60 AUC | Strong (often suspicious — check for leakage) |
| Sharpe > 1 | Decent risk-adjusted return |
| Sharpe > 2 | Excellent |
| ~10% | Historical S&P 500 annualized return |
| ~15% | Typical SPY annualized volatility |
| ~0.75 | Typical SPY long-term Sharpe |
| 5 bps | Realistic retail execution cost per side |
| 80-90% | % of active managers that underperform benchmarks (SPIVA) |

## Project-Specific

| Term | What it means here |
|---|---|
| Step X / Step Xb | Original / improved version of a step |
| All_data_DF | Main DataFrame with OHLCV + indicators + sector + regime |
| Technicals_DF | Separate DataFrame of just indicator values |
| Regime | Market state label (0=calmest, K-1=most volatile) |
| Cluster | Old ticker-grouping scheme (deprecated) |
| Target_Direction | Binary target: 1 if price up in 6 days, else 0 |
| Confluence | Multiple signals agreeing before entering a trade |
| Flexible exit | ATR-based stop-loss + take-profit + trailing + time + lost-conviction |
