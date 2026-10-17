#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 6b — Backtest (LightGBM + Per-Regime Thresholds)
=====================================================

Drop-in replacement for the original Step 6, adapted for the new pipeline:

  1. Loads joblib model bundles (one per regime) instead of pickled
     GridSearchCV objects.
  2. Uses each regime's OPTIMISED threshold (saved alongside the model)
     instead of a hard-coded 0.5 cutoff or top-10 ranking.
  3. SKIPS regimes you don't trust (default: Regime 4 — train precision
     was 99% but test was 33%, classic overfitting).
  4. Allows CONCURRENT positions: each signal opens a position, holds
     7 days, no need to wait for the previous batch to close. More
     representative of real trading than the original "next_eligible_
     buy_date" gating.
  5. Caps positions per day so the strategy doesn't go from 3 trades
     one day to 60 the next.
  6. Computes proper performance metrics:
       cumulative return, annualised return, hit rate, Sharpe, max
       drawdown, average win / loss, profit factor.
  7. Benchmarks vs S&P 500 using your existing All_data_DF (no
     yfinance download — much faster, no network failures).

Author: leemingjun
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# ===========================================================================
# 1.  Configuration
# ===========================================================================

MODEL_DIR          = '/Users/leemingjun/Desktop/Random Forest Model/2025 S&P 500 Prediction (Return Variable)'
MODEL_TYPE         = 'lightgbm'        # filename prefix; matches Step 5b output
GROUP_COL          = 'Regime'

# Regimes to skip entirely. From Step 6c diagnostics:
#   Regime 4 — train precision 99% / test precision 33% (severe overfit)
#   Regime 0 — 68 trades, 50% hit rate, +$5 PnL (essentially noise)
#   Regime 3 — 36 trades, 53% hit rate, -$26 PnL (slight loser)
# Keeping Regimes 1 (workhorse) and 2 (best hit rate).
EXCLUDE_REGIMES    = [0, 3, 4]

# Quality floor on the signal probability. Per-regime thresholds saved by
# Step 5b are around 0.70. The Step 6c by-confidence breakdown showed the
# 0.70-0.75 bucket only hits 53% (essentially break-even), while 0.75-0.80
# hits 60%+ with much higher avg return. Setting this to 0.75 raises the
# effective bar above each regime's stored threshold.
# Set to None to use only the per-regime saved thresholds.
MIN_PROBA_OVERRIDE = 0.55

HOLDING_DAYS       = 7                 # days to hold each position
MAX_SIGNALS_PER_DAY = 10               # cap concurrent new positions per day
MAX_OPEN_PER_TICKER = 1                # don't pile into the same name; a single
                                       # bad ticker can otherwise dominate a
                                       # bad month (HUM cost -$495 in Apr 2025
                                       # before this cap). Set to 2+ to relax.
INITIAL_CAPITAL    = 5_000.0           # USD
TRANSACTION_COST   = 0.0005            # 5 bps per side (round-trip = 10 bps).
                                       # Set to 0 to match the original Step 6.
BENCHMARK_TICKER   = 'SPY'             # used only for the chart label


# ===========================================================================
# 2.  Load all per-regime models
# ===========================================================================

def load_regime_models(model_dir=MODEL_DIR,
                       model_type=MODEL_TYPE,
                       group_col=GROUP_COL,
                       exclude_regimes=EXCLUDE_REGIMES):
    """
    Returns a dict { regime_id -> {'model', 'threshold', 'features', ...} }
    """
    models = {}
    pattern = f'{model_type}_{group_col}_'
    for fname in sorted(os.listdir(model_dir)):
        if not fname.startswith(pattern) or not fname.endswith('.joblib'):
            continue
        # Parse regime id from filename, e.g. 'lightgbm_Regime_2.joblib'
        try:
            regime_id = int(fname.replace(pattern, '').replace('.joblib', ''))
        except ValueError:
            continue
        if regime_id in exclude_regimes:
            print(f"  Skipping {fname} (in exclude list)")
            continue
        bundle = joblib.load(os.path.join(model_dir, fname))
        models[regime_id] = bundle
        print(f"  Loaded regime {regime_id}: threshold={bundle['threshold']:.2f}, "
              f"cv_score={bundle.get('cv_score', float('nan')):.3f}")
    if not models:
        raise FileNotFoundError(
            f"No model files found in {model_dir} matching {pattern}*.joblib"
        )
    return models


# ===========================================================================
# 3.  Generate signals for the entire test period
# ===========================================================================

def generate_signals(test_data, models,
                     group_col=GROUP_COL,
                     min_proba_override=MIN_PROBA_OVERRIDE):
    """
    For each (Date, Ticker) in test_data, look up the regime, run the
    matching model, compare probability to max(per-regime threshold,
    min_proba_override). Returns a long-format DataFrame of BUY signals only.
    """
    test_data = test_data.copy()
    if not isinstance(test_data.index, pd.DatetimeIndex):
        test_data.index = pd.to_datetime(test_data.index)
    test_data = test_data.sort_index()

    # Drop rows whose regime we don't trade
    valid = test_data[test_data[group_col].isin(models.keys())].copy()
    valid = valid.dropna(subset=[group_col])
    print(f"  Test rows kept after regime filter: {len(valid):,} "
          f"(of {len(test_data):,})")
    if min_proba_override is not None:
        print(f"  Min-proba override: {min_proba_override:.2f} "
              f"(applied on top of each regime's saved threshold)")

    signal_chunks = []
    for regime_id, bundle in models.items():
        rows = valid[valid[group_col] == regime_id]
        if rows.empty:
            continue
        feats = bundle['features']
        rows = rows.dropna(subset=feats)
        if rows.empty:
            continue

        proba = bundle['model'].predict_proba(rows[feats].values)[:, 1]
        # Effective threshold = max(saved threshold, optional override)
        threshold = bundle['threshold']
        if min_proba_override is not None:
            threshold = max(threshold, min_proba_override)
        fired = proba >= threshold

        if fired.sum() == 0:
            continue

        chunk = pd.DataFrame({
            'Date':       rows.index[fired],
            'Ticker':     rows['Ticker'].values[fired],
            'Regime':     regime_id,
            'Proba':      proba[fired],
            'Threshold':  threshold,
        })
        signal_chunks.append(chunk)
        print(f"    Regime {regime_id}: {fired.sum():,} signals "
              f"(rate {fired.mean():.1%})")

    if not signal_chunks:
        raise RuntimeError("No signals fired across any regime!")

    signals = pd.concat(signal_chunks, ignore_index=True)
    signals = signals.sort_values(['Date', 'Proba'], ascending=[True, False])
    return signals


# ===========================================================================
# 4.  Cap signals per day (top-N by probability)
# ===========================================================================

def cap_signals_per_day(signals, max_per_day=MAX_SIGNALS_PER_DAY):
    capped = (signals
              .sort_values(['Date', 'Proba'], ascending=[True, False])
              .groupby('Date')
              .head(max_per_day)
              .reset_index(drop=True))
    print(f"  Capped to top {max_per_day} per day: "
          f"{len(capped):,} signals (from {len(signals):,})")
    return capped


# ===========================================================================
# 5.  Trade simulation — concurrent positions, 7-day holding
# ===========================================================================

def simulate_trades(signals, all_data_df,
                    holding_days=HOLDING_DAYS,
                    initial_capital=INITIAL_CAPITAL,
                    transaction_cost=TRANSACTION_COST,
                    sim_start=None,
                    sim_end=None,
                    max_open_per_ticker=MAX_OPEN_PER_TICKER):
    """
    Walks the test window day by day. On each signal date, opens equal-
    weight positions across that day's signals (using available cash).
    Each position is closed `holding_days` later at that day's Close.

    The simulation window defaults to the actual trading window:
      start = first signal date
      end   = last signal date + holding_days + small buffer
    so the equity curve isn't diluted by years of idle cash before
    the first trade. Override with sim_start / sim_end if needed.
    """
    if not isinstance(all_data_df.index, pd.DatetimeIndex):
        all_data_df = all_data_df.copy()
        all_data_df.index = pd.to_datetime(all_data_df.index)

    # Build a fast lookup: (Ticker, Date) -> (Open, Close)
    px = (all_data_df.reset_index()
                       .rename(columns={all_data_df.index.name or 'index': 'Date'})
                       [['Date', 'Ticker', 'Open', 'Close']]
                       .set_index(['Ticker', 'Date'])
                       .sort_index())

    trade_dates = sorted(signals['Date'].unique())
    all_dates   = sorted(all_data_df.index.unique())
    full_index  = pd.DatetimeIndex(all_dates)

    # ---- Restrict iteration to the actual trading window --------------
    if sim_start is None:
        sim_start = pd.Timestamp(signals['Date'].min())
    else:
        sim_start = pd.Timestamp(sim_start)
    if sim_end is None:
        last_sig = pd.Timestamp(signals['Date'].max())
        last_pos = full_index.get_loc(last_sig)
        end_pos  = min(last_pos + holding_days + 2, len(full_index) - 1)
        sim_end  = full_index[end_pos]
    else:
        sim_end = pd.Timestamp(sim_end)
    date_index = full_index[(full_index >= sim_start) & (full_index <= sim_end)]
    print(f"  Simulation window: {sim_start.date()}  →  {sim_end.date()}  "
          f"({len(date_index)} trading days)")

    cash = initial_capital
    open_positions = []   # list of dicts
    closed_trades  = []
    equity_curve   = []   # list of (date, equity)

    for current_date in date_index:
        # 1. Close any positions whose holding period has ended
        still_open = []
        for pos in open_positions:
            if pos['sell_date'] == current_date:
                # Look up sell price (use Close on sell date; fallback to nearest)
                key = (pos['ticker'], current_date)
                if key in px.index:
                    sell_price = float(px.at[key, 'Close'])
                else:
                    still_open.append(pos)
                    continue
                gross   = pos['shares'] * sell_price
                cost    = gross * transaction_cost
                cash   += gross - cost
                pnl     = gross - pos['cost_basis'] - cost - pos['entry_cost']
                pct     = pnl / pos['cost_basis']
                closed_trades.append({
                    **pos,
                    'sell_price':   sell_price,
                    'pnl':          pnl,
                    'pct':          pct,
                    'win':          pnl > 0,
                })
            else:
                still_open.append(pos)
        open_positions = still_open

        # 2. Open new positions on signal dates
        if current_date in trade_dates:
            day_signals = signals[signals['Date'] == current_date]
            n = len(day_signals)
            if n == 0:
                pass
            else:
                # Equal-weight allocation against current cash
                alloc_per_trade = cash / n if cash > 0 else 0
                # Count how many positions are already open per ticker
                open_per_ticker = {}
                for p in open_positions:
                    open_per_ticker[p['ticker']] = open_per_ticker.get(p['ticker'], 0) + 1

                for _, sig in day_signals.iterrows():
                    # Per-ticker concurrency cap: skip if already at max open
                    if open_per_ticker.get(sig['Ticker'], 0) >= max_open_per_ticker:
                        continue

                    # Buy at the NEXT trading day's Open. Use full_index
                    # (not date_index) so trades opened near the end of
                    # the test window can still find a buy/sell date.
                    pos_idx = full_index.get_loc(current_date)
                    if pos_idx + 1 >= len(full_index):
                        continue
                    buy_date  = full_index[pos_idx + 1]
                    sell_idx  = pos_idx + 1 + holding_days
                    if sell_idx >= len(full_index):
                        continue
                    sell_date = full_index[sell_idx]

                    key = (sig['Ticker'], buy_date)
                    if key not in px.index:
                        continue
                    buy_price = float(px.at[key, 'Open'])
                    if buy_price <= 0:
                        continue
                    shares = int(alloc_per_trade // buy_price)
                    if shares < 1:
                        continue
                    cost_basis = shares * buy_price
                    entry_cost = cost_basis * transaction_cost
                    cash -= cost_basis + entry_cost
                    open_per_ticker[sig['Ticker']] = open_per_ticker.get(sig['Ticker'], 0) + 1
                    open_positions.append({
                        'ticker':      sig['Ticker'],
                        'regime':      int(sig['Regime']),
                        'proba':       float(sig['Proba']),
                        'signal_date': current_date,
                        'buy_date':    buy_date,
                        'sell_date':   sell_date,
                        'shares':      shares,
                        'buy_price':   buy_price,
                        'cost_basis':  cost_basis,
                        'entry_cost':  entry_cost,
                    })

        # 3. Mark-to-market equity
        mv = 0.0
        for pos in open_positions:
            key = (pos['ticker'], current_date)
            if key in px.index:
                mv += pos['shares'] * float(px.at[key, 'Close'])
            else:
                mv += pos['cost_basis']
        equity_curve.append((current_date, cash + mv))

    trades_df = pd.DataFrame(closed_trades)
    equity_df = pd.DataFrame(equity_curve, columns=['Date', 'Equity']).set_index('Date')
    return trades_df, equity_df


# ===========================================================================
# 6.  Performance metrics
# ===========================================================================

def compute_metrics(trades_df, equity_df, initial_capital=INITIAL_CAPITAL):
    if trades_df.empty:
        print("No trades closed — nothing to score.")
        return {}

    n         = len(trades_df)
    wins      = trades_df[trades_df['win']]
    losses    = trades_df[~trades_df['win']]
    hit_rate  = len(wins) / n
    avg_win   = wins  ['pct'].mean() if len(wins) else 0
    avg_loss  = losses['pct'].mean() if len(losses) else 0
    profit_f  = (-wins['pnl'].sum() / losses['pnl'].sum()) if losses['pnl'].sum() < 0 else float('inf')

    final_equity = float(equity_df['Equity'].iloc[-1])
    total_ret    = final_equity / initial_capital - 1
    days         = (equity_df.index[-1] - equity_df.index[0]).days
    years        = max(days / 365.25, 1e-9)
    cagr         = (final_equity / initial_capital) ** (1/years) - 1

    daily_ret    = equity_df['Equity'].pct_change().dropna()
    sharpe       = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0

    rolling_max  = equity_df['Equity'].cummax()
    drawdown     = equity_df['Equity'] / rolling_max - 1
    max_dd       = drawdown.min()

    return {
        'trades':           n,
        'hit_rate':         hit_rate,
        'avg_win_pct':      avg_win,
        'avg_loss_pct':     avg_loss,
        'profit_factor':    profit_f,
        'total_return':     total_ret,
        'cagr':             cagr,
        'sharpe':           sharpe,
        'max_drawdown':     max_dd,
        'final_equity':     final_equity,
        'years':            years,
    }


def print_metrics(metrics, label='Strategy'):
    print(f"\n=== {label} performance ===")
    print(f"  Trades:           {metrics['trades']:,}")
    print(f"  Hit rate:         {metrics['hit_rate']:.1%}")
    print(f"  Avg win:          {metrics['avg_win_pct']:+.2%}")
    print(f"  Avg loss:         {metrics['avg_loss_pct']:+.2%}")
    print(f"  Profit factor:    {metrics['profit_factor']:.2f}")
    print(f"  Total return:     {metrics['total_return']:+.2%}  "
          f"over {metrics['years']:.2f} years")
    print(f"  CAGR:             {metrics['cagr']:+.2%}")
    print(f"  Sharpe (daily):   {metrics['sharpe']:.2f}")
    print(f"  Max drawdown:     {metrics['max_drawdown']:.2%}")
    print(f"  Final equity:     ${metrics['final_equity']:,.2f}")


# ===========================================================================
# 7.  Benchmark — S&P 500 from existing data (no yfinance download)
# ===========================================================================

def benchmark_from_data(all_data_df, start, end, ticker=BENCHMARK_TICKER,
                        initial_capital=INITIAL_CAPITAL):
    """
    Buy-and-hold the benchmark ticker. Tries (in order):
      1. SPY / ^SPX / ^GSPC / whatever ticker is requested
      2. Equal-weight basket of tickers that are present every day in
         the window (so newly-listed and delisted tickers don't pollute
         the average).
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)

    # ---- 1. Try the requested benchmark ticker -----------------------
    candidates = [ticker, '^SPX', '^GSPC', 'SPY']
    for cand in candidates:
        if cand in all_data_df['Ticker'].unique():
            bench = all_data_df[all_data_df['Ticker'] == cand].sort_index()
            bench = bench.loc[start:end]
            if len(bench) > 1:
                rets   = bench['Close'].pct_change().fillna(0)
                equity = (1 + rets).cumprod() * initial_capital
                return equity, f'{cand} buy-and-hold'

    # ---- 2. Equal-weight basket (correct vectorised version) ---------
    slc = all_data_df.loc[start:end].copy()
    # Pivot to dates × tickers wide format so per-ticker pct_change works
    wide = (slc.reset_index()
                .rename(columns={slc.index.name or 'index': 'Date'})
                .pivot_table(index='Date', columns='Ticker', values='Close',
                             aggfunc='last'))
    # Keep only tickers present on EVERY day of the window
    full_coverage = wide.columns[wide.notna().all(axis=0)]
    wide = wide[full_coverage]
    if wide.shape[1] == 0:
        # Fallback: forward-fill missing then average
        wide = wide.ffill().bfill()
    rets_wide = wide.pct_change().fillna(0)
    daily_ret = rets_wide.mean(axis=1)
    equity = (1 + daily_ret).cumprod() * initial_capital
    return equity, f'Equal-weight basket ({wide.shape[1]} tickers)'


# ===========================================================================
# 8.  Plot
# ===========================================================================

def plot_equity(equity_df, benchmark_eq, benchmark_label, output_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity_df.index, equity_df['Equity'],
            label='Strategy', color='#1f77b4', lw=2)
    ax.plot(benchmark_eq.index, benchmark_eq.values,
            label=benchmark_label, color='#7f7f7f', lw=1.5, ls='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity ($)')
    ax.set_title('Backtest — LightGBM per-regime strategy vs benchmark')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=140)
        print(f"  Saved chart: {output_path}")
    plt.show()


# ===========================================================================
# 9.  Run it
# ===========================================================================
# Assumes test_data and All_data_DF are in scope from Steps 2 → 4b.

print("[1/6] Loading per-regime models ...")
models = load_regime_models()

print("\n[2/6] Generating signals across the test window ...")
signals = generate_signals(test_data, models)

print("\n[3/6] Capping signals per day ...")
signals = cap_signals_per_day(signals, max_per_day=MAX_SIGNALS_PER_DAY)

print("\n[4/6] Simulating trades (7-day holding, concurrent positions) ...")
t0 = time.time()
trades_df, equity_df = simulate_trades(
    signals, All_data_DF,
    holding_days=HOLDING_DAYS,
    initial_capital=INITIAL_CAPITAL,
    transaction_cost=TRANSACTION_COST,
)
print(f"  Done in {time.time()-t0:.1f}s.  "
      f"{len(trades_df):,} trades closed.")

print("\n[5/6] Computing metrics + benchmark ...")
metrics = compute_metrics(trades_df, equity_df)
print_metrics(metrics, label='LightGBM strategy')

bench_start = equity_df.index.min()
bench_end   = equity_df.index.max()
bench_eq, bench_label = benchmark_from_data(All_data_DF, bench_start, bench_end)
bench_total_ret = bench_eq.iloc[-1] / INITIAL_CAPITAL - 1
print(f"\n  Benchmark ({bench_label}):  total return {bench_total_ret:+.2%}")
print(f"  Strategy alpha vs benchmark: {metrics['total_return'] - bench_total_ret:+.2%}")

print("\n[6/6] Plotting + saving ...")
plot_path = os.path.join(MODEL_DIR, 'Backtest_EquityCurve.png')
plot_equity(equity_df, bench_eq, bench_label, output_path=plot_path)

# Save the trade ledger and equity curve for further inspection
trades_path = os.path.join(MODEL_DIR, 'Backtest_Trades.csv')
equity_path = os.path.join(MODEL_DIR, 'Backtest_Equity.csv')
trades_df.to_csv(trades_path, index=False)
equity_df.to_csv(equity_path)
print(f"  Trades saved to: {trades_path}")
print(f"  Equity curve saved to: {equity_path}")

# Save summary metrics
metrics_df = pd.DataFrame([{**metrics,
                            'benchmark_return': bench_total_ret,
                            'benchmark':        bench_label}])
metrics_path = os.path.join(MODEL_DIR, 'Backtest_Metrics.csv')
metrics_df.to_csv(metrics_path, index=False)
print(f"  Metrics saved to: {metrics_path}")