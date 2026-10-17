#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 9 — Post-Earnings Announcement Drift (PEAD) Strategy
==========================================================

A FUNDAMENTALLY DIFFERENT approach from the technical-indicator pipeline.

Core idea (one of the most documented anomalies in finance):
  Stocks that beat analyst earnings expectations by a meaningful margin
  tend to drift UP for 30-60 trading days afterward. Stocks that miss
  drift DOWN. This persists because:

  1. Behavioral underreaction — investors don't fully process earnings
     surprise magnitude immediately
  2. Institutional rebalancing constraints — funds reposition slowly
  3. Analyst upgrade cascades — follow surprises with delay
  4. Capacity-limited — too small for big funds to fully arbitrage

Academic references:
  Bernard & Thomas (1989): documented PEAD persists 60+ days
  Foster et al. (1984): original empirical work
  Numerous follow-ups confirm the effect is still present as of 2020+

Why this is worth trying when technical-indicator approaches failed
-------------------------------------------------------------------
Your CV AUC plateaued at ~0.51 because price/volume features alone are
arbitraged. PEAD uses ENTIRELY DIFFERENT information: analyst expectations
vs reported earnings. This is event-driven, not technical-derived. If it
works, you'll see Sharpe > 1 in walk-forward without any model training —
the alpha comes from the structural anomaly, not from prediction.

If PEAD ALSO doesn't work in walk-forward, the conclusion is unambiguous:
this universe / timeframe combination doesn't have retail-accessible alpha.
That's Option 4 territory.

Author: leemingjun
"""

import os
import time
import json
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
sns.set()


# ===========================================================================
# Configuration
# ===========================================================================

CACHE_DIR     = '/Users/leemingjun/Desktop/Random Forest Model/PaperTrading/pead_cache'
EARNINGS_DIR  = os.path.join(CACHE_DIR, 'earnings')
OUTPUT_DIR    = '/Users/leemingjun/Desktop/Random Forest Model/2025 S&P 500 Prediction (Return Variable)'
for d in (CACHE_DIR, EARNINGS_DIR):
    os.makedirs(d, exist_ok=True)

# --- Signal generation ---
SURPRISE_THRESHOLD_BUY  = 5.0      # % surprise to trigger BUY (academic standard)
SURPRISE_THRESHOLD_SHORT = -5.0    # % to SHORT (long-short variant; off by default)
USE_LONG_SHORT          = False    # True = also short strong misses

# --- Trade execution ---
ENTRY_DELAY_DAYS        = 1        # buy 1 trading day AFTER earnings (avoid same-day)
HOLD_DAYS               = 60       # canonical PEAD horizon
INITIAL_CAPITAL         = 5_000.0
TRANSACTION_COST        = 0.0005   # 5 bps per side
MAX_CONCURRENT_POSITIONS = 30      # cap exposure (otherwise too many trades)

# --- Walk-forward folds (same as Step 7d/7e for direct comparison) ---
WALK_FORWARD_FOLDS = [
    ('2021-07-12', '2022-12-31'),    # bear market
    ('2023-01-13', '2024-06-30'),    # bull recovery
    ('2024-01-15', '2026-04-30'),    # bull continuation
]

# --- Optional add-ons (off by default — test pure PEAD first) ---
USE_QUALITY_FILTER      = False    # only buy stocks above their 50-MA
USE_STOP_LOSS           = False    # add ATR stop-loss to plain PEAD
STOP_LOSS_ATR_MULT      = 2.0      # only used if USE_STOP_LOSS=True


# ===========================================================================
# 1.  Fetch earnings history per ticker (cached)
# ===========================================================================

def fetch_earnings_history(ticker, force_refresh=False, max_age_days=14):
    """
    Pulls per-ticker earnings dates + EPS Estimate + Reported EPS + Surprise%
    from yfinance. Returns DataFrame indexed by earnings date.

    Cached to disk for re-runs.
    """
    cache_file = os.path.join(EARNINGS_DIR, f'{ticker}.csv')
    if os.path.exists(cache_file) and not force_refresh:
        age_days = (time.time() - os.path.getmtime(cache_file)) / 86400
        if age_days < max_age_days:
            return pd.read_csv(cache_file, parse_dates=['Date'], index_col='Date')

    try:
        t = yf.Ticker(ticker)
        ed = t.earnings_dates
        if ed is None or ed.empty:
            return pd.DataFrame()
        ed = ed.copy()
        ed.index = pd.to_datetime(ed.index).tz_localize(None)
        ed.index.name = 'Date'
        # Normalize column names across yfinance versions
        col_map = {}
        for c in ed.columns:
            if 'estimate' in c.lower(): col_map[c] = 'EPS_Estimate'
            elif 'reported' in c.lower() or 'actual' in c.lower(): col_map[c] = 'EPS_Reported'
            elif 'surprise' in c.lower(): col_map[c] = 'Surprise_Pct'
        ed = ed.rename(columns=col_map)
        keep = [c for c in ['EPS_Estimate', 'EPS_Reported', 'Surprise_Pct']
                if c in ed.columns]
        ed = ed[keep].copy()
        ed.to_csv(cache_file)
        return ed
    except Exception as e:
        return pd.DataFrame()


def fetch_all_earnings(tickers, max_tickers=None):
    """Bulk-fetch earnings for all tickers. Cached so subsequent runs are fast."""
    if max_tickers:
        tickers = tickers[:max_tickers]
    print(f"  Fetching earnings for {len(tickers)} tickers ...")
    all_earnings = []
    for i, t in enumerate(tickers):
        if i % 50 == 0 and i > 0:
            print(f"    {i}/{len(tickers)} ...")
        ed = fetch_earnings_history(t)
        if not ed.empty:
            ed = ed.copy()
            ed['Ticker'] = t
            all_earnings.append(ed.reset_index())
    if not all_earnings:
        raise RuntimeError("No earnings data fetched — check yfinance / network")
    df = pd.concat(all_earnings, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    print(f"  Got {len(df):,} earnings events across {df['Ticker'].nunique()} tickers")
    return df


# ===========================================================================
# 2.  Compute surprises and categorize signals
# ===========================================================================

def compute_signals(earnings_df,
                    threshold_buy=SURPRISE_THRESHOLD_BUY,
                    threshold_short=SURPRISE_THRESHOLD_SHORT):
    """
    For each earnings event, attach a signal_strength label:
      'strong_beat'  surprise > threshold_buy
      'beat'         0 < surprise < threshold_buy
      'miss'         threshold_short < surprise < 0
      'strong_miss'  surprise < threshold_short
    """
    df = earnings_df.copy().dropna(subset=['Surprise_Pct'])
    df['signal_strength'] = pd.cut(
        df['Surprise_Pct'],
        bins=[-np.inf, threshold_short, 0, threshold_buy, np.inf],
        labels=['strong_miss', 'miss', 'beat', 'strong_beat'],
    )

    # Summary stats
    counts = df['signal_strength'].value_counts()
    print(f"\n  Signal distribution:")
    for sig, n in counts.items():
        pct = n / len(df) * 100
        print(f"    {sig:15s}: {n:5d} events ({pct:.1f}%)")
    return df


# ===========================================================================
# 3.  Generate trade signals: list of (Date, Ticker, side)
# ===========================================================================

def generate_pead_signals(signals_df, use_long_short=USE_LONG_SHORT):
    """
    Returns DataFrame of trade signals with columns:
      Date (= earnings date), Ticker, side ('long' or 'short'), Surprise_Pct
    """
    long_signals = signals_df[signals_df['signal_strength'] == 'strong_beat'].copy()
    long_signals['side'] = 'long'

    if use_long_short:
        short_signals = signals_df[signals_df['signal_strength'] == 'strong_miss'].copy()
        short_signals['side'] = 'short'
        signals = pd.concat([long_signals, short_signals], ignore_index=True)
    else:
        signals = long_signals

    signals = signals[['Date', 'Ticker', 'side', 'Surprise_Pct']].sort_values('Date')
    print(f"\n  Generated {len(signals):,} signals "
          f"({(signals['side']=='long').sum()} long, "
          f"{(signals['side']=='short').sum()} short)")
    return signals


# ===========================================================================
# 4.  Backtest — buy at next-day open, hold N days, sell at close
# ===========================================================================

def backtest_pead(signals, all_data_df,
                  hold_days=HOLD_DAYS,
                  entry_delay=ENTRY_DELAY_DAYS,
                  initial_capital=INITIAL_CAPITAL,
                  transaction_cost=TRANSACTION_COST,
                  max_concurrent=MAX_CONCURRENT_POSITIONS,
                  use_quality_filter=USE_QUALITY_FILTER,
                  use_stop_loss=USE_STOP_LOSS,
                  stop_loss_atr_mult=STOP_LOSS_ATR_MULT,
                  date_window=None):
    """
    Walk through every trading day in the price data:
      1. For each open position, check if hold_days elapsed (or stop-loss hit)
         — if so, close at today's close.
      2. For new signals firing in the window, open a position at the buy_date
         (= signal_date + entry_delay trading days).
      3. Equal-weight allocation across concurrent positions.
    """
    if not isinstance(all_data_df.index, pd.DatetimeIndex):
        all_data_df = all_data_df.copy()
        all_data_df.index = pd.to_datetime(all_data_df.index)
    all_data_df = all_data_df.sort_index()

    # Price + ATR lookup
    cols_needed = ['Open', 'Close']
    if use_stop_loss and 'ATR_15' in all_data_df.columns:
        cols_needed.append('ATR_15')
    if use_quality_filter and 'SMA_ratio' in all_data_df.columns:
        cols_needed.append('SMA_ratio')

    px = (all_data_df.reset_index()
                       .rename(columns={all_data_df.index.name or 'index': 'Date'})
                       [['Date', 'Ticker'] + cols_needed]
                       .set_index(['Ticker', 'Date'])
                       .sort_index())

    full_idx = pd.DatetimeIndex(sorted(all_data_df.index.unique()))

    # Restrict to date_window if given (for walk-forward)
    if date_window:
        start, end = pd.Timestamp(date_window[0]), pd.Timestamp(date_window[1])
        signals = signals[(signals['Date'] >= start) & (signals['Date'] <= end)]
        sim_idx = full_idx[(full_idx >= start) & (full_idx <= end + pd.Timedelta(days=hold_days*2))]
    else:
        sim_idx = full_idx

    # Pre-compute mapping from signal_date -> buy_date (signal_date + entry_delay)
    signals = signals.copy()
    signals['buy_date'] = signals['Date'].apply(
        lambda d: full_idx[min(full_idx.searchsorted(d) + entry_delay, len(full_idx) - 1)]
    )

    cash = initial_capital
    open_pos = []
    closed = []
    equity = []
    sigs_by_buy_date = signals.groupby('buy_date')

    for cur in sim_idx:
        # 1. Close positions whose hold period is up (or stop hit)
        still_open = []
        for pos in open_pos:
            key = (pos['ticker'], cur)
            if key not in px.index:
                still_open.append(pos)
                continue

            today_close = float(px.at[key, 'Close'])
            days_held = sum(1 for d in full_idx
                            if pos['buy_date'] <= d <= cur) - 1

            exit_reason = None
            if days_held >= hold_days:
                exit_reason = 'time_stop'
            elif use_stop_loss and 'entry_atr' in pos:
                if pos['side'] == 'long' and today_close <= pos['stop_price']:
                    exit_reason = 'stop_loss'
                elif pos['side'] == 'short' and today_close >= pos['stop_price']:
                    exit_reason = 'stop_loss'

            if exit_reason is None:
                still_open.append(pos)
                continue

            # Close position
            sell_value = pos['shares'] * today_close
            sell_cost = sell_value * transaction_cost
            if pos['side'] == 'long':
                cash += sell_value - sell_cost
                pnl = sell_value - pos['cost_basis'] - pos['entry_cost'] - sell_cost
            else:  # short
                cash += pos['cost_basis'] + (pos['cost_basis'] - sell_value) - sell_cost
                pnl = pos['cost_basis'] - sell_value - pos['entry_cost'] - sell_cost
            closed.append({
                **pos,
                'sell_date': cur,
                'sell_price': today_close,
                'pnl': pnl,
                'pct': pnl / pos['cost_basis'],
                'win': pnl > 0,
                'days_held': days_held,
                'exit_reason': exit_reason,
            })
        open_pos = still_open

        # 2. Open new positions for signals whose buy_date == cur
        if cur in sigs_by_buy_date.groups:
            today_signals = sigs_by_buy_date.get_group(cur)
            held = {p['ticker'] for p in open_pos}
            today_signals = today_signals[~today_signals['Ticker'].isin(held)]

            # Cap by max_concurrent
            slots_available = max_concurrent - len(open_pos)
            if slots_available <= 0:
                pass
            else:
                # Take strongest surprises first
                today_signals = today_signals.assign(
                    abs_surprise=today_signals['Surprise_Pct'].abs()
                ).sort_values('abs_surprise', ascending=False).head(slots_available)

                if len(today_signals) > 0 and cash > 0:
                    alloc = cash / len(today_signals)
                    for _, sig in today_signals.iterrows():
                        tk = sig['Ticker']
                        key = (tk, cur)
                        if key not in px.index:
                            continue

                        # Quality filter (optional)
                        if use_quality_filter:
                            sma_r = float(px.at[key, 'SMA_ratio']) if 'SMA_ratio' in px.columns else 1.0
                            if sig['side'] == 'long' and sma_r < 1.0:
                                continue
                            if sig['side'] == 'short' and sma_r > 1.0:
                                continue

                        buy_price = float(px.at[key, 'Open'])
                        if buy_price <= 0:
                            continue
                        shares = int(alloc // buy_price)
                        if shares < 1:
                            continue
                        cb = shares * buy_price
                        ec = cb * transaction_cost
                        cash -= cb + ec

                        pos_dict = {
                            'ticker': tk, 'side': sig['side'],
                            'signal_date': sig['Date'],
                            'surprise_pct': sig['Surprise_Pct'],
                            'buy_date': cur, 'buy_price': buy_price,
                            'shares': shares, 'cost_basis': cb,
                            'entry_cost': ec,
                        }
                        if use_stop_loss and 'ATR_15' in px.columns:
                            atr = float(px.at[key, 'ATR_15'])
                            if not np.isnan(atr) and atr > 0:
                                pos_dict['entry_atr'] = atr
                                if sig['side'] == 'long':
                                    pos_dict['stop_price'] = buy_price - stop_loss_atr_mult * atr
                                else:
                                    pos_dict['stop_price'] = buy_price + stop_loss_atr_mult * atr
                        open_pos.append(pos_dict)

        # 3. Mark-to-market
        mv = 0.0
        for p in open_pos:
            key = (p['ticker'], cur)
            if key in px.index:
                price = float(px.at[key, 'Close'])
                if p['side'] == 'long':
                    mv += p['shares'] * price
                else:
                    mv += p['cost_basis'] + (p['cost_basis'] - p['shares'] * price)
            else:
                mv += p['cost_basis']
        equity.append((cur, cash + mv))

    trades_df = pd.DataFrame(closed)
    equity_df = pd.DataFrame(equity, columns=['Date', 'Equity']).set_index('Date')
    return trades_df, equity_df


# ===========================================================================
# 5.  Performance metrics + benchmark comparison
# ===========================================================================

def compute_metrics(trades, equity, initial_capital=INITIAL_CAPITAL):
    if trades is None or trades.empty:
        return {}
    n = len(trades)
    wins = trades['win'].sum()
    pf = (-trades.loc[trades['win'], 'pnl'].sum() /
          trades.loc[~trades['win'], 'pnl'].sum()) if (n - wins) > 0 else float('inf')
    final = float(equity['Equity'].iloc[-1])
    yrs = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    cagr = (final / initial_capital) ** (1/yrs) - 1
    daily = equity['Equity'].pct_change().dropna()
    sharpe = (daily.mean() / daily.std()) * np.sqrt(252) if daily.std() > 0 else 0
    max_dd = (equity['Equity'] / equity['Equity'].cummax() - 1).min()
    return {
        'trades': n, 'hit_rate': wins / n,
        'avg_days_held': trades['days_held'].mean(),
        'avg_win_pct': trades.loc[trades['win'], 'pct'].mean() if wins else 0,
        'avg_loss_pct': trades.loc[~trades['win'], 'pct'].mean() if (n - wins) else 0,
        'profit_fact': pf,
        'total_return': final / initial_capital - 1,
        'cagr': cagr, 'sharpe': sharpe, 'max_dd': max_dd,
        'final_equity': final, 'years': yrs,
    }


def print_metrics(m, label='PEAD'):
    if not m:
        print(f"  {label}: no trades.")
        return
    print(f"\n=== {label} performance ===")
    print(f"  Trades:           {m['trades']:,}")
    print(f"  Hit rate:         {m['hit_rate']:.1%}")
    print(f"  Avg days held:    {m['avg_days_held']:.1f}")
    print(f"  Avg win:          {m['avg_win_pct']:+.2%}")
    print(f"  Avg loss:         {m['avg_loss_pct']:+.2%}")
    print(f"  Profit factor:    {m['profit_fact']:.2f}")
    print(f"  Total return:     {m['total_return']:+.2%} over {m['years']:.2f} years")
    print(f"  CAGR:             {m['cagr']:+.2%}")
    print(f"  Sharpe (daily):   {m['sharpe']:.2f}")
    print(f"  Max drawdown:     {m['max_dd']:.2%}")
    print(f"  Final equity:     ${m['final_equity']:,.2f}")


def benchmark_spy(equity, initial_capital=INITIAL_CAPITAL):
    """Buy-and-hold SPY over the same window for comparison."""
    start = equity.index.min()
    end = equity.index.max() + pd.Timedelta(days=2)
    try:
        spy = yf.download('SPY', start=start, end=end, progress=False, auto_adjust=True)
        if spy.empty:
            return None
        spy_close = spy['Close']
        if isinstance(spy_close, pd.DataFrame):
            spy_close = spy_close.iloc[:, 0]
        spy_close = spy_close.dropna().reindex(equity.index, method='ffill').dropna()
        spy_norm = (spy_close / spy_close.iloc[0]) * initial_capital
        spy_total_ret = float(spy_norm.iloc[-1]) / initial_capital - 1
        return spy_norm, spy_total_ret
    except Exception:
        return None


# ===========================================================================
# 6.  Walk-forward across multiple periods
# ===========================================================================

def walk_forward(signals, all_data_df, folds=WALK_FORWARD_FOLDS):
    """Run PEAD backtest separately on each fold's window."""
    print(f"\n{'='*78}\nWALK-FORWARD across {len(folds)} historical windows\n{'='*78}")

    fold_metrics = []
    for i, (start, end) in enumerate(folds, 1):
        print(f"\n--- Fold {i}/{len(folds)}: {start} → {end} ---")
        trades, equity = backtest_pead(signals, all_data_df,
                                       date_window=(start, end))
        m = compute_metrics(trades, equity)
        m['fold'] = i
        m['start'] = start
        m['end'] = end
        fold_metrics.append(m)
        print(f"  trades={m.get('trades', 0)}, return={m.get('total_return', 0):+.2%}, "
              f"sharpe={m.get('sharpe', 0):.2f}, max_dd={m.get('max_dd', 0):.2%}")

    summary = pd.DataFrame(fold_metrics)
    summary_path = os.path.join(OUTPUT_DIR, 'PEAD_WalkForward_Folds.csv')
    summary.to_csv(summary_path, index=False)

    print(f"\n{'='*78}\nWALK-FORWARD AGGREGATE\n{'='*78}")
    print(f"  Mean return:       {summary['total_return'].mean():+.2%}")
    print(f"  Mean CAGR:         {summary['cagr'].mean():+.2%}")
    print(f"  Mean Sharpe:       {summary['sharpe'].mean():.2f}")
    print(f"  Worst fold:        {summary['total_return'].min():+.2%}")
    print(f"  Folds positive:    {(summary['total_return'] > 0).sum()}/{len(summary)}")
    print(f"\n  Saved: {summary_path}")
    return summary


# ===========================================================================
# 7.  Main pipeline
# ===========================================================================

def main(all_data_df, max_tickers=None):
    print("=" * 70)
    print("Step 9 — Post-Earnings Announcement Drift Strategy")
    print("=" * 70)

    # 1. Tickers from your existing data
    tickers = sorted(all_data_df['Ticker'].unique().tolist())
    print(f"\n[1] Will scan {len(tickers)} tickers for earnings history")

    # 2. Fetch earnings for all tickers (cached)
    print(f"\n[2] Fetching earnings history (cached after first run) ...")
    earnings_df = fetch_all_earnings(tickers, max_tickers=max_tickers)

    # 3. Compute surprise categories
    print(f"\n[3] Categorizing surprises ...")
    signals_df = compute_signals(earnings_df)

    # 4. Generate trade signals
    print(f"\n[4] Generating trade signals (threshold = {SURPRISE_THRESHOLD_BUY}%) ...")
    signals = generate_pead_signals(signals_df)

    if signals.empty:
        print("\nNo signals generated. Check earnings data and threshold.")
        return None

    # 5. Backtest on full window
    print(f"\n[5] Backtesting on full available window ...")
    t0 = time.time()
    trades, equity = backtest_pead(signals, all_data_df)
    print(f"  Done in {time.time()-t0:.1f}s.  {len(trades):,} trades closed.")

    metrics = compute_metrics(trades, equity)
    print_metrics(metrics, label='PEAD (full window)')

    # 6. Compare to SPY
    print(f"\n[6] Benchmarking vs SPY ...")
    bench = benchmark_spy(equity)
    if bench is not None:
        spy_norm, spy_ret = bench
        print(f"  SPY buy-and-hold over same window:  {spy_ret:+.2%}")
        print(f"  PEAD alpha vs SPY:                  "
              f"{metrics.get('total_return', 0) - spy_ret:+.2%}")

    # 7. Save
    trades.to_csv(os.path.join(OUTPUT_DIR, 'PEAD_Trades.csv'), index=False)
    equity.to_csv(os.path.join(OUTPUT_DIR, 'PEAD_Equity.csv'))
    print(f"\n  Trades + equity saved to {OUTPUT_DIR}")

    # 8. Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity.index, equity['Equity'], label='PEAD strategy', color='#1f77b4', lw=2)
    if bench is not None:
        ax.plot(spy_norm.index, spy_norm.values, label='SPY buy-and-hold',
                color='#7f7f7f', lw=1.5, ls='--')
    ax.set_title(f'PEAD Strategy — {SURPRISE_THRESHOLD_BUY}% threshold, '
                 f'{HOLD_DAYS}-day hold')
    ax.set_xlabel('Date'); ax.set_ylabel('Equity ($)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'PEAD_Equity_vs_SPY.png'), dpi=140)
    plt.show()

    # 9. Walk-forward across multiple periods
    print(f"\n[7] Running walk-forward validation ...")
    wf_summary = walk_forward(signals, all_data_df)

    return signals, trades, equity, metrics, wf_summary


# ===========================================================================
# Run it
# ===========================================================================
# Assumes All_data_DF is in scope from Step 1b.
# First run takes ~10-15 minutes (downloads earnings for ~500 tickers).
# Subsequent runs are fast (everything cached).

results = main(All_data_DF)


# ===========================================================================
# What to do based on the result
# ===========================================================================
#
# 1. If full-window Sharpe > 1.0 AND walk-forward mean Sharpe > 0.8:
#    PEAD has real alpha. Three optional enhancements:
#      a) Lower SURPRISE_THRESHOLD_BUY to 3% — more signals, may improve Sharpe
#      b) Set USE_LONG_SHORT = True — adds shorts on strong misses
#      c) Set USE_QUALITY_FILTER = True — only buy uptrending names
#
# 2. If full-window Sharpe > 1.0 BUT walk-forward Sharpe < 0.5:
#    Strategy works in some periods, fails in others. PEAD has degraded
#    over time as the market becomes more efficient. Trade-able but
#    expect lumpy returns.
#
# 3. If both Sharpes < 0.5:
#    Even PEAD doesn't survive on this universe. The S&P 500 daily-bars
#    problem genuinely doesn't have retail-accessible alpha. Time for
#    Option 4 — pivot to small-caps, options income, or accept passive.
#
# 4. If full-window Sharpe NEGATIVE:
#    yfinance earnings data may have issues (lookahead, mis-aligned dates).
#    Spot-check a few signals manually before drawing conclusions.
