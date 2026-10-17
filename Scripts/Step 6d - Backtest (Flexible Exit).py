#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 6d — Backtest with FLEXIBLE EXIT (instead of fixed 7-day hold)
====================================================================

Drop-in replacement for Step 6b that uses the SAME saved LightGBM models
but exits positions dynamically based on multiple signals instead of a
fixed timer:

  EXIT WHEN any of:
    1. Model loses confidence
         model's calibrated probability(up next 7 days) < EXIT_PROBA
    2. Stop-loss hit
         price drops more than STOP_LOSS_ATR_MULT × entry_ATR from buy price
    3. Take-profit hit
         price rises more than TAKE_PROFIT_ATR_MULT × entry_ATR from buy price
    4. Trailing stop hit
         price drops TRAILING_ATR_MULT × ATR from the high-water-mark
    5. Time stop
         held for MAX_HOLD_DAYS without any other exit triggering

Why this matters
----------------
Fixed 7-day exit holds losers full duration even when the model has clearly
flipped. It also forces winners to close even when the trade is just hitting
its stride. Flexible exit:

  * Cuts losses faster (stop-loss prevents the -10% disasters)
  * Lets winners run longer (no artificial 7-day cap)
  * Re-evaluates conviction daily (sells when model loses confidence)
  * Locks in gains via trailing stop

Typically lifts Sharpe by 0.2-0.5 vs fixed hold even with the same model,
because you're managing risk dynamically instead of statically.

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
# Configuration
# ===========================================================================

MODEL_DIR        = '/Users/leemingjun/Desktop/Random Forest Model/2025 S&P 500 Prediction (Return Variable)'
MODEL_TYPE       = 'lightgbm'
GROUP_COL        = 'Regime'
EXCLUDE_REGIMES  = [0, 3, 4]

# --- Entry parameters --------------------------------------------------------
ENTRY_PROBA       = 0.55       # buy when model proba >= this (post-calibration)
MAX_NEW_PER_DAY   = 10         # cap concurrent new entries per day
MAX_OPEN_PER_TICKER = 1

# --- Exit parameters --------------------------------------------------------
EXIT_PROBA              = 0.45    # exit when proba drops below this (asymmetric)
STOP_LOSS_ATR_MULT      = 1.0     # exit if price drops > 1.5 × ATR from entry
TAKE_PROFIT_ATR_MULT    = 3.0     # exit if price rises > 3.0 × ATR from entry (asymmetric)
TRAILING_STOP_ATR_MULT  = 3.0     # trail the stop 2.0 × ATR below high-water-mark
MAX_HOLD_DAYS           = 14      # absolute time cap (was 7 in fixed version)

# --- Capital + costs --------------------------------------------------------
INITIAL_CAPITAL  = 5_000.0
TRANSACTION_COST = 0.0005


# ===========================================================================
# Load saved per-regime models (same as Step 6b)
# ===========================================================================

def load_regime_models(model_dir=MODEL_DIR, model_type=MODEL_TYPE,
                       group_col=GROUP_COL, exclude_regimes=EXCLUDE_REGIMES):
    models = {}
    pattern = f'{model_type}_{group_col}_'
    for fname in sorted(os.listdir(model_dir)):
        if not fname.startswith(pattern) or not fname.endswith('.joblib'):
            continue
        try:
            r = int(fname.replace(pattern, '').replace('.joblib', ''))
        except ValueError:
            continue
        if r in exclude_regimes:
            print(f"  Skipping {fname} (in exclude list)")
            continue
        models[r] = joblib.load(os.path.join(model_dir, fname))
        print(f"  Loaded regime {r}: features={len(models[r]['features'])}")
    if not models:
        raise FileNotFoundError(f"No model files found in {model_dir}")
    return models


# ===========================================================================
# Pre-compute daily probabilities for ALL test rows
# (More efficient than predicting one-row-at-a-time inside the loop.)
# ===========================================================================

def precompute_probabilities(test_data, models, group_col=GROUP_COL):
    """
    Returns a wide-format Series indexed by (Date, Ticker) → probability.
    Lookups during simulation are then O(1).
    """
    test_data = test_data.copy()
    if not isinstance(test_data.index, pd.DatetimeIndex):
        test_data.index = pd.to_datetime(test_data.index)
    test_data = test_data.sort_index()

    # Need a (Date, Ticker) lookup
    valid = test_data[test_data[group_col].isin(models.keys())].copy()
    chunks = []
    for r, b in models.items():
        rows = valid[valid[group_col] == r].dropna(subset=b['features'])
        if rows.empty:
            continue
        proba = b['model'].predict_proba(rows[b['features']].values)[:, 1]
        chunks.append(pd.DataFrame({
            'Date':   rows.index,
            'Ticker': rows['Ticker'].values,
            'Proba':  proba,
            'Regime': r,
        }))
    proba_df = pd.concat(chunks, ignore_index=True)
    proba_df['Date'] = pd.to_datetime(proba_df['Date'])
    print(f"  Pre-computed {len(proba_df):,} (Date, Ticker) probabilities")
    return proba_df.set_index(['Date', 'Ticker'])


# ===========================================================================
# Trade simulation with flexible exit
# ===========================================================================

def simulate_flexible_exit(proba_df, all_data_df,
                           entry_proba=ENTRY_PROBA,
                           exit_proba=EXIT_PROBA,
                           stop_loss_atr_mult=STOP_LOSS_ATR_MULT,
                           take_profit_atr_mult=TAKE_PROFIT_ATR_MULT,
                           trailing_atr_mult=TRAILING_STOP_ATR_MULT,
                           max_hold_days=MAX_HOLD_DAYS,
                           max_new_per_day=MAX_NEW_PER_DAY,
                           max_open_per_ticker=MAX_OPEN_PER_TICKER,
                           initial_capital=INITIAL_CAPITAL,
                           transaction_cost=TRANSACTION_COST):
    """
    Walk through the test window day-by-day. Each day:
      1. For every open position, compute today's exit triggers (loss-of-conviction,
         stop-loss, take-profit, trailing stop, time stop). Close if any fire.
      2. For every new signal (proba >= entry_proba) that doesn't conflict with
         the per-ticker concurrency cap, open a position at NEXT day's open.
      3. Mark-to-market the equity curve.
    """
    # Index price + ATR for fast lookup
    if not isinstance(all_data_df.index, pd.DatetimeIndex):
        all_data_df = all_data_df.copy()
        all_data_df.index = pd.to_datetime(all_data_df.index)

    px = (all_data_df.reset_index()
                       .rename(columns={all_data_df.index.name or 'index': 'Date'})
                       [['Date', 'Ticker', 'Open', 'Close', 'ATR_15']]
                       .set_index(['Ticker', 'Date'])
                       .sort_index())

    full_idx = pd.DatetimeIndex(sorted(all_data_df.index.unique()))

    # Restrict simulation to the test window (where we have probabilities)
    sim_dates = sorted({d for (d, _) in proba_df.index.unique()})
    sim_start = pd.Timestamp(min(sim_dates))
    sim_end_pos = min(full_idx.get_loc(pd.Timestamp(max(sim_dates))) + max_hold_days + 2,
                      len(full_idx) - 1)
    sim_end = full_idx[sim_end_pos]
    iter_idx = full_idx[(full_idx >= sim_start) & (full_idx <= sim_end)]
    print(f"  Sim window: {sim_start.date()} → {sim_end.date()} "
          f"({len(iter_idx)} trading days)")

    cash         = initial_capital
    open_pos     = []     # list of dicts
    closed       = []
    equity       = []

    for cur in iter_idx:
        # ---- 1. Evaluate exits on every open position ------------------
        still_open = []
        for pos in open_pos:
            key_today = (pos['ticker'], cur)
            if key_today not in px.index:
                still_open.append(pos)   # missing bar; hold
                continue

            today_close = float(px.at[key_today, 'Close'])
            today_proba = float(proba_df.loc[(cur, pos['ticker']), 'Proba']
                                if (cur, pos['ticker']) in proba_df.index else np.nan)
            days_held = (cur - pos['buy_date']).days

            # Update high-water-mark for trailing stop
            pos['hwm'] = max(pos.get('hwm', pos['buy_price']), today_close)

            # Exit triggers (in priority order)
            exit_reason = None
            if today_close <= pos['stop_loss']:
                exit_reason = 'stop_loss'
            elif today_close >= pos['take_profit']:
                exit_reason = 'take_profit'
            elif today_close <= pos['hwm'] - trailing_atr_mult * pos['entry_atr']:
                exit_reason = 'trailing_stop'
            elif not np.isnan(today_proba) and today_proba < exit_proba:
                exit_reason = 'lost_conviction'
            elif days_held >= max_hold_days:
                exit_reason = 'time_stop'

            if exit_reason is None:
                still_open.append(pos)
                continue

            # Close at today's close
            sell_value = pos['shares'] * today_close
            sell_cost  = sell_value * transaction_cost
            cash += sell_value - sell_cost
            pnl = sell_value - pos['cost_basis'] - pos['entry_cost'] - sell_cost
            closed.append({
                **pos,
                'sell_date':   cur,
                'sell_price':  today_close,
                'pnl':         pnl,
                'pct':         pnl / pos['cost_basis'],
                'win':         pnl > 0,
                'days_held':   days_held,
                'exit_reason': exit_reason,
            })
        open_pos = still_open

        # ---- 2. Look for entry signals -------------------------------
        # Today's signals: every (Date, Ticker) where proba >= entry_proba
        if cur in {d for (d, _) in proba_df.index}:
            today_signals = (proba_df.xs(cur, level='Date')
                                       .query('Proba >= @entry_proba')
                                       .sort_values('Proba', ascending=False))
        else:
            today_signals = pd.DataFrame(columns=['Proba', 'Regime'])

        if len(today_signals) > 0:
            # Skip tickers we already hold (per concurrency cap)
            held = {p['ticker'] for p in open_pos}
            today_signals = today_signals[~today_signals.index.isin(held)]
            today_signals = today_signals.head(max_new_per_day)

            n = len(today_signals)
            if n > 0 and cash > 0:
                alloc = cash / n
                pos_idx = full_idx.get_loc(cur)
                if pos_idx + 1 < len(full_idx):
                    buy_date = full_idx[pos_idx + 1]
                    for tk, row in today_signals.iterrows():
                        key_buy = (tk, buy_date)
                        if key_buy not in px.index:
                            continue
                        buy_price = float(px.at[key_buy, 'Open'])
                        atr       = float(px.at[(tk, cur), 'ATR_15'])
                        if buy_price <= 0 or np.isnan(atr) or atr <= 0:
                            continue
                        shares = int(alloc // buy_price)
                        if shares < 1:
                            continue
                        cb = shares * buy_price
                        ec = cb * transaction_cost
                        cash -= cb + ec
                        open_pos.append({
                            'ticker':       tk,
                            'regime':       int(row['Regime']),
                            'entry_proba':  float(row['Proba']),
                            'signal_date':  cur,
                            'buy_date':     buy_date,
                            'buy_price':    buy_price,
                            'shares':       shares,
                            'cost_basis':   cb,
                            'entry_cost':   ec,
                            'entry_atr':    atr,
                            'stop_loss':    buy_price - stop_loss_atr_mult * atr,
                            'take_profit':  buy_price + take_profit_atr_mult * atr,
                            'hwm':          buy_price,
                        })

        # ---- 3. Mark-to-market ---------------------------------------
        mv = 0.0
        for pos in open_pos:
            key = (pos['ticker'], cur)
            if key in px.index:
                mv += pos['shares'] * float(px.at[key, 'Close'])
            else:
                mv += pos['cost_basis']
        equity.append((cur, cash + mv))

    trades_df = pd.DataFrame(closed)
    equity_df = pd.DataFrame(equity, columns=['Date', 'Equity']).set_index('Date')
    return trades_df, equity_df


# ===========================================================================
# Performance metrics (with exit-reason breakdown)
# ===========================================================================

def compute_metrics_with_exits(trades_df, equity_df, initial_capital=INITIAL_CAPITAL):
    if trades_df.empty:
        print("No trades closed — nothing to score.")
        return {}

    n        = len(trades_df)
    wins     = trades_df[trades_df['win']]
    losses   = trades_df[~trades_df['win']]

    # Standard metrics
    total_ret = float(equity_df['Equity'].iloc[-1]) / initial_capital - 1
    yrs       = max((equity_df.index[-1] - equity_df.index[0]).days / 365.25, 1e-9)
    cagr      = (1 + total_ret) ** (1/yrs) - 1
    daily     = equity_df['Equity'].pct_change().dropna()
    sharpe    = (daily.mean() / daily.std()) * np.sqrt(252) if daily.std() > 0 else 0
    max_dd    = (equity_df['Equity'] / equity_df['Equity'].cummax() - 1).min()
    pf        = (-wins['pnl'].sum() / losses['pnl'].sum()) if losses['pnl'].sum() < 0 else float('inf')

    metrics = {
        'trades':       n,
        'hit_rate':     len(wins) / n,
        'profit_fact':  pf,
        'avg_win_pct':  wins  ['pct'].mean() if len(wins)   else 0,
        'avg_loss_pct': losses['pct'].mean() if len(losses) else 0,
        'avg_days_held': trades_df['days_held'].mean(),
        'total_return': total_ret,
        'cagr':         cagr,
        'sharpe':       sharpe,
        'max_dd':       max_dd,
        'final_equity': float(equity_df['Equity'].iloc[-1]),
        'years':        yrs,
    }

    print(f"\n=== Flexible-exit strategy performance ===")
    print(f"  Trades:                {n:,}")
    print(f"  Hit rate:              {metrics['hit_rate']:.1%}")
    print(f"  Avg days held:         {metrics['avg_days_held']:.1f}")
    print(f"  Avg win:               {metrics['avg_win_pct']:+.2%}")
    print(f"  Avg loss:              {metrics['avg_loss_pct']:+.2%}")
    print(f"  Profit factor:         {metrics['profit_fact']:.2f}")
    print(f"  Total return:          {metrics['total_return']:+.2%} over {yrs:.2f} years")
    print(f"  CAGR:                  {metrics['cagr']:+.2%}")
    print(f"  Sharpe (daily):        {metrics['sharpe']:.2f}")
    print(f"  Max drawdown:          {metrics['max_dd']:.2%}")
    print(f"  Final equity:          ${metrics['final_equity']:,.2f}")

    # Exit-reason breakdown — the headline diagnostic for flexible exit
    print(f"\n=== Exit-reason breakdown ===")
    by_reason = trades_df.groupby('exit_reason').agg(
        count=('pnl', 'size'),
        hit_rate=('win', 'mean'),
        avg_pct=('pct', 'mean'),
        avg_days=('days_held', 'mean'),
        total_pnl=('pnl', 'sum'),
    ).sort_values('total_pnl', ascending=False)
    print(by_reason.to_string())

    return metrics, by_reason


# ===========================================================================
# Run it
# ===========================================================================
print("[1/5] Loading per-regime models ...")
models = load_regime_models()

print("\n[2/5] Pre-computing daily probabilities ...")
proba_df = precompute_probabilities(test_data, models)

print("\n[3/5] Simulating flexible-exit trades ...")
t0 = time.time()
trades_df, equity_df = simulate_flexible_exit(proba_df, All_data_DF)
print(f"  Done in {time.time()-t0:.1f}s.  {len(trades_df):,} trades closed.")

print("\n[4/5] Computing metrics ...")
metrics, by_reason = compute_metrics_with_exits(trades_df, equity_df)

print("\n[5/5] Saving + plotting ...")
trades_df.to_csv(os.path.join(MODEL_DIR, 'Backtest_Flexible_Trades.csv'), index=False)
equity_df.to_csv(os.path.join(MODEL_DIR, 'Backtest_Flexible_Equity.csv'))
by_reason.to_csv(os.path.join(MODEL_DIR, 'Backtest_Flexible_ExitReasons.csv'))

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(equity_df.index, equity_df['Equity'], color='#1f77b4', lw=2)
ax.set_title('Flexible-Exit Strategy Equity Curve')
ax.set_xlabel('Date'); ax.set_ylabel('Equity ($)'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'Backtest_Flexible_Equity.png'), dpi=140)
plt.show()
print(f"  Saved chart + CSVs.")
