#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 6e — Backtest with CONFLUENCE FILTER + Flexible Exit
==========================================================

Same model bundles + same flexible-exit logic as Step 6d, but only enters
positions when MULTIPLE independent signals all agree.

Why this matters
----------------
Walk-forward (Step 7d) showed the model lost money in Fold 1 (bear market
2021-22): 868 stop-losses vs only 180 take-profits. The LightGBM model kept
saying "buy" while the market kept punishing every entry.

The fix: require multiple independent confirmations before opening a trade.
A signal that's wrong about ONE thing might be ignored, but a signal that
the LightGBM AND the trend AND the volatility regime AND momentum all agree
on is much more likely to be a real opportunity.

Five toggleable confluence filters
----------------------------------
  1. MARKET TREND        — broader market (median SMA_ratio across S&P 500)
                           must be uptrending. Single biggest filter; would
                           have blocked most Fold 1 trades.
  2. STOCK TREND         — the stock itself must be above its 50-MA
                           (its own SMA_ratio > 1.0)
  3. VOLATILITY REGIME   — stock's ATR_Ratio not in extreme upper percentile
                           (avoid catching falling knives)
  4. NOT OVERBOUGHT      — RSI_15 below ceiling (avoid blow-off tops)
  5. POSITIVE MOMENTUM   — 20-day return positive (don't buy downtrenders)

Each filter is independently toggleable so you can test which ones matter.

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
ENTRY_PROBA       = 0.55
MAX_NEW_PER_DAY   = 10
MAX_OPEN_PER_TICKER = 1

# --- Confluence filters (toggleable) ----------------------------------------
USE_MARKET_TREND   = True       # most impactful — kills most bear-market trades
USE_STOCK_TREND    = True
USE_VOL_REGIME     = False
USE_RSI_CEILING    = False
USE_MOMENTUM       = False

# Filter thresholds
MARKET_TREND_THRESH    = 1.00    # median SMA_ratio across all stocks must be > this
STOCK_TREND_THRESH     = 1.00    # ticker's own SMA_ratio must be > this
VOL_REGIME_PCTILE      = 0.85    # skip if ATR_Ratio > this percentile (cross-sectional)
RSI_CEILING            = 75      # skip if RSI_15 > this
MOMENTUM_LOOKBACK      = 20      # days for momentum check
MOMENTUM_THRESH        = 0.0     # 20-day return must be > this

# --- Flexible exit (matches Step 6d tuned defaults) -------------------------
EXIT_PROBA              = 0.45
STOP_LOSS_ATR_MULT      = 1.0
TAKE_PROFIT_ATR_MULT    = 3.0
TRAILING_STOP_ATR_MULT  = 3.0
MAX_HOLD_DAYS           = 14

# --- Capital + costs --------------------------------------------------------
INITIAL_CAPITAL  = 5_000.0
TRANSACTION_COST = 0.0005


# ===========================================================================
# Load saved per-regime models
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
# Pre-compute model probabilities
# ===========================================================================

def precompute_probabilities(test_data, models, group_col=GROUP_COL):
    test_data = test_data.copy()
    if not isinstance(test_data.index, pd.DatetimeIndex):
        test_data.index = pd.to_datetime(test_data.index)
    test_data = test_data.sort_index()
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
# Compute confluence filters per (Date, Ticker)
# ===========================================================================

def compute_filter_table(test_data, momentum_lookback=MOMENTUM_LOOKBACK):
    """
    For each (Date, Ticker), compute 5 boolean filter values:
      market_trend, stock_trend, vol_regime, not_overbought, positive_momentum
    Returns DataFrame indexed by (Date, Ticker).
    """
    df = test_data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # 1. Stock's own trend (SMA_ratio > threshold = short MA > long MA = uptrend)
    df['_stock_trend'] = df['SMA_ratio'] > STOCK_TREND_THRESH

    # 2. Market trend: median SMA_ratio across all stocks per date
    market_trend_per_day = df.groupby(df.index)['SMA_ratio'].transform('median')
    df['_market_trend'] = market_trend_per_day > MARKET_TREND_THRESH

    # 3. Volatility regime: stock's ATR_Ratio not in extreme upper percentile
    vol_pctile_per_day = df.groupby(df.index)['ATR_Ratio'].transform(
        lambda x: x.rank(pct=True)
    )
    df['_vol_regime'] = vol_pctile_per_day < VOL_REGIME_PCTILE

    # 4. Not overbought
    df['_not_overbought'] = df['RSI_15'] < RSI_CEILING

    # 5. Positive momentum (compute 20-day return per ticker)
    g = df.groupby('Ticker', group_keys=False)
    df['_mom_return'] = g['Close'].pct_change(momentum_lookback)
    df['_positive_momentum'] = df['_mom_return'] > MOMENTUM_THRESH

    # Build result table
    df = df.reset_index().rename(columns={df.index.name or 'index': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])

    return df.set_index(['Date', 'Ticker'])[
        ['_market_trend', '_stock_trend', '_vol_regime',
         '_not_overbought', '_positive_momentum']
    ]


def apply_confluence_filters(proba_df, filter_df,
                             use_market=USE_MARKET_TREND,
                             use_stock=USE_STOCK_TREND,
                             use_vol=USE_VOL_REGIME,
                             use_rsi=USE_RSI_CEILING,
                             use_mom=USE_MOMENTUM,
                             entry_proba=ENTRY_PROBA):
    """
    Joins probabilities with filter table, keeps only rows where:
      - Proba >= entry_proba AND all active filters pass
    Reports per-filter pass rate.
    """
    # Join proba and filters on (Date, Ticker)
    merged = proba_df.join(filter_df, how='left')

    # Start with all signals above probability threshold
    base_mask = merged['Proba'] >= entry_proba
    n_base = base_mask.sum()
    print(f"\n  Base signals (Proba >= {entry_proba}): {n_base:,}")

    # Apply each filter that's enabled, log pass rate
    final_mask = base_mask.copy()
    filter_specs = [
        ('market_trend',       use_market, '_market_trend'),
        ('stock_trend',        use_stock,  '_stock_trend'),
        ('vol_regime',         use_vol,    '_vol_regime'),
        ('not_overbought',     use_rsi,    '_not_overbought'),
        ('positive_momentum',  use_mom,    '_positive_momentum'),
    ]
    for name, on, col in filter_specs:
        if not on:
            print(f"    [skip] {name}")
            continue
        passes = merged[col].fillna(False)
        n_pass = (base_mask & passes).sum()
        pct = n_pass / max(n_base, 1)
        print(f"    [keep] {name:20s}  passes {n_pass:,} of {n_base:,}  "
              f"({pct:.1%})")
        final_mask &= passes

    n_final = final_mask.sum()
    print(f"\n  Signals after ALL active filters: {n_final:,}  "
          f"({n_final/max(n_base,1):.1%} of base)")

    return merged[final_mask][['Proba', 'Regime']]


# ===========================================================================
# Trade simulation with flexible exit (lifted from Step 6d)
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
    if not isinstance(all_data_df.index, pd.DatetimeIndex):
        all_data_df = all_data_df.copy()
        all_data_df.index = pd.to_datetime(all_data_df.index)

    px = (all_data_df.reset_index()
                       .rename(columns={all_data_df.index.name or 'index': 'Date'})
                       [['Date', 'Ticker', 'Open', 'Close', 'ATR_15']]
                       .set_index(['Ticker', 'Date'])
                       .sort_index())
    full_idx = pd.DatetimeIndex(sorted(all_data_df.index.unique()))

    sim_dates = sorted({d for (d, _) in proba_df.index.unique()})
    if not sim_dates:
        print("  No signals after filtering — nothing to simulate.")
        return pd.DataFrame(), pd.DataFrame(columns=['Date', 'Equity']).set_index('Date')

    sim_start = pd.Timestamp(min(sim_dates))
    last_sig  = pd.Timestamp(max(sim_dates))
    end_pos   = min(full_idx.get_loc(last_sig) + max_hold_days + 2, len(full_idx) - 1)
    sim_end   = full_idx[end_pos]
    iter_idx  = full_idx[(full_idx >= sim_start) & (full_idx <= sim_end)]
    print(f"\n  Simulation window: {sim_start.date()} → {sim_end.date()} "
          f"({len(iter_idx)} trading days)")

    cash = initial_capital
    open_pos, closed = [], []
    equity = []
    sig_dates_set = set(sim_dates)

    for cur in iter_idx:
        # 1. Evaluate exits
        still_open = []
        for pos in open_pos:
            key = (pos['ticker'], cur)
            if key not in px.index:
                still_open.append(pos)
                continue
            today_close = float(px.at[key, 'Close'])
            today_proba = float(proba_df.loc[(cur, pos['ticker']), 'Proba']
                                if (cur, pos['ticker']) in proba_df.index else np.nan)
            days_held = (cur - pos['buy_date']).days
            pos['hwm'] = max(pos.get('hwm', pos['buy_price']), today_close)

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

            sell_value = pos['shares'] * today_close
            sell_cost  = sell_value * transaction_cost
            cash += sell_value - sell_cost
            pnl = sell_value - pos['cost_basis'] - pos['entry_cost'] - sell_cost
            closed.append({
                **pos, 'sell_date': cur, 'sell_price': today_close,
                'pnl': pnl, 'pct': pnl / pos['cost_basis'],
                'win': pnl > 0, 'days_held': days_held,
                'exit_reason': exit_reason,
            })
        open_pos = still_open

        # 2. New entries (already pre-filtered through confluence)
        if cur in sig_dates_set:
            today_signals = (proba_df.xs(cur, level='Date')
                                       .sort_values('Proba', ascending=False))
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
                            'ticker': tk, 'regime': int(row['Regime']),
                            'entry_proba': float(row['Proba']),
                            'signal_date': cur, 'buy_date': buy_date,
                            'buy_price': buy_price, 'shares': shares,
                            'cost_basis': cb, 'entry_cost': ec,
                            'entry_atr': atr,
                            'stop_loss': buy_price - stop_loss_atr_mult * atr,
                            'take_profit': buy_price + take_profit_atr_mult * atr,
                            'hwm': buy_price,
                        })

        # 3. Mark-to-market
        mv = sum(p['shares'] * float(px.at[(p['ticker'], cur), 'Close'])
                 if (p['ticker'], cur) in px.index else p['cost_basis']
                 for p in open_pos)
        equity.append((cur, cash + mv))

    trades_df = pd.DataFrame(closed)
    equity_df = pd.DataFrame(equity, columns=['Date', 'Equity']).set_index('Date')
    return trades_df, equity_df


# ===========================================================================
# Metrics + breakdown
# ===========================================================================

def compute_metrics(trades_df, equity_df, initial_capital=INITIAL_CAPITAL):
    if trades_df.empty:
        print("\nNo trades closed — strategy never entered.")
        return {}, pd.DataFrame()

    n = len(trades_df)
    wins = trades_df[trades_df['win']]
    losses = trades_df[~trades_df['win']]
    pf = (-wins['pnl'].sum() / losses['pnl'].sum()) if losses['pnl'].sum() < 0 else float('inf')
    final = float(equity_df['Equity'].iloc[-1])
    yrs = max((equity_df.index[-1] - equity_df.index[0]).days / 365.25, 1e-9)
    cagr = (final / initial_capital) ** (1/yrs) - 1
    daily = equity_df['Equity'].pct_change().dropna()
    sharpe = (daily.mean() / daily.std()) * np.sqrt(252) if daily.std() > 0 else 0
    max_dd = (equity_df['Equity'] / equity_df['Equity'].cummax() - 1).min()

    metrics = {
        'trades': n, 'hit_rate': len(wins) / n,
        'avg_days_held': trades_df['days_held'].mean(),
        'avg_win_pct': wins['pct'].mean() if len(wins) else 0,
        'avg_loss_pct': losses['pct'].mean() if len(losses) else 0,
        'profit_fact': pf,
        'total_return': final / initial_capital - 1,
        'cagr': cagr, 'sharpe': sharpe, 'max_dd': max_dd,
        'final_equity': final, 'years': yrs,
    }

    print(f"\n=== Confluence-filter strategy performance ===")
    print(f"  Trades:                {n:,}")
    print(f"  Hit rate:              {metrics['hit_rate']:.1%}")
    print(f"  Avg days held:         {metrics['avg_days_held']:.1f}")
    print(f"  Avg win:               {metrics['avg_win_pct']:+.2%}")
    print(f"  Avg loss:              {metrics['avg_loss_pct']:+.2%}")
    print(f"  Profit factor:         {metrics['profit_fact']:.2f}")
    print(f"  Total return:          {metrics['total_return']:+.2%}  over {yrs:.2f} years")
    print(f"  CAGR:                  {metrics['cagr']:+.2%}")
    print(f"  Sharpe (daily):        {metrics['sharpe']:.2f}")
    print(f"  Max drawdown:          {metrics['max_dd']:.2%}")
    print(f"  Final equity:          ${metrics['final_equity']:,.2f}")

    by_reason = trades_df.groupby('exit_reason').agg(
        count=('pnl', 'size'), hit_rate=('win', 'mean'),
        avg_pct=('pct', 'mean'), avg_days=('days_held', 'mean'),
        total_pnl=('pnl', 'sum'),
    ).sort_values('total_pnl', ascending=False)
    print(f"\n=== Exit-reason breakdown ===")
    print(by_reason.to_string())
    return metrics, by_reason


# ===========================================================================
# Run it
# ===========================================================================
print("[1/6] Loading per-regime models ...")
models = load_regime_models()

print("\n[2/6] Pre-computing daily probabilities ...")
proba_df = precompute_probabilities(test_data, models)

print("\n[3/6] Computing confluence filters ...")
filter_df = compute_filter_table(test_data)

print("\n[4/6] Applying confluence filters to signals ...")
filtered = apply_confluence_filters(proba_df, filter_df)

print("\n[5/6] Simulating with filtered signals ...")
t0 = time.time()
trades_df, equity_df = simulate_flexible_exit(filtered, All_data_DF)
print(f"  Done in {time.time()-t0:.1f}s.  {len(trades_df):,} trades closed.")

print("\n[6/6] Computing metrics + saving ...")
metrics, by_reason = compute_metrics(trades_df, equity_df)

trades_df.to_csv(os.path.join(MODEL_DIR, 'Backtest_Confluence_Trades.csv'), index=False)
equity_df.to_csv(os.path.join(MODEL_DIR, 'Backtest_Confluence_Equity.csv'))
if not by_reason.empty:
    by_reason.to_csv(os.path.join(MODEL_DIR, 'Backtest_Confluence_ExitReasons.csv'))

if not equity_df.empty:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity_df.index, equity_df['Equity'], color='#1f77b4', lw=2)
    ax.set_title('Confluence-Filter + Flexible-Exit Strategy')
    ax.set_xlabel('Date'); ax.set_ylabel('Equity ($)'); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'Backtest_Confluence_Equity.png'), dpi=140)
    plt.show()
print(f"  Saved CSVs and chart to {MODEL_DIR}")
