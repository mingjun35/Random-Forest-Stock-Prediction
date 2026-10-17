#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 6c — Backtest Diagnostics
==============================

Three follow-up analyses on the Step 6b backtest output:

  1. SPY benchmark
       — fetches actual SPY (cap-weighted S&P 500) over the same
         window and compares total return, CAGR, Sharpe, drawdown.

  2. Transaction-cost sensitivity
       — re-prices every closed trade at 0 / 5 / 10 / 20 bps per
         side. Tells you how fragile the strategy is to real-world
         execution friction.

  3. Trade concentration / robustness
       — top wins, worst losses, monthly P&L, ticker concentration,
         per-regime performance, and win-rate by signal confidence.

Inputs (created by Step 6b):
       Backtest_Trades.csv
       Backtest_Equity.csv

Outputs:
       Backtest_vs_SPY.png
       Backtest_Concentration.png
       Backtest_CostSensitivity.csv
       Backtest_Diagnostics.txt   (full stdout dump)

Author: leemingjun
"""

import os
import sys
import io
import contextlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
sns.set()


# ===========================================================================
# Configuration
# ===========================================================================

MODEL_DIR  = '/Users/leemingjun/Desktop/Random Forest Model/2025 S&P 500 Prediction (Return Variable)'
TRADES_FP  = os.path.join(MODEL_DIR, 'Backtest_Trades.csv')
EQUITY_FP  = os.path.join(MODEL_DIR, 'Backtest_Equity.csv')

COST_LEVELS_BPS = [0, 5, 10, 20]      # per side
RISK_FREE_RATE  = 0.045               # ~4.5% annual T-bill
TRADING_DAYS    = 252


# ===========================================================================
# Tee stdout to a file so we can save the full diagnostic dump
# ===========================================================================

class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s); st.flush()
    def flush(self):
        for st in self.streams:
            st.flush()


# ===========================================================================
# Load Step 6b output
# ===========================================================================

def load_artifacts():
    if not os.path.exists(TRADES_FP):
        raise FileNotFoundError(f"Couldn't find {TRADES_FP} — run Step 6b first.")

    trades = pd.read_csv(TRADES_FP, parse_dates=['signal_date', 'buy_date', 'sell_date'])
    equity = pd.read_csv(EQUITY_FP, parse_dates=['Date']).set_index('Date')

    initial_capital = float(equity['Equity'].iloc[0])
    print(f"Loaded {len(trades):,} trades, {len(equity):,} equity rows.")
    print(f"Initial capital inferred: ${initial_capital:,.2f}")
    print(f"Window: {equity.index.min().date()}  →  {equity.index.max().date()}")
    return trades, equity, initial_capital


# ===========================================================================
# 1.  SPY benchmark (real cap-weighted S&P 500)
# ===========================================================================

def benchmark_vs_spy(equity, initial_capital):
    print("\n" + "=" * 78)
    print("1. STRATEGY vs SPY (real cap-weighted S&P 500)")
    print("=" * 78)

    start = equity.index.min()
    end   = equity.index.max() + pd.Timedelta(days=2)

    print(f"Fetching SPY from yfinance: {start.date()}  →  {end.date()} ...")
    spy_raw = yf.download('SPY', start=start, end=end, progress=False, auto_adjust=True)
    if spy_raw.empty:
        print("  WARNING: yfinance returned no data for SPY. "
              "Skipping SPY benchmark.")
        return None

    # Newer yfinance returns multi-index columns; flatten if needed
    spy_close = spy_raw['Close']
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.iloc[:, 0]
    spy_close = spy_close.dropna()

    # Align to strategy equity dates (so the comparison plot is on common dates)
    spy_close = spy_close.reindex(equity.index, method='ffill').dropna()
    spy_norm  = (spy_close / spy_close.iloc[0]) * initial_capital

    # ---- Metrics ------------------------------------------------------
    years = (equity.index.max() - equity.index.min()).days / 365.25

    def metrics(eq_series, label):
        total_ret = float(eq_series.iloc[-1]) / float(eq_series.iloc[0]) - 1
        cagr      = (1 + total_ret) ** (1/years) - 1
        daily     = eq_series.pct_change().dropna()
        sharpe    = ((daily.mean() - RISK_FREE_RATE/TRADING_DAYS) / daily.std()) * np.sqrt(TRADING_DAYS) if daily.std() > 0 else 0
        max_dd    = (eq_series / eq_series.cummax() - 1).min()
        return {
            'Total Return': total_ret,
            'CAGR':         cagr,
            'Sharpe':       sharpe,
            'Max DD':       max_dd,
            'Final $':      float(eq_series.iloc[-1]),
        }

    strat_m = metrics(equity['Equity'], 'Strategy')
    spy_m   = metrics(spy_norm,         'SPY')

    comp = pd.DataFrame({'Strategy': strat_m, 'SPY': spy_m})
    comp['Alpha'] = comp['Strategy'] - comp['SPY']
    comp_pretty = comp.copy()
    for col in comp_pretty.columns:
        comp_pretty[col] = comp_pretty[col].apply(
            lambda x: f'${x:,.2f}' if isinstance(x, float) and abs(x) > 100
                      else (f'{x:.2%}' if abs(x) < 10 else f'{x:.2f}')
        )
    print("\n" + comp_pretty.to_string())

    # ---- Plot ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity.index, equity['Equity'],  label='Strategy',
            color='#1f77b4', lw=2)
    ax.plot(spy_norm.index, spy_norm.values, label='SPY buy-and-hold',
            color='#7f7f7f', lw=1.5, ls='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity ($)')
    ax.set_title(f'Strategy vs SPY — '
                 f'Strategy +{strat_m["Total Return"]:.1%}  vs  '
                 f'SPY +{spy_m["Total Return"]:.1%}')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out_png = os.path.join(MODEL_DIR, 'Backtest_vs_SPY.png')
    plt.savefig(out_png, dpi=140)
    plt.show()
    print(f"\n  Saved: {out_png}")

    return comp


# ===========================================================================
# 2.  Cost sensitivity — re-price each closed trade at different cost levels
# ===========================================================================

def cost_sensitivity(trades, initial_capital):
    print("\n" + "=" * 78)
    print("2. TRANSACTION-COST SENSITIVITY")
    print("=" * 78)

    rows = []
    for cost_bps in COST_LEVELS_BPS:
        c = cost_bps / 10_000.0
        gross   = trades['shares'] * (trades['sell_price'] - trades['buy_price'])
        round_t = (trades['cost_basis'] + trades['shares'] * trades['sell_price']) * c
        adj_pnl = gross - round_t
        wins    = adj_pnl > 0
        loss_sum = adj_pnl[~wins].sum()
        win_sum  = adj_pnl[wins].sum()

        rows.append({
            'cost_per_side_bps': cost_bps,
            'total_pnl_$':       adj_pnl.sum(),
            'total_return':      adj_pnl.sum() / initial_capital,
            'hit_rate':          wins.mean(),
            'avg_win_$':         adj_pnl[wins].mean()  if wins.any() else 0,
            'avg_loss_$':        adj_pnl[~wins].mean() if (~wins).any() else 0,
            'profit_factor':     -win_sum / loss_sum if loss_sum < 0 else float('inf'),
        })
    sens = pd.DataFrame(rows)
    sens_pretty = sens.copy()
    sens_pretty['total_pnl_$']   = sens_pretty['total_pnl_$'].map(lambda x: f'${x:,.2f}')
    sens_pretty['total_return']  = sens_pretty['total_return'].map(lambda x: f'{x:+.2%}')
    sens_pretty['hit_rate']      = sens_pretty['hit_rate'].map(lambda x: f'{x:.1%}')
    sens_pretty['avg_win_$']     = sens_pretty['avg_win_$'].map(lambda x: f'${x:+.2f}')
    sens_pretty['avg_loss_$']    = sens_pretty['avg_loss_$'].map(lambda x: f'${x:+.2f}')
    sens_pretty['profit_factor'] = sens_pretty['profit_factor'].map(lambda x: f'{x:.2f}')
    print("\n" + sens_pretty.to_string(index=False))

    # Edge erosion summary
    base    = sens.iloc[0]['total_return']
    at_5bp  = sens[sens.cost_per_side_bps == 5 ].iloc[0]['total_return']
    at_10bp = sens[sens.cost_per_side_bps == 10].iloc[0]['total_return']
    at_20bp = sens[sens.cost_per_side_bps == 20].iloc[0]['total_return']
    print(f"\n  Edge erosion vs zero-cost baseline:")
    print(f"    @ 5 bps/side:   {(base-at_5bp)/base:.1%} of edge consumed")
    print(f"    @ 10 bps/side:  {(base-at_10bp)/base:.1%} of edge consumed")
    print(f"    @ 20 bps/side:  {(base-at_20bp)/base:.1%} of edge consumed")
    if at_10bp <= 0:
        print(f"    WARNING: strategy is unprofitable at 10 bps round-trip costs.")

    sens.to_csv(os.path.join(MODEL_DIR, 'Backtest_CostSensitivity.csv'), index=False)
    return sens


# ===========================================================================
# 3.  Trade concentration / robustness
# ===========================================================================

def concentration_analysis(trades):
    print("\n" + "=" * 78)
    print("3. TRADE CONCENTRATION & ROBUSTNESS")
    print("=" * 78)

    # ---- 3a. Top wins / worst losses ----------------------------------
    print("\n3a. Top 10 winning trades:")
    top_wins = trades.nlargest(10, 'pnl')[
        ['ticker', 'regime', 'buy_date', 'sell_date', 'proba', 'pnl', 'pct']
    ]
    print(top_wins.to_string(index=False))

    print("\n3b. Top 10 losing trades:")
    top_losses = trades.nsmallest(10, 'pnl')[
        ['ticker', 'regime', 'buy_date', 'sell_date', 'proba', 'pnl', 'pct']
    ]
    print(top_losses.to_string(index=False))

    total_pnl    = trades['pnl'].sum()
    top10_share  = top_wins['pnl'].sum() / total_pnl
    print(f"\n  Top 10 wins contributed {top10_share:.1%} of total P&L "
          f"(higher = more fragile / lucky-driven)")

    # ---- 3b. Monthly P&L ----------------------------------------------
    trades['month'] = trades['sell_date'].dt.to_period('M').astype(str)
    monthly = trades.groupby('month').agg(
        trades=('ticker', 'count'),
        hit_rate=('win', 'mean'),
        pnl=('pnl', 'sum'),
        avg_pct=('pct', 'mean'),
    ).reset_index()
    print(f"\n3c. Monthly performance ({len(monthly)} months):")
    print(monthly.to_string(index=False))
    pos_months = (monthly['pnl'] > 0).sum()
    print(f"\n  Profitable months: {pos_months}/{len(monthly)} "
          f"({pos_months/len(monthly):.0%})")

    # ---- 3c. Top contributing tickers ---------------------------------
    by_ticker = trades.groupby('ticker').agg(
        trades=('ticker', 'count'),
        hit_rate=('win', 'mean'),
        total_pnl=('pnl', 'sum'),
        avg_pct=('pct', 'mean'),
    ).sort_values('total_pnl', ascending=False)

    print(f"\n3d. Top 15 contributing tickers (by total $PnL):")
    print(by_ticker.head(15).to_string())
    print(f"\n3e. Bottom 10 tickers (biggest drag):")
    print(by_ticker.tail(10).to_string())
    n_winners_needed = (by_ticker['total_pnl'].sort_values(ascending=False).cumsum()
                        > total_pnl).idxmax()
    rank = list(by_ticker.sort_values('total_pnl', ascending=False).index).index(n_winners_needed) + 1
    print(f"\n  {rank} top tickers explain 100% of profit "
          f"(out of {len(by_ticker)} traded). "
          f"Lower = more concentrated = more fragile.")

    # ---- 3d. By regime ------------------------------------------------
    by_regime = trades.groupby('regime').agg(
        trades=('ticker', 'count'),
        hit_rate=('win', 'mean'),
        avg_pct=('pct', 'mean'),
        total_pnl=('pnl', 'sum'),
    )
    print(f"\n3f. Performance by regime:")
    print(by_regime.to_string())

    # ---- 3e. By signal confidence ------------------------------------
    trades['proba_bucket'] = pd.cut(trades['proba'],
                                    bins=[0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0])
    by_proba = trades.groupby('proba_bucket', observed=True).agg(
        trades=('ticker', 'count'),
        hit_rate=('win', 'mean'),
        avg_pct=('pct', 'mean'),
        total_pnl=('pnl', 'sum'),
    )
    print(f"\n3g. Performance by signal confidence "
          f"(higher proba should = higher hit rate):")
    print(by_proba.to_string())

    # ---- Plot: cumulative P&L with concentration markers --------------
    fig, ax = plt.subplots(figsize=(13, 6))
    sorted_trades = trades.sort_values('sell_date').reset_index(drop=True)
    sorted_trades['cum_pnl'] = sorted_trades['pnl'].cumsum()
    ax.plot(sorted_trades['sell_date'], sorted_trades['cum_pnl'],
            color='#1f77b4', lw=1.4)

    # Highlight outsized single trades (top/bottom 1% by absolute pnl)
    big_thresh = sorted_trades['pnl'].abs().quantile(0.99)
    big_wins   = sorted_trades[sorted_trades['pnl'] >  big_thresh]
    big_losses = sorted_trades[sorted_trades['pnl'] < -big_thresh]
    ax.scatter(big_wins['sell_date'], big_wins['cum_pnl'],
               color='green', s=40, alpha=0.7,
               label=f'Top 1% wins (n={len(big_wins)})')
    ax.scatter(big_losses['sell_date'], big_losses['cum_pnl'],
               color='red', s=40, alpha=0.7,
               label=f'Top 1% losses (n={len(big_losses)})')
    ax.set_xlabel('Sell date')
    ax.set_ylabel('Cumulative P&L ($)')
    ax.set_title('Cumulative P&L over time — concentration check')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(MODEL_DIR, 'Backtest_Concentration.png')
    plt.savefig(out_png, dpi=140)
    plt.show()
    print(f"\n  Saved: {out_png}")

    return {
        'top10_share':      top10_share,
        'pos_months_pct':   pos_months / len(monthly),
        'n_tickers_for_100pct_pnl': rank,
    }


# ===========================================================================
# Run all three
# ===========================================================================

if __name__ == '__main__':
    log_path = os.path.join(MODEL_DIR, 'Backtest_Diagnostics.txt')
    log_buf  = open(log_path, 'w')

    with contextlib.redirect_stdout(_Tee(sys.stdout, log_buf)):
        trades, equity, initial_capital = load_artifacts()

        spy_comp = benchmark_vs_spy(equity, initial_capital)
        cost_sens = cost_sensitivity(trades, initial_capital)
        robust    = concentration_analysis(trades)

        print("\n" + "=" * 78)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 78)
        if spy_comp is not None:
            alpha = spy_comp.loc['Total Return', 'Alpha']
            print(f"  Alpha vs SPY:                 {alpha:+.2%}")
        print(f"  Top 10 wins share of P&L:     {robust['top10_share']:.1%}  "
              f"(<30% is healthy)")
        print(f"  Profitable months:            {robust['pos_months_pct']:.0%}  "
              f"(>60% is healthy)")
        print(f"  Tickers needed for 100% PnL:  {robust['n_tickers_for_100pct_pnl']}  "
              f"(higher = more diversified)")

    log_buf.close()
    print(f"\nFull log saved to: {log_path}")
