#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 7e — Walk-Forward Validation (Confluence Filter + Flexible Exit)
======================================================================

Same fold structure as Step 7 / 7d — but applies Step 6e's confluence
filter to entry signals BEFORE running the flexible-exit simulator.

The critical question this answers
----------------------------------
Step 7d showed:
  Fold 1 (bear 2021-22):     -2.4%  (868 stops vs 180 take-profits — disaster)
  Fold 2 (bull 2023-mid-24): +29%   (works in bulls)
  Fold 3 (bull 2024):        +16%

If the confluence filter (specifically MARKET_TREND) is doing its job,
Fold 1 should improve from -2.4% to ~0% or better — even if Fold 2 and 3
get smaller. The filter is bear-market insurance.

If Step 7e's mean Sharpe ≥ Step 7d's mean Sharpe AND Fold 1 is positive
(or much closer to zero), the filter is worth keeping.
If all three folds get worse, the filter approach itself is wrong and
we should drop it.

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

OUTPUT_DIR = '/Users/leemingjun/Desktop/Random Forest Model/2025 S&P 500 Prediction (Return Variable)'

FOLDS = [
    ('2018-01-01',  '2021-06-30',   '2021-07-12',   '2022-12-31'),
    ('2019-01-01',  '2022-12-31',   '2023-01-13',   '2024-06-30'),
    ('2020-01-01',  '2023-12-31',   '2024-01-15',   '2024-08-06'),
]

# --- Entry / sizing ---
INITIAL_CAPITAL     = 5_000.0
ENTRY_PROBA         = 0.55
MAX_NEW_PER_DAY     = 10
MAX_OPEN_PER_TICKER = 1
TRANSACTION_COST    = 0.0005
EXCLUDE_REGIMES_AT  = (4,)
SKIP_TINY_REGIMES   = True

# --- Confluence filters (Experiment A defaults: just market + stock trend) ---
USE_MARKET_TREND   = True
USE_STOCK_TREND    = True
USE_VOL_REGIME     = False
USE_RSI_CEILING    = False
USE_MOMENTUM       = False

MARKET_TREND_THRESH = 1.00
STOCK_TREND_THRESH  = 1.00
VOL_REGIME_PCTILE   = 0.85
RSI_CEILING         = 75
MOMENTUM_LOOKBACK   = 20
MOMENTUM_THRESH     = 0.0

# --- Flexible exit (tuned defaults) ---
EXIT_PROBA              = 0.45
STOP_LOSS_ATR_MULT      = 1.0
TAKE_PROFIT_ATR_MULT    = 3.0
TRAILING_STOP_ATR_MULT  = 3.0
MAX_HOLD_DAYS           = 14


# ===========================================================================
# Imports
# ===========================================================================

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score

try:
    import lightgbm as lgb
except ImportError:
    raise RuntimeError("lightgbm required.  pip install lightgbm")

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_OK = True
except ImportError:
    HMM_OK = False
    print("WARNING: hmmlearn not installed — falling back to GMM for regime fit")
    from sklearn.mixture import GaussianMixture


REGIME_FEATURES = [
    'ATR_Ratio', 'ADX_15', 'RSI_15', 'SMA_ratio', '%B', 'SMA_Volume_Ratio',
]

LGBM_PARAMS = {
    'num_leaves':        [15, 31],
    'learning_rate':     [0.03, 0.05],
    'n_estimators':      [200, 300],
    'min_child_samples': [100, 200, 300],
    'colsample_bytree':  [0.7, 1.0],
    'subsample':         [0.7, 1.0],
    'reg_lambda':        [0.1, 1.0],
}


# ===========================================================================
# Per-fold training helpers (same as Step 7d)
# ===========================================================================

def fit_regimes(train_df, k=5, seed=42):
    feats = [c for c in REGIME_FEATURES if c in train_df.columns]
    X = train_df[feats].dropna().values.astype(np.float32)
    if len(X) < 1000:
        raise ValueError(f"Not enough rows ({len(X)}) for regime fit")
    scaler = StandardScaler().fit(X)
    Xz = scaler.transform(X)
    if HMM_OK:
        lengths = (train_df.dropna(subset=feats)
                           .groupby('Ticker', sort=False).size().tolist())
        model = GaussianHMM(n_components=k, covariance_type='full',
                            n_iter=30, random_state=seed).fit(Xz, lengths=lengths)
    else:
        model = GaussianMixture(n_components=k, covariance_type='full',
                                random_state=seed, n_init=1, max_iter=100).fit(Xz)
    return model, scaler, feats


def predict_regimes(model, scaler, feats, df):
    valid_mask = df[feats].notna().all(axis=1).values
    out = np.full(len(df), np.nan)
    if not valid_mask.any():
        return out
    valid_df = df.iloc[valid_mask]
    Xz = scaler.transform(valid_df[feats].values.astype(np.float32))
    if HMM_OK:
        lengths = valid_df.groupby('Ticker', sort=False).size().tolist()
        labels = model.predict(Xz, lengths=lengths)
    else:
        labels = model.predict(Xz)
    out[valid_mask] = labels
    return out


def relabel_by_vol(labels, vol_series):
    labels = np.asarray(labels, dtype=float)
    vol    = np.asarray(vol_series, dtype=float)
    valid  = (labels >= 0) & ~np.isnan(labels) & ~np.isnan(vol)
    if not valid.any():
        return np.full_like(labels, np.nan)
    df = pd.DataFrame({'l': labels[valid].astype(int), 'v': vol[valid]})
    means = df.groupby('l')['v'].mean().sort_values()
    mapping = dict(zip(means.index.tolist(), range(len(means))))
    out = np.full_like(labels, np.nan)
    for old, new in mapping.items():
        out[(labels == old) & valid] = new
    return out


def train_lgbm_per_regime(train_df, technical_vars, regime_col='Regime',
                          target_col='Target_Direction',
                          min_rows=3_000, n_iter=15, cv_splits=3, seed=42):
    bundles = {}
    for r in sorted(train_df[regime_col].dropna().unique()):
        if r in EXCLUDE_REGIMES_AT:
            continue
        sub = train_df[train_df[regime_col] == r].dropna(
            subset=technical_vars + [target_col]
        )
        if SKIP_TINY_REGIMES and len(sub) < min_rows:
            print(f"      regime {int(r)}: skipped ({len(sub)} rows < {min_rows})")
            continue
        X = sub[technical_vars].values
        y = sub[target_col].values.astype(int)
        spw = (y == 0).sum() / max((y == 1).sum(), 1)
        est = lgb.LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1,
                                 scale_pos_weight=spw)
        srch = RandomizedSearchCV(est, LGBM_PARAMS, n_iter=n_iter,
                                  scoring='roc_auc',
                                  cv=TimeSeriesSplit(n_splits=cv_splits),
                                  n_jobs=1, random_state=seed)
        srch.fit(X, y)
        cal = CalibratedClassifierCV(srch.best_estimator_, method='isotonic',
                                     cv=2, n_jobs=1)
        cal.fit(X, y)
        bundles[int(r)] = {
            'model': cal, 'features': technical_vars,
            'cv_score': srch.best_score_,
        }
        print(f"      regime {int(r)}: cv_auc={srch.best_score_:.3f}  "
              f"rows={len(sub):,}")
    return bundles


# ===========================================================================
# Pre-compute probabilities + confluence filter
# ===========================================================================

def precompute_probabilities(test_df, bundles, group_col='Regime'):
    chunks = []
    for r, b in bundles.items():
        rows = test_df[test_df[group_col] == r].dropna(subset=b['features'])
        if rows.empty:
            continue
        proba = b['model'].predict_proba(rows[b['features']].values)[:, 1]
        chunks.append(pd.DataFrame({
            'Date':   rows.index,
            'Ticker': rows['Ticker'].values,
            'Proba':  proba,
            'Regime': r,
        }))
    if not chunks:
        return None
    proba_df = pd.concat(chunks, ignore_index=True)
    proba_df['Date'] = pd.to_datetime(proba_df['Date'])
    return proba_df.set_index(['Date', 'Ticker'])


def compute_filter_table(test_data, momentum_lookback=MOMENTUM_LOOKBACK):
    df = test_data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df['_stock_trend'] = df['SMA_ratio'] > STOCK_TREND_THRESH
    market_trend = df.groupby(df.index)['SMA_ratio'].transform('median')
    df['_market_trend'] = market_trend > MARKET_TREND_THRESH
    vol_pctile = df.groupby(df.index)['ATR_Ratio'].transform(lambda x: x.rank(pct=True))
    df['_vol_regime'] = vol_pctile < VOL_REGIME_PCTILE
    df['_not_overbought'] = df['RSI_15'] < RSI_CEILING
    g = df.groupby('Ticker', group_keys=False)
    df['_mom_return'] = g['Close'].pct_change(momentum_lookback)
    df['_positive_momentum'] = df['_mom_return'] > MOMENTUM_THRESH

    df = df.reset_index().rename(columns={df.index.name or 'index': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    return df.set_index(['Date', 'Ticker'])[
        ['_market_trend', '_stock_trend', '_vol_regime',
         '_not_overbought', '_positive_momentum']
    ]


def apply_confluence_filters(proba_df, filter_df,
                             entry_proba=ENTRY_PROBA):
    merged = proba_df.join(filter_df, how='left')
    base_mask = merged['Proba'] >= entry_proba
    final_mask = base_mask.copy()
    pass_summary = {}
    for name, on, col in [
        ('market_trend',       USE_MARKET_TREND, '_market_trend'),
        ('stock_trend',        USE_STOCK_TREND,  '_stock_trend'),
        ('vol_regime',         USE_VOL_REGIME,   '_vol_regime'),
        ('not_overbought',     USE_RSI_CEILING,  '_not_overbought'),
        ('positive_momentum',  USE_MOMENTUM,     '_positive_momentum'),
    ]:
        if not on:
            continue
        passes = merged[col].fillna(False)
        pass_summary[name] = (base_mask & passes).sum() / max(base_mask.sum(), 1)
        final_mask &= passes
    return merged[final_mask][['Proba', 'Regime']], pass_summary


# ===========================================================================
# Flexible-exit simulator (lifted from Step 6d/6e)
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
        return pd.DataFrame(), pd.DataFrame(columns=['Date', 'Equity']).set_index('Date')

    sim_start = pd.Timestamp(min(sim_dates))
    last_sig  = pd.Timestamp(max(sim_dates))
    end_pos   = min(full_idx.get_loc(last_sig) + max_hold_days + 2, len(full_idx) - 1)
    sim_end   = full_idx[end_pos]
    iter_idx  = full_idx[(full_idx >= sim_start) & (full_idx <= sim_end)]

    cash = initial_capital
    open_pos, closed = [], []
    equity = []
    sig_dates_set = set(sim_dates)

    for cur in iter_idx:
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

        mv = sum(p['shares'] * float(px.at[(p['ticker'], cur), 'Close'])
                 if (p['ticker'], cur) in px.index else p['cost_basis']
                 for p in open_pos)
        equity.append((cur, cash + mv))

    trades_df = pd.DataFrame(closed)
    equity_df = pd.DataFrame(equity, columns=['Date', 'Equity']).set_index('Date')
    return trades_df, equity_df


# ===========================================================================
# Per-fold metrics
# ===========================================================================

def compute_fold_metrics(trades, equity, initial_capital):
    if trades is None or trades.empty:
        return {}, pd.DataFrame()
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
    metrics = {
        'trades': n, 'hit_rate': wins / n,
        'avg_days_held': trades['days_held'].mean(),
        'profit_fact': pf, 'total_return': final / initial_capital - 1,
        'cagr': cagr, 'sharpe': sharpe, 'max_dd': max_dd, 'final_$': final,
    }
    by_reason = trades.groupby('exit_reason').agg(
        count=('pnl', 'size'), hit_rate=('win', 'mean'),
        avg_pct=('pct', 'mean'), total_pnl=('pnl', 'sum'),
    ).sort_values('total_pnl', ascending=False)
    return metrics, by_reason


# ===========================================================================
# Run all folds
# ===========================================================================

def run_walk_forward_confluence(All_data_DF, technical_variables, folds=FOLDS):
    if not isinstance(All_data_DF.index, pd.DatetimeIndex):
        All_data_DF = All_data_DF.copy()
        All_data_DF.index = pd.to_datetime(All_data_DF.index)
    All_data_DF = All_data_DF.sort_index()

    all_metrics = []
    all_exit_reasons = []
    all_equity_curves = {}
    all_filter_passes = []

    active_filters = [name for name, on in [
        ('market_trend', USE_MARKET_TREND),
        ('stock_trend',  USE_STOCK_TREND),
        ('vol_regime',   USE_VOL_REGIME),
        ('not_overbought', USE_RSI_CEILING),
        ('positive_momentum', USE_MOMENTUM),
    ] if on]
    print(f"Active confluence filters: {active_filters}")

    for i, (tr_start, tr_end, te_start, te_end) in enumerate(folds, 1):
        print(f"\n{'='*78}\nFOLD {i}/{len(folds)}: "
              f"train {tr_start}→{tr_end}  |  test {te_start}→{te_end}\n{'='*78}")
        t0 = time.time()

        train = All_data_DF.loc[tr_start:tr_end].copy()
        test  = All_data_DF.loc[te_start:te_end].copy()

        print("  [a] Fit regime model on training slice ...")
        rmodel, rscaler, rfeats = fit_regimes(train)
        train['Regime'] = predict_regimes(rmodel, rscaler, rfeats, train)
        test ['Regime'] = predict_regimes(rmodel, rscaler, rfeats, test)
        train['Regime'] = relabel_by_vol(train['Regime'].values,
                                         train['ATR_Ratio'].values)
        test ['Regime'] = relabel_by_vol(test ['Regime'].values,
                                         test ['ATR_Ratio'].values)

        print("  [b] Train LightGBM per regime ...")
        bundles = train_lgbm_per_regime(train, technical_variables)

        print("  [c] Pre-compute test probabilities ...")
        proba_df = precompute_probabilities(test, bundles)
        if proba_df is None:
            print("      No probabilities; skipping fold.")
            continue

        print("  [d] Compute confluence filters ...")
        filter_df = compute_filter_table(test)
        filtered_proba, pass_summary = apply_confluence_filters(proba_df, filter_df)
        print(f"      Base signals: {(proba_df['Proba'] >= ENTRY_PROBA).sum():,}  "
              f"→ after filters: {len(filtered_proba):,}")
        for fname, frate in pass_summary.items():
            print(f"        {fname:20s}: {frate:.1%} pass rate")

        print("  [e] Run flexible-exit simulator on filtered signals ...")
        trades, equity = simulate_flexible_exit(filtered_proba, All_data_DF)

        m, by_reason = compute_fold_metrics(trades, equity, INITIAL_CAPITAL)
        m['fold']        = i
        m['train_start'] = tr_start
        m['train_end']   = tr_end
        m['test_start']  = te_start
        m['test_end']    = te_end
        m['elapsed_s']   = round(time.time() - t0, 1)
        all_metrics.append(m)

        if not by_reason.empty:
            by_reason['fold'] = i
            all_exit_reasons.append(by_reason.reset_index())

        if equity is not None and not equity.empty:
            all_equity_curves[i] = equity

        pass_summary['fold'] = i
        all_filter_passes.append(pass_summary)

        print(f"\n  Fold {i} result:")
        print(f"    return  = {m.get('total_return', 0):+.2%}")
        print(f"    cagr    = {m.get('cagr', 0):+.2%}")
        print(f"    sharpe  = {m.get('sharpe', 0):.2f}")
        print(f"    max_dd  = {m.get('max_dd', 0):.2%}")
        print(f"    trades  = {m.get('trades', 0)}")
        print(f"    elapsed = {m['elapsed_s']}s")
        if not by_reason.empty:
            print(f"    exit-reason mix:")
            print(by_reason.to_string())

    summary = pd.DataFrame(all_metrics)
    summary_path = os.path.join(OUTPUT_DIR, 'WalkForward_Confluence_Folds.csv')
    summary.to_csv(summary_path, index=False)
    print(f"\n{'='*78}\nWALK-FORWARD (CONFLUENCE + FLEXIBLE) SUMMARY\n{'='*78}")
    print(summary[['fold', 'test_start', 'test_end', 'trades', 'hit_rate',
                   'avg_days_held', 'total_return', 'cagr', 'sharpe',
                   'max_dd']].to_string(index=False))
    print(f"\n  Saved: {summary_path}")

    if all_exit_reasons:
        ex = pd.concat(all_exit_reasons, ignore_index=True)
        ex_path = os.path.join(OUTPUT_DIR, 'WalkForward_Confluence_ExitReasons.csv')
        ex.to_csv(ex_path, index=False)
        print(f"  Saved: {ex_path}")

    print("\nAggregated across folds:")
    print(f"  Mean total return:     {summary['total_return'].mean():+.2%}")
    print(f"  Mean CAGR:             {summary['cagr'].mean():+.2%}")
    print(f"  Mean Sharpe:           {summary['sharpe'].mean():.2f}")
    print(f"  Worst-fold return:     {summary['total_return'].min():+.2%}")
    print(f"  Worst-fold max DD:     {summary['max_dd'].min():.2%}")
    print(f"  Folds positive:        {(summary['total_return'] > 0).sum()}/{len(summary)}")

    print("\nComparison vs Step 7d (no filter):")
    print("  If Fold 1 went from -2.4% to >0% AND mean Sharpe improved → filter works")
    print("  If all folds got worse → drop the filter approach")

    if all_equity_curves:
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, eq in all_equity_curves.items():
            ax.plot(eq.index, eq['Equity'] / INITIAL_CAPITAL,
                    label=f'Fold {i}', lw=1.6)
        ax.axhline(1.0, color='k', ls=':', lw=0.8)
        ax.set_xlabel('Date'); ax.set_ylabel('Equity (×initial)')
        ax.set_title('Walk-forward — confluence filter + flexible exit')
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        out_png = os.path.join(OUTPUT_DIR,
                               'WalkForward_Confluence_EquityCurves.png')
        plt.savefig(out_png, dpi=140)
        plt.show()
        print(f"  Saved: {out_png}")

    return summary


# ===========================================================================
# Run it
# ===========================================================================
# Assumes All_data_DF and Technical_variables are in scope.

summary = run_walk_forward_confluence(All_data_DF, Technical_variables)
