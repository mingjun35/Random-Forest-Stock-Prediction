#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 7 — Walk-Forward Validation
================================

Stress-tests the strategy by re-training and re-backtesting it on multiple
non-overlapping windows. The goal: confirm the Aug 2024 → Apr 2026 result
isn't a one-window fluke that happened to land on a lucky market regime.

How it works:
   For each fold:
     1. Define a TRAIN window (~3-5 years) and a TEST window (~6-12 months)
        immediately after, with a 6-day embargo gap.
     2. Refit the regime model on the train slice.
     3. Refit per-regime LightGBM on the train slice.
     4. Generate signals + simulate trades on the test slice.
     5. Record per-fold metrics.

Required inputs already in scope (from your existing pipeline):
     All_data_DF       — full OHLCV + Ticker + Target_Direction + features
     Technicals_DF     — same as Step 2 output
     Technical_variables  — feature list

Outputs:
     WalkForward_Folds.csv          — one row per fold with metrics
     WalkForward_EquityCurves.png   — overlay of equity curves across folds

Note this is computationally heavy: each fold re-runs Step 3b + Step 5b on
its train slice. Plan ~10–25 minutes per fold on 1M-row data, depending on
how many regimes survive the MIN_ROWS_PER_GROUP filter.

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

# Define the folds. Each entry is (train_start, train_end, test_start, test_end).
# Embargo of ~6 trading days (≈ 9 calendar days) between train_end and test_start.
FOLDS = [
    # train_start    train_end       test_start      test_end
    ('2018-01-01',  '2021-06-30',   '2021-07-12',   '2022-12-31'),  # train ~3.5y, test 1.5y
    ('2019-01-01',  '2022-12-31',   '2023-01-13',   '2024-06-30'),  # train  4y,  test 1.5y
    ('2020-01-01',  '2023-12-31',   '2024-01-15',   '2024-08-06'),  # train  4y,  test ~7m
    # Optionally add the live window for completeness:
    # ('2018-01-01', '2024-08-06', '2024-08-15', '2026-04-16'),
]

INITIAL_CAPITAL     = 5_000.0
HOLDING_DAYS        = 7
TRANSACTION_COST    = 0.0005
MAX_SIGNALS_PER_DAY = 10
MAX_OPEN_PER_TICKER = 1
MIN_PROBA_OVERRIDE  = 0.75
EXCLUDE_REGIMES_AT  = (4,)              # always exclude obviously broken
SKIP_TINY_REGIMES   = True              # also skip regimes with <3000 train rows


# ===========================================================================
# Lean per-fold pipeline (avoids re-importing all of Step 3b/5b/6b)
# ===========================================================================

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_score

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


def fit_regimes(train_df, k=5, seed=42):
    """Fit HMM (or GMM fallback) on the training slice; return labels per row."""
    feats = [c for c in REGIME_FEATURES if c in train_df.columns]
    X = train_df[feats].dropna().values.astype(np.float32)
    if len(X) < 1000:
        raise ValueError(f"Not enough rows ({len(X)}) to fit a {k}-regime model")
    scaler = StandardScaler().fit(X)
    Xz = scaler.transform(X)

    if HMM_OK:
        # Per-ticker lengths so HMM doesn't compute fake transitions across tickers
        lengths = (train_df.dropna(subset=feats)
                           .groupby('Ticker', sort=False).size().tolist())
        model = GaussianHMM(n_components=k, covariance_type='full',
                            n_iter=30, random_state=seed).fit(Xz, lengths=lengths)
    else:
        model = GaussianMixture(n_components=k, covariance_type='full',
                                random_state=seed, n_init=1, max_iter=100).fit(Xz)
    return model, scaler, feats


def predict_regimes(model, scaler, feats, df):
    """
    Returns a numpy array of length len(df) with regime labels per row,
    or NaN where features are missing. Uses POSITIONAL assignment so it
    works correctly even when df has a duplicated index (Date repeats
    across tickers in the user's pipeline).
    """
    valid_mask = df[feats].notna().all(axis=1).values
    out = np.full(len(df), np.nan)
    if not valid_mask.any():
        return out

    valid_df = df.iloc[valid_mask]
    Xz = scaler.transform(valid_df[feats].values.astype(np.float32))
    if HMM_OK:
        lengths = valid_df.groupby('Ticker', sort=False).size().tolist()
        labels  = model.predict(Xz, lengths=lengths)
    else:
        labels  = model.predict(Xz)

    out[valid_mask] = labels
    return out


def relabel_by_vol(labels, vol_series):
    """
    Reorder regime IDs so 0 = calmest, K-1 = most volatile.
    NaN labels (or -1) are passed through unchanged.
    """
    labels = np.asarray(labels, dtype=float)
    vol    = np.asarray(vol_series, dtype=float)
    valid  = (labels >= 0) & ~np.isnan(labels) & ~np.isnan(vol)
    if not valid.any():
        return labels

    df    = pd.DataFrame({'l': labels[valid].astype(int), 'v': vol[valid]})
    means = df.groupby('l')['v'].mean().sort_values()
    mapping = dict(zip(means.index.tolist(), range(len(means))))

    out = np.full_like(labels, np.nan)        # missing rows → NaN, not -1
    for old, new in mapping.items():
        out[(labels == old) & valid] = new
    return out


def train_lgbm_per_regime(train_df, technical_vars, regime_col='Regime',
                          target_col='Target_Direction',
                          min_rows=3_000, n_iter=15, cv_splits=3, seed=42):
    """Train one calibrated LightGBM per regime. Returns dict of bundles."""
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
        # Threshold: maximise precision on training data
        p = cal.predict_proba(X)[:, 1]
        best_t, best_prec = 0.5, -np.inf
        for t in np.linspace(0.5, 0.7, 21):
            pred = (p >= t).astype(int)
            if pred.sum() == 0:
                continue
            pr = precision_score(y, pred, zero_division=0)
            if pr > best_prec:
                best_prec, best_t = pr, t
        bundles[int(r)] = {
            'model': cal, 'threshold': float(best_t),
            'features': technical_vars, 'cv_score': srch.best_score_,
        }
        print(f"      regime {int(r)}: cv_auc={srch.best_score_:.3f} "
              f"thresh={best_t:.2f}  rows={len(sub):,}")
    return bundles


# ===========================================================================
# Trade simulation (lean copy of Step 6b's logic)
# ===========================================================================

def generate_signals_and_trade(test_df, all_data_df, bundles,
                               technical_vars, holding_days=HOLDING_DAYS,
                               initial_capital=INITIAL_CAPITAL,
                               transaction_cost=TRANSACTION_COST,
                               max_per_day=MAX_SIGNALS_PER_DAY,
                               max_open_per_ticker=MAX_OPEN_PER_TICKER,
                               min_proba_override=MIN_PROBA_OVERRIDE):
    # ---- Generate signals ----
    chunks = []
    for r, b in bundles.items():
        rows = test_df[test_df['Regime'] == r].dropna(subset=b['features'])
        if rows.empty:
            continue
        proba = b['model'].predict_proba(rows[b['features']].values)[:, 1]
        thresh = max(b['threshold'], min_proba_override or b['threshold'])
        fired = proba >= thresh
        if fired.any():
            chunks.append(pd.DataFrame({
                'Date': rows.index[fired], 'Ticker': rows['Ticker'].values[fired],
                'Proba': proba[fired],
            }))
    if not chunks:
        return None, None
    sig = (pd.concat(chunks)
              .sort_values(['Date', 'Proba'], ascending=[True, False])
              .groupby('Date').head(max_per_day).reset_index(drop=True))

    # ---- Simulate ----
    px = (all_data_df.reset_index()
                       .rename(columns={all_data_df.index.name or 'index': 'Date'})
                       [['Date', 'Ticker', 'Open', 'Close']]
                       .set_index(['Ticker', 'Date']).sort_index())
    full_idx = pd.DatetimeIndex(sorted(all_data_df.index.unique()))
    sim_start = pd.Timestamp(sig['Date'].min())
    last_sig  = pd.Timestamp(sig['Date'].max())
    end_pos   = min(full_idx.get_loc(last_sig) + holding_days + 2, len(full_idx)-1)
    sim_end   = full_idx[end_pos]
    iter_idx  = full_idx[(full_idx >= sim_start) & (full_idx <= sim_end)]

    cash = initial_capital
    open_pos, closed = [], []
    equity = []
    sig_dates = set(sig['Date'].unique())

    for cur in iter_idx:
        keep = []
        for p in open_pos:
            if p['sell_date'] == cur:
                key = (p['ticker'], cur)
                if key in px.index:
                    sp = float(px.at[key, 'Close'])
                    cash += p['shares'] * sp - p['shares'] * sp * transaction_cost
                    pnl = (p['shares'] * sp) - p['cost_basis'] - p['entry_cost'] - p['shares'] * sp * transaction_cost
                    closed.append({**p, 'sell_price': sp, 'pnl': pnl,
                                   'pct': pnl / p['cost_basis'], 'win': pnl > 0})
                else:
                    keep.append(p)
            else:
                keep.append(p)
        open_pos = keep

        if cur in sig_dates:
            day = sig[sig['Date'] == cur]
            n = len(day)
            alloc = cash / n if cash > 0 else 0
            opt = {}
            for p in open_pos:
                opt[p['ticker']] = opt.get(p['ticker'], 0) + 1
            for _, s in day.iterrows():
                if opt.get(s['Ticker'], 0) >= max_open_per_ticker:
                    continue
                pos_idx = full_idx.get_loc(cur)
                if pos_idx + 1 >= len(full_idx): continue
                bd = full_idx[pos_idx + 1]
                si = pos_idx + 1 + holding_days
                if si >= len(full_idx): continue
                sd = full_idx[si]
                k = (s['Ticker'], bd)
                if k not in px.index: continue
                bp = float(px.at[k, 'Open'])
                if bp <= 0: continue
                shares = int(alloc // bp)
                if shares < 1: continue
                cb = shares * bp
                ec = cb * transaction_cost
                cash -= cb + ec
                opt[s['Ticker']] = opt.get(s['Ticker'], 0) + 1
                open_pos.append({'ticker': s['Ticker'], 'buy_date': bd,
                                 'sell_date': sd, 'shares': shares,
                                 'buy_price': bp, 'cost_basis': cb,
                                 'entry_cost': ec})

        mv = sum(p['shares'] * float(px.at[(p['ticker'], cur), 'Close'])
                 if (p['ticker'], cur) in px.index else p['cost_basis']
                 for p in open_pos)
        equity.append((cur, cash + mv))

    return pd.DataFrame(closed), pd.DataFrame(equity, columns=['Date', 'Equity']).set_index('Date')


# ===========================================================================
# Per-fold metrics
# ===========================================================================

def compute_fold_metrics(trades, equity, initial_capital):
    if trades is None or trades.empty:
        return {}
    n      = len(trades)
    wins   = trades['win'].sum()
    losses = n - wins
    pf     = (-trades.loc[trades['win'], 'pnl'].sum() /
              trades.loc[~trades['win'], 'pnl'].sum()) if losses > 0 else float('inf')
    final  = float(equity['Equity'].iloc[-1])
    yrs    = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    cagr   = (final / initial_capital) ** (1/yrs) - 1
    daily  = equity['Equity'].pct_change().dropna()
    sharpe = (daily.mean() / daily.std()) * np.sqrt(252) if daily.std() > 0 else 0
    max_dd = (equity['Equity'] / equity['Equity'].cummax() - 1).min()
    return {
        'trades':       n,
        'hit_rate':     wins / n,
        'profit_fact':  pf,
        'total_return': final / initial_capital - 1,
        'cagr':         cagr,
        'sharpe':       sharpe,
        'max_dd':       max_dd,
        'final_$':      final,
    }


# ===========================================================================
# Run all folds
# ===========================================================================

def run_walk_forward(All_data_DF, technical_variables, folds=FOLDS):
    """Main entry. All_data_DF must already have Target_Direction + features."""
    if not isinstance(All_data_DF.index, pd.DatetimeIndex):
        All_data_DF = All_data_DF.copy()
        All_data_DF.index = pd.to_datetime(All_data_DF.index)
    All_data_DF = All_data_DF.sort_index()

    all_metrics = []
    all_equity_curves = {}

    for i, (tr_start, tr_end, te_start, te_end) in enumerate(folds, 1):
        print(f"\n{'='*78}\nFOLD {i}/{len(folds)}: "
              f"train {tr_start}→{tr_end}  |  test {te_start}→{te_end}\n{'='*78}")
        t0 = time.time()

        # .copy() so we don't mutate slices of All_data_DF (silences chained-
        # assignment warnings and keeps each fold cleanly isolated)
        train = All_data_DF.loc[tr_start:tr_end].copy()
        test  = All_data_DF.loc[te_start:te_end].copy()

        # 1. Fit regimes on train
        print("  [a] Fitting regime model on training slice ...")
        rmodel, rscaler, rfeats = fit_regimes(train)
        train['Regime'] = predict_regimes(rmodel, rscaler, rfeats, train)
        test ['Regime'] = predict_regimes(rmodel, rscaler, rfeats, test)
        # Stable label ordering by ATR_Ratio (vol proxy). Missing rows
        # (NaN regime) pass through as NaN and are dropped by downstream
        # train_lgbm_per_regime.
        train['Regime'] = relabel_by_vol(train['Regime'].values,
                                         train['ATR_Ratio'].values)
        test ['Regime'] = relabel_by_vol(test ['Regime'].values,
                                         test ['ATR_Ratio'].values)

        # 2. Train per-regime LightGBM
        print("  [b] Training LightGBM per regime ...")
        bundles = train_lgbm_per_regime(train, technical_variables)

        # 3. Generate signals + simulate trades on test
        print("  [c] Backtesting on test slice ...")
        trades, equity = generate_signals_and_trade(test, All_data_DF, bundles,
                                                     technical_variables)

        # 4. Score
        m = compute_fold_metrics(trades, equity, INITIAL_CAPITAL)
        m['fold']        = i
        m['train_start'] = tr_start
        m['train_end']   = tr_end
        m['test_start']  = te_start
        m['test_end']    = te_end
        m['elapsed_s']   = round(time.time() - t0, 1)
        all_metrics.append(m)
        if equity is not None:
            all_equity_curves[i] = equity

        print(f"\n  Fold {i} result:  return={m.get('total_return', 0):+.2%}  "
              f"sharpe={m.get('sharpe', 0):.2f}  trades={m.get('trades', 0)}  "
              f"({m['elapsed_s']}s)")

    # ---- Summarise ----
    summary = pd.DataFrame(all_metrics)
    summary_path = os.path.join(OUTPUT_DIR, 'WalkForward_Folds.csv')
    summary.to_csv(summary_path, index=False)
    print(f"\n{'='*78}\nWALK-FORWARD SUMMARY\n{'='*78}")
    print(summary[['fold', 'test_start', 'test_end', 'trades', 'hit_rate',
                   'total_return', 'cagr', 'sharpe', 'max_dd']].to_string(index=False))
    print(f"\n  Saved: {summary_path}")

    # ---- Aggregated stats ----
    print("\nAggregated across folds:")
    print(f"  Mean total return:  {summary['total_return'].mean():+.2%}")
    print(f"  Mean CAGR:          {summary['cagr'].mean():+.2%}")
    print(f"  Mean Sharpe:        {summary['sharpe'].mean():.2f}")
    print(f"  Worst-fold return:  {summary['total_return'].min():+.2%}")
    print(f"  Folds positive:     {(summary['total_return'] > 0).sum()}/{len(summary)}")

    # ---- Plot ----
    if all_equity_curves:
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, eq in all_equity_curves.items():
            ax.plot(eq.index, eq['Equity'] / INITIAL_CAPITAL,
                    label=f'Fold {i}', lw=1.6)
        ax.axhline(1.0, color='k', ls=':', lw=0.8)
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity (×initial)')
        ax.set_title('Walk-forward equity curves')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        out_png = os.path.join(OUTPUT_DIR, 'WalkForward_EquityCurves.png')
        plt.savefig(out_png, dpi=140)
        plt.show()
        print(f"  Saved: {out_png}")

    return summary


# ===========================================================================
# Run it
# ===========================================================================
# Uncomment when All_data_DF and Technical_variables are in scope:
#
summary = run_walk_forward(All_data_DF, Technical_variables)
