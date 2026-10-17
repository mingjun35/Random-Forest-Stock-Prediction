#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4b — Purged Time-Based Train / Test Split (with Embargo)
=============================================================

Drop-in replacement for the original Step 4. Same outputs (train_data,
test_data, x_train, y_train) so Step 5 needs zero changes.

What this fixes vs the original Step 4
--------------------------------------
1. TARGET LEAKAGE  (the important one)
   Your Target_Direction is built from Close.shift(-6), so the LAST 6 rows
   of training data have labels that depend on prices inside the test
   window. Without an embargo, your model is partially trained on the
   answers to the test set. We chop off `embargo_days` (= 6) trading days
   between train_end and test_start to remove that overlap. This is the
   "purge and embargo" technique from López de Prado's Advances in
   Financial Machine Learning.

2. DATE-BASED SPLIT  (instead of row-position based)
   The original uses `data.iloc[:split_index]`. With ~500 tickers per day,
   that can split a single date across train and test. We split on a date
   threshold so every row from any given date lives entirely in one set.

3. EXPLICIT NaN DROP
   Indicators have NaN at the start of each ticker (rolling-window warm-up).
   We drop them once at split time so split sizes are deterministic and
   reproducible.

4. CLASS-BALANCE DIAGNOSTIC
   Prints the up/down ratio in train vs test so you can spot regime drift
   (e.g. you trained in a bull market, you're testing in a bear market).

5. SORT BY (Ticker, Date)
   Defensive — guarantees deterministic ordering with multi-stock data.

6. WORKING VALIDATION CURVE
   The original had it commented out. Here it actually runs and saves the
   plot so you can pick sensible hyperparameter ranges before Step 5's
   GridSearchCV.

Author: leemingjun
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve, TimeSeriesSplit

sns.set()


# ===========================================================================
# 1.  Configuration  — change these to match your setup
# ===========================================================================

# Your existing technical features. Add 'Regime' here once Step 3b has run.
Technical_variables = [
    # Original 15 features
    'SMA_ratio', 'ATR_5', 'ATR_15', 'ATR_Ratio',
    'ADX_5', 'ADX_15', 'SMA_Volume_Ratio',
    'Stochastic_5', 'Stochastic_15', 'Stochastic_Ratio',
    'RSI_5', 'RSI_15', 'RSI_ratio',
    'MACD', '%B',
    # NEW from Step 2c
    # 'Sector_Relative_5d', 'Sector_Relative_20d',
    # 'VIX', 'VIX_Change_5d', 'VIX_60d_Z',
    # 'RSI_15_Rank', 'ATR_Ratio_Rank', 'Stochastic_5_Rank', 'MACD_Rank',
]

TARGET_COL    = 'Target_Direction'
TRAIN_SIZE    = 0.80         # fraction of unique trading dates in training
EMBARGO_DAYS  = 6            # MUST match the lookahead in your target.
                             # If Target uses Close.shift(-6) → embargo = 6.

OUTPUT_DIR    = '/Users/leemingjun/Desktop/Random Forest Model/2025 S&P 500 Prediction (Return Variable)'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# 2.  Helpers
# ===========================================================================

def _ensure_date_index(df):
    """Make sure df has a DatetimeIndex named 'Date'. Tolerates Date-as-column."""
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if 'Date' in out.columns:
            out = out.set_index('Date')
        else:
            out.index = pd.to_datetime(out.index)
    out.index = pd.to_datetime(out.index)
    out.index.name = 'Date'
    return out.sort_index()


def _print_class_balance(name, y):
    counts = pd.Series(y).value_counts(normalize=True).sort_index()
    pretty = '  '.join(f'{int(k)}={v:.1%}' for k, v in counts.items())
    print(f'  {name:<15} class balance:  {pretty}   (n = {len(y):,})')


# ===========================================================================
# 3.  Purged time-based split
# ===========================================================================

def split_data_purged(
    data,
    technical_variables,
    target_col=TARGET_COL,
    train_size=TRAIN_SIZE,
    embargo_days=EMBARGO_DAYS,
    verbose=True,
):
    """
    Split a multi-ticker time-series dataset into train / test using a
    DATE-based threshold with a purge gap to prevent target leakage from
    the forward-looking target column.

    Returns
    -------
    x_train, y_train          : training features & target (DataFrame, Series)
    train_data, test_data     : full train and test slices (with all columns)
    split_info                : dict of diagnostics (cutoff dates, sizes, etc.)
    """

    # ---- 1. Normalise index + sort ---------------------------------------
    df = _ensure_date_index(data)
    # Single sort by (Date, Ticker) — faster than sort_values + sort_index
    df = df.reset_index().sort_values(['Date', 'Ticker']).set_index('Date')

    # ---- 2. Drop rows with NaN in any feature or target ------------------
    needed_cols = list(technical_variables) + [target_col]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Columns missing from data: {missing}")

    n_before = len(df)
    df = df.dropna(subset=needed_cols)
    n_after = len(df)
    if verbose:
        print(f"Dropped {n_before - n_after:,} rows with NaN in features/target "
              f"({n_after:,} rows remaining)")

    # ---- 3. Build the date-based split with embargo ----------------------
    all_dates = np.array(sorted(df.index.unique()))
    n_dates   = len(all_dates)
    n_train   = int(math.floor(n_dates * train_size))

    # Purge: drop the LAST `embargo_days` training dates so their
    # forward-looking targets cannot peek into the test window.
    train_end_idx = n_train - embargo_days
    if train_end_idx <= 0:
        raise ValueError("embargo_days is too large for this train_size")

    train_end_date  = all_dates[train_end_idx - 1]
    test_start_date = all_dates[n_train]            # first test date

    train_data = df.loc[df.index <= train_end_date].copy()
    test_data  = df.loc[df.index >= test_start_date].copy()

    # ---- 4. Feature / target frames --------------------------------------
    x_train = train_data[technical_variables]
    y_train = train_data[target_col]
    x_test  = test_data [technical_variables]
    y_test  = test_data [target_col]

    # ---- 5. Diagnostics ---------------------------------------------------
    split_info = {
        'train_start':      str(train_data.index.min().date()),
        'train_end':        str(train_data.index.max().date()),
        'embargo_start':    str((train_data.index.max() + pd.Timedelta(days=1)).date()),
        'embargo_end':      str((test_data.index.min()  - pd.Timedelta(days=1)).date()),
        'test_start':       str(test_data.index.min().date()),
        'test_end':         str(test_data.index.max().date()),
        'train_rows':       len(train_data),
        'test_rows':        len(test_data),
        'train_dates':      train_data.index.nunique(),
        'test_dates':       test_data.index.nunique(),
        'tickers':          df['Ticker'].nunique() if 'Ticker' in df.columns else None,
        'embargo_days':     embargo_days,
    }

    if verbose:
        print("\n=== Purged Time-Based Split ===")
        print(f"  Train: {split_info['train_start']}  →  {split_info['train_end']}"
              f"   ({split_info['train_rows']:,} rows / "
              f"{split_info['train_dates']:,} dates)")
        print(f"  EMBARGO ({embargo_days} trading days):  "
              f"{split_info['embargo_start']}  →  {split_info['embargo_end']}")
        print(f"  Test:  {split_info['test_start']}  →  {split_info['test_end']}"
              f"   ({split_info['test_rows']:,} rows / "
              f"{split_info['test_dates']:,} dates)")
        if split_info['tickers']:
            print(f"  Tickers: {split_info['tickers']:,}")
        print()
        _print_class_balance('Train', y_train)
        _print_class_balance('Test',  y_test)

        # Warn if class balance shifts significantly
        train_pos = (y_train == 1).mean()
        test_pos  = (y_test  == 1).mean()
        if abs(train_pos - test_pos) > 0.10:
            print(f"  WARNING: class balance shifts by "
                  f"{abs(train_pos - test_pos):.1%} between train and test. "
                  f"Likely a regime change between the two periods.")
        print()

    return x_train, y_train, train_data, test_data, split_info


# ===========================================================================
# 4.  Validation curve  — run this BEFORE Step 5's GridSearchCV
# ===========================================================================

def plot_validation_curve(
    x_train, y_train,
    param_name='n_estimators',
    param_range=(50, 150, 300),
    n_splits=2,
    output_path=None,
    sample_size=150_000,
    rf_kwargs=None,
    seed=42,
):
    """
    Sweeps a single hyperparameter and plots train vs CV score.

    Speed knobs (matter a lot at 1M+ rows):
      sample_size : random sample of training rows used for the sweep.
                    150k is plenty to estimate the train/CV gap shape.
                    Set to None to use the full training set.
      param_range : keep to ~3 values when scanning. The full hyperparam
                    grid happens in Step 5; this is only for picking a
                    sensible range.
      n_splits    : 2 CV folds is enough for a directional read.
      rf_kwargs   : dict of extra RandomForest kwargs to keep the trees
                    cheap during the sweep, e.g. {'max_depth': 15}.

    Use it to pick a sensible range BEFORE doing GridSearchCV in Step 5.
    """
    import time
    t0 = time.time()

    # ---- Subsample to keep this a quick diagnostic, not a long job ------
    if sample_size and len(x_train) > sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x_train), sample_size, replace=False)
        x_sub = x_train.iloc[idx] if hasattr(x_train, 'iloc') else x_train[idx]
        y_sub = y_train.iloc[idx] if hasattr(y_train, 'iloc') else y_train[idx]
        print(f"Validation curve for '{param_name}' over {list(param_range)} "
              f"on {sample_size:,} sampled rows (of {len(x_train):,}) ...")
    else:
        x_sub, y_sub = x_train, y_train
        print(f"Validation curve for '{param_name}' over {list(param_range)} "
              f"on full {len(x_train):,} rows ...")

    rf_defaults = dict(
        random_state=seed, n_jobs=-1,
        max_depth=15,           # cap tree depth so each fit is bounded
        min_samples_leaf=50,    # mirrors Step 5's tighter end of the search
    )
    rf_defaults.update(rf_kwargs or {})
    rf = RandomForestClassifier(**rf_defaults)

    train_scores, test_scores = validation_curve(
        rf, X=x_sub, y=y_sub,
        param_name=param_name,
        param_range=list(param_range),
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring='accuracy',
        n_jobs=-1,
    )

    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    test_mean,  test_std  = test_scores.mean(axis=1),  test_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(param_range, train_mean, marker='o', label='Train', color='#1f77b4')
    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color='#1f77b4')
    ax.plot(param_range, test_mean, marker='s', label='Cross-Val', color='#d62728')
    ax.fill_between(param_range, test_mean - test_std, test_mean + test_std,
                    alpha=0.15, color='#d62728')
    ax.set_xlabel(param_name)
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Validation Curve — {param_name}'
                 f'  (sampled n={sample_size:,})' if sample_size else
                 f'Validation Curve — {param_name}')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, f'ValidationCurve_{param_name}.png')
    plt.savefig(output_path, dpi=140)
    plt.close()

    elapsed = time.time() - t0
    print(f"  Saved: {output_path}   ({elapsed:.1f}s elapsed)\n")

    return pd.DataFrame({
        'param_value': list(param_range),
        'train_acc':   train_mean,
        'cv_acc':      test_mean,
        'gap':         train_mean - test_mean,   # large gap = overfitting
    })


# ===========================================================================
# 5.  Run it
# ===========================================================================
# Assumes All_data_DF (with Date index, Ticker column, Target_Direction column,
# and all technical features) is already in scope from Step 2 (and Step 3b if
# you've added Regime to Technical_variables above).

x_train, y_train, train_data, test_data, split_info = split_data_purged(
    All_data_DF,
    technical_variables=Technical_variables,
    target_col=TARGET_COL,
    train_size=TRAIN_SIZE,
    embargo_days=EMBARGO_DAYS,
)

# Optional: run the validation curve to see where overfitting starts.
# Defaults are tuned to be quick (~30s on 1M rows): 3 param values, 2 CV
# folds, 150k-row sample. Set sample_size=None for the full sweep.
val_curve_df = plot_validation_curve(
    x_train, y_train,
    param_name='n_estimators',
    param_range=(50, 150, 300),
    n_splits=2,
    sample_size=150_000,
)
print(val_curve_df.to_string(index=False))

# Repeat for the other two hyperparameters that matter most. Cheap to run.
plot_validation_curve(x_train, y_train, 'max_depth',        (5, 10, 20))
plot_validation_curve(x_train, y_train, 'min_samples_leaf', (25, 50, 100))
