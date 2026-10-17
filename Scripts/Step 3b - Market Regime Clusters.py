#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3b — Market-Regime Clustering for the Random Forest Pipeline
=================================================================

Replaces the cross-sectional ticker clustering in the original Step 3 with a
PER-ROW MARKET-REGIME label that the Random Forest in Step 5 can use as a
feature.

Why this change?
----------------
The original Step 3 fits a GMM on each ticker's *most recent* technical
snapshot (one row per ticker), so the cluster you assign to AAPL is decided by
a single noisy day. It also never tells the model whether a particular bar of
data was taken in a calm bull market or a panic crash — which is where most of
the predictive lift hides for a buy/sell classifier.

This script instead:
  1. Builds a small "regime feature vector" per (Ticker, Date) row using the
     indicators Step 2 has already calculated.
  2. Fits the regime model on the TRAINING WINDOW ONLY (no lookahead).
  3. Applies it to the full dataset to attach a `Regime` label to every row.
  4. Saves the labelled dataset and a simple regime-transition diagnostic.

Step 5 then uses `Regime` either:
  (a) as an extra feature in a single global Random Forest, or
  (b) as a routing key to train one Random Forest per regime.

Both wiring options are sketched at the bottom of this file.

Author: leemingjun
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Optional: HMM. If hmmlearn isn't installed:  pip install hmmlearn
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("hmmlearn not installed — HMM option disabled. "
          "`pip install hmmlearn` to enable it.")


# ===========================================================================
# 1.  Inputs
# ===========================================================================
# Expected dataframes from prior steps:
#   All_data_DF   — OHLCV + Date + Ticker
#   Technicals_DF — same index, with indicator columns from Step 2
#
# If you run this script standalone, set these paths to your saved CSVs:
#
# All_data_DF   = pd.read_csv('.../All_data_DF.csv',   parse_dates=['Date'])
# Technicals_DF = pd.read_csv('.../Technicals_DF.csv', parse_dates=['Date'])

OUTPUT_DIR = '/Users/leemingjun/Desktop/Random Forest Model/2025 S&P 500 Prediction (Return Variable)'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# 2.  Build the regime feature vector
# ===========================================================================
# Keep this small (4–6 features) so clusters stay interpretable.
# We deliberately use indicators you ALREADY computed in Step 2.

REGIME_FEATURES = [
    'ATR_Ratio',         # short-term vs long-term volatility — captures vol regime
    'ADX_15',            # trend strength (0–100)
    'RSI_15',            # overbought / oversold momentum
    'SMA_ratio',         # trend direction / mean reversion pressure
    '%B',                # position within Bollinger band — vol-normalised price
    'SMA_Volume_Ratio',  # volume regime (spike vs baseline)
]


def _ensure_date_column(df):
    """
    Your pipeline uses Date as the DataFrame INDEX, not a column.
    This helper returns a copy with Date promoted back to a column so the
    merge / sort / split logic below can use it uniformly.
    """
    out = df.copy()
    if 'Date' not in out.columns:
        # Date is the index — bring it out as a column
        out = out.reset_index().rename(columns={'index': 'Date'})
        # If the index already had the name 'Date', reset_index will preserve it
        if 'Date' not in out.columns:
            # Last resort: assume the first column is the date
            out = out.rename(columns={out.columns[0]: 'Date'})
    out['Date'] = pd.to_datetime(out['Date'])
    return out


def build_regime_dataset(all_data_df, technicals_df, feature_cols=REGIME_FEATURES):
    """
    Merge OHLCV + technicals, drop rows with NaN regime features, return tidy
    long-format dataframe sorted by (Ticker, Date).

    Works whether Date is a column or the DataFrame index.
    """
    a = _ensure_date_column(all_data_df)
    t = _ensure_date_column(technicals_df)

    df = (
        a[['Date', 'Ticker', 'Close']]
        .merge(t[['Date', 'Ticker'] + feature_cols],
               on=['Date', 'Ticker'], how='inner')
        .sort_values(['Ticker', 'Date'])
        .dropna(subset=feature_cols)
        .reset_index(drop=True)
    )
    return df


# ===========================================================================
# 3.  Train / test split (TIME-BASED, no lookahead)
# ===========================================================================
def time_split(df, train_end='2022-12-31'):
    train = df[df['Date'] <= train_end].copy()
    test  = df[df['Date'] >  train_end].copy()
    return train, test


# ===========================================================================
# 4.  Choose number of regimes via BIC (GMM)
# ===========================================================================
def choose_k_gmm(X, k_range=range(2, 6), seed=42, sample_size=100_000):
    """
    Pick K by BIC on a Gaussian Mixture, using a random sample of rows.
    BIC penalises complexity — adding regimes stops paying off when BIC stops
    falling. 100k rows is plenty to estimate this reliably.
    """
    if len(X) > sample_size:
        rng = np.random.default_rng(seed)
        Xs = X[rng.choice(len(X), sample_size, replace=False)]
    else:
        Xs = X

    bics = []
    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type='full',
                              random_state=seed, n_init=1, max_iter=100).fit(Xs)
        bics.append(gmm.bic(Xs))
        print(f"   K={k}: BIC = {bics[-1]:,.0f}")
    best_k = list(k_range)[int(np.argmin(bics))]
    return best_k, dict(zip(list(k_range), bics))


# ===========================================================================
# 5.  Fit the regime model — pick ONE of three engines
# ===========================================================================
def fit_regime_model(X_train, k, method='hmm', seed=42,
                     lengths=None, sample_size=200_000):
    """
    method ∈ {'kmeans', 'gmm', 'hmm'}.

    Speed knobs (matter a lot at 1M+ rows):
      - sample_size  : for KMeans/GMM, fit on a random sample of rows.
                       The cluster shapes are essentially identical to the
                       full-data fit but ~10× faster. Set None to disable.
      - lengths      : REQUIRED for HMM on multi-ticker data. List of
                       per-ticker sequence lengths so the HMM does NOT
                       compute fake transitions across ticker boundaries.

    Returns (fitted_model, scaler).
    """
    scaler = StandardScaler().fit(X_train)
    Xz = scaler.transform(X_train).astype(np.float32)   # half the memory
    rng = np.random.default_rng(seed)

    if method == 'kmeans':
        if sample_size and len(Xz) > sample_size:
            idx = rng.choice(len(Xz), sample_size, replace=False)
            Xfit = Xz[idx]
            print(f"   KMeans fitting on {sample_size:,} sampled rows "
                  f"(of {len(Xz):,})")
        else:
            Xfit = Xz
        model = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Xfit)

    elif method == 'gmm':
        if sample_size and len(Xz) > sample_size:
            idx = rng.choice(len(Xz), sample_size, replace=False)
            Xfit = Xz[idx]
            print(f"   GMM fitting on {sample_size:,} sampled rows "
                  f"(of {len(Xz):,})")
        else:
            Xfit = Xz
        model = GaussianMixture(n_components=k, covariance_type='full',
                                random_state=seed, n_init=1,
                                max_iter=150).fit(Xfit)

    elif method == 'hmm':
        if not HMM_AVAILABLE:
            raise RuntimeError("hmmlearn not installed.")
        if lengths is None:
            print("   WARNING: HMM called without `lengths` — "
                  "transition matrix will be polluted by ticker boundaries.")
        else:
            print(f"   HMM fitting on {len(lengths)} sequences "
                  f"(total {sum(lengths):,} rows)")
        model = GaussianHMM(n_components=k, covariance_type='full',
                            n_iter=50, tol=1e-3,
                            random_state=seed).fit(Xz, lengths=lengths)
    else:
        raise ValueError(f"Unknown method: {method}")

    return model, scaler


def predict_regime(model, scaler, X, lengths=None):
    Xz = scaler.transform(X).astype(np.float32)
    if isinstance(model, KMeans):
        return model.predict(Xz)
    if isinstance(model, GaussianMixture):
        return model.predict(Xz)
    # GaussianHMM — use lengths so Viterbi runs per ticker, not across
    return model.predict(Xz, lengths=lengths)


# ===========================================================================
# 6.  Re-label regimes 0..K-1 in order of mean realized vol
#     (so "0" is always the calm regime, "K-1" is always the panic one — makes
#     the RF feature semantically stable across re-fits)
# ===========================================================================
def relabel_by_vol(labels, vol_col):
    means = pd.Series(vol_col).groupby(labels).mean().sort_values()
    mapping = {old: new for new, old in enumerate(means.index)}
    return np.array([mapping[l] for l in labels])


# ===========================================================================
# 7.  Run the pipeline
# ===========================================================================
def _per_ticker_lengths(df):
    """
    Return list of per-ticker sequence lengths in the SAME ORDER the rows
    appear in the dataframe. Required by hmmlearn so its forward/backward
    pass restarts at each ticker boundary (otherwise the model thinks AAPL's
    last day transitions into AMD's first day — meaningless).
    """
    return df.groupby('Ticker', sort=False).size().tolist()


def run_pipeline(all_data_df, technicals_df,
                 method='hmm',
                 train_end='2022-12-31',
                 k=None,
                 feature_cols=REGIME_FEATURES,
                 seed=42,
                 silhouette_sample=30_000,
                 fit_sample_size=200_000):

    import time
    t0 = time.time()

    print("[1/5] Building regime feature dataset ...")
    df = build_regime_dataset(all_data_df, technicals_df, feature_cols)
    print(f"      {len(df):,} rows, {df['Ticker'].nunique()} tickers, "
          f"{df['Date'].nunique()} dates")

    # Sort by Ticker first, then Date — REQUIRED for HMM lengths to align
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    print("[2/5] Time-based train/test split ...")
    train, test = time_split(df, train_end)
    train = train.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    X_train = train[feature_cols].values.astype(np.float32)
    X_full  = df   [feature_cols].values.astype(np.float32)

    train_lengths = _per_ticker_lengths(train) if method == 'hmm' else None
    full_lengths  = _per_ticker_lengths(df)    if method == 'hmm' else None

    # Auto-pick K with BIC on a subsample if K not specified
    if k is None:
        print("[3/5] Choosing K by BIC on a 100k-row sample ...")
        scaler_pick = StandardScaler().fit(X_train)
        k, bic_table = choose_k_gmm(scaler_pick.transform(X_train))
        print(f"      Auto-selected K = {k}  (BIC table: {bic_table})")
    else:
        print(f"[3/5] Using K = {k} (manually specified)")

    print(f"[4/5] Fitting {method.upper()} ...")
    model, scaler = fit_regime_model(
        X_train, k=k, method=method, seed=seed,
        lengths=train_lengths, sample_size=fit_sample_size,
    )

    print(f"[5/5] Predicting regimes on full dataset ({len(df):,} rows) ...")
    raw_labels = predict_regime(model, scaler, X_full, lengths=full_lengths)

    # Stable re-labelling by ATR_Ratio (proxy for realized vol)
    labels = relabel_by_vol(raw_labels, df['ATR_Ratio'].values)
    df['Regime'] = labels.astype(int)

    # ---- Diagnostics (subsample silhouette so it doesn't take an hour) ----
    if silhouette_sample and len(X_full) > silhouette_sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_full), silhouette_sample, replace=False)
        sil = silhouette_score(scaler.transform(X_full[idx]), labels[idx])
        print(f"Silhouette score ({silhouette_sample:,}-row sample): {sil:.3f}")
    else:
        sil = silhouette_score(scaler.transform(X_full), labels)
        print(f"Silhouette score (full data): {sil:.3f}")

    transitions_per_year = (
        df.sort_values(['Ticker', 'Date'])
          .groupby('Ticker')['Regime']
          .apply(lambda s: (s.diff().fillna(0) != 0).sum())
          .mean() / ((df['Date'].max() - df['Date'].min()).days / 365.25)
    )
    print(f"Avg regime transitions per ticker per year: {transitions_per_year:.1f}")

    # Save -----------------------------------------------------------------
    out_path = os.path.join(OUTPUT_DIR, f'Regime_Labels_{method}_K{k}.csv')
    df[['Date', 'Ticker', 'Regime']].to_csv(out_path, index=False)
    print(f"Saved regime labels to: {out_path}")

    # Quick visualisation: regime distribution over time on a sample ticker
    sample = df[df['Ticker'] == df['Ticker'].iloc[0]].set_index('Date')
    fig, ax = plt.subplots(figsize=(12, 4))
    cmap = plt.get_cmap('RdYlGn_r')
    colors = cmap(np.linspace(0.15, 0.85, k))
    start = 0
    s = sample['Regime'].values
    idx = sample.index
    for i in range(1, len(s)):
        if s[i] != s[start]:
            ax.axvspan(idx[start], idx[i], color=colors[s[start]], alpha=0.35, lw=0)
            start = i
    ax.axvspan(idx[start], idx[-1], color=colors[s[start]], alpha=0.35, lw=0)
    ax.plot(sample.index, sample['Close'], color='black', lw=0.9)
    ax.set_title(f'Regimes on {sample["Ticker"].iloc[0]} — {method.upper()} (K={k})')
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f'Regime_Plot_{method}_K{k}.png')
    plt.savefig(plot_path, dpi=140)
    plt.close()
    print(f"Saved regime plot to: {plot_path}")

    return df, model, scaler


# ===========================================================================
# 8.  Example call
# ===========================================================================
# Run this when All_data_DF and Technicals_DF are in scope.
# Note: this codebase uses Date as the DataFrame INDEX (not a column).
# build_regime_dataset() handles that automatically; the merge below restores
# the Date index after attaching the Regime column.

regime_df, regime_model, regime_scaler = run_pipeline(
    All_data_DF, Technicals_DF,
    method='hmm',          # 'kmeans' | 'gmm' | 'hmm'
    train_end='2022-12-31',
    k=None,                # let BIC choose, or pass an int
)

# --- Merge Regime back into All_data_DF (Date is the index in this codebase) ---
# Step 1: temporarily promote Date out of the index so we can merge on it
_all_with_date = All_data_DF.reset_index().rename(columns={'index': 'Date'})
if 'Date' not in _all_with_date.columns:
    _all_with_date = _all_with_date.rename(columns={_all_with_date.columns[0]: 'Date'})
_all_with_date['Date'] = pd.to_datetime(_all_with_date['Date'])

# Step 2: merge regime label
_all_with_date = _all_with_date.merge(
    regime_df[['Date', 'Ticker', 'Regime']],
    on=['Date', 'Ticker'], how='left'
)

# Step 3: put Date back as the index, matching the original layout
All_data_DF = _all_with_date.set_index('Date').sort_index()

print("Regime label attached. New columns:", [c for c in All_data_DF.columns if 'Regime' in c])
print(All_data_DF[['Ticker', 'Regime']].head())


# --- (Optional) Add lagged regime features so the RF can learn from transitions ---
for lag in [5, 20]:
    All_data_DF[f'Regime_lag{lag}'] = (
        All_data_DF.groupby('Ticker')['Regime'].shift(lag)
    )


# ===========================================================================
# 9.  How to wire Regime into Step 5
# ===========================================================================
# Option A — Single global RF, Regime as a feature
# -------------------------------------------------
# Technical_variables = [...your existing list..., 'Regime']
# # Optionally one-hot encode Regime if you don't want the tree treating it
# # as ordinal:
# # All_data_DF = pd.get_dummies(All_data_DF, columns=['Regime'], prefix='R')
#
#
# Option B — One RF per regime  (analogous to your current per-cluster loop)
# -------------------------------------------------------------------------
# for regime_id in sorted(All_data_DF['Regime'].unique()):
#     train_r = train_data[train_data['Regime'] == regime_id].dropna()
#     test_r  = test_data [test_data ['Regime'] == regime_id].dropna()
#     X_train = train_r[Technical_variables]
#     y_train = train_r['Target_Direction'].values.ravel()
#     RF_CV = GridSearchCV(RandomForestClassifier(), params,
#                          cv=TimeSeriesSplit(n_splits=n_splits), n_jobs=-1)
#     RF_CV.fit(X_train, y_train)
#     pickle.dump(RF_CV, open(f'.../RF_Regime_{regime_id}.pickle', 'wb'))
#
#
# Option C — Hierarchical (most powerful)
# ---------------------------------------
# Combine ticker-cluster (your existing Clusters_DF) WITH regime:
# train_subset = train_data[(train_data['Cluster'] == c) & (train_data['Regime'] == r)]
# Needs a lot of data per (cluster, regime) cell; only do this if you have
# the full S&P 500 history and at least a few thousand rows per cell.
