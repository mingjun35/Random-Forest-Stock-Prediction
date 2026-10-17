#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1d — SEC Form 4 Insider Transaction Features
==================================================

Adds genuinely orthogonal alpha: insider buying/selling activity from SEC
Form 4 filings. This is information your existing technical features have
zero exposure to.

Why this matters
----------------
Your CV AUC has plateaued at ~0.51 — barely above random. That's the natural
ceiling of price/volume features on liquid US large-caps where alpha has been
arbitraged away. The ONLY way to push past that ceiling is to bring in data
the existing features don't have. SEC Form 4 (insider transactions) is the
highest-value retail-accessible source of that:

  * FREE — direct from SEC EDGAR
  * STRUCTURED — XBRL/XML format, decades of clean history
  * REAL ALPHA — multiple peer-reviewed studies show insider buying
    (especially cluster buys) predicts 3-7% outperformance over 1-3 months
  * UNDERUSED by retail — most don't query EDGAR systematically

Two-tier approach
-----------------
TIER 1 (default — fast, ~1 minute for all 500 tickers):
    Just FILING COUNTS and RECENCY. Doesn't tell you buy vs sell or amounts,
    but it does tell you "is there unusual insider activity around this
    ticker right now?" Often that alone moves CV AUC by 0.01-0.02.

TIER 2 (optional — slow, ~3-5 hours for full history):
    Parse each Form 4 XML for transaction direction, share count, dollar
    value, insider role. Gives you buy/sell direction, cluster signals
    (multiple distinct insiders), C-suite vs board flags.

Run Tier 1 first. If it adds CV AUC lift in retraining, scale to Tier 2.
If Tier 1 shows zero lift, the strategy doesn't survive into the alternative-
data regime either, and Option 4 (step away from this approach) is honest.

Author: leemingjun
"""

import os
import time
import json
import warnings
import requests
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')


# ===========================================================================
# Configuration
# ===========================================================================

CACHE_DIR     = '/Users/leemingjun/Desktop/Random Forest Model/PaperTrading/edgar_cache'
SUBMISSIONS_DIR = os.path.join(CACHE_DIR, 'submissions')
FORM4_XML_DIR   = os.path.join(CACHE_DIR, 'form4_xml')
for d in (CACHE_DIR, SUBMISSIONS_DIR, FORM4_XML_DIR):
    os.makedirs(d, exist_ok=True)

# SEC requires identifying yourself in the User-Agent. Customize this!
USER_AGENT = "MJ Lee leemingjun12@gmail.com"

# SEC rate limit is 10 req/sec. We sleep 0.11s = ~9 req/sec to be safe.
RATE_LIMIT_SLEEP = 0.11

# How far back to pull insider data. Should match your training data range.
SINCE_DATE = '2018-01-01'

# Tier 2 toggle — XML parsing takes hours for 500 tickers. Default OFF.
PARSE_FORM4_XMLS = False

# How many tickers to scrape (limit for first test runs).
# Set to None to do all S&P 500. Start with 50 to verify the pipeline works.
MAX_TICKERS = None     # set to e.g. 50 for first test, None for full S&P 500


# ===========================================================================
# 1. Get ticker → CIK mapping (SEC's own JSON, ~1 request)
# ===========================================================================

def get_ticker_cik_map(force_refresh=False):
    """SEC publishes a master JSON mapping all tickers to CIKs. Cache locally."""
    cache_file = os.path.join(CACHE_DIR, 'company_tickers.json')
    if os.path.exists(cache_file) and not force_refresh:
        age_days = (time.time() - os.path.getmtime(cache_file)) / 86400
        if age_days < 30:
            with open(cache_file) as f:
                return json.load(f)

    print(f"  Fetching ticker→CIK mapping from SEC ...")
    resp = requests.get(
        'https://www.sec.gov/files/company_tickers.json',
        headers={'User-Agent': USER_AGENT},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    # Reformat from {"0": {"cik_str": ..., "ticker": ..., "title": ...}, ...}
    mapping = {row['ticker']: int(row['cik_str']) for row in data.values()}

    with open(cache_file, 'w') as f:
        json.dump(mapping, f)
    print(f"  Cached {len(mapping):,} ticker→CIK pairs to {cache_file}")
    return mapping


# ===========================================================================
# 2. For each ticker, fetch its submissions metadata from EDGAR
#    (one JSON per ticker, contains all filings, ~500 total requests)
# ===========================================================================

def fetch_submissions(cik, ticker, force_refresh=False, max_age_days=7):
    """
    Returns the SEC's submissions JSON for one CIK, which lists ALL filings
    for that company including Form 4. Cached.
    """
    cache_file = os.path.join(SUBMISSIONS_DIR, f'{ticker}_{cik:010d}.json')
    if os.path.exists(cache_file) and not force_refresh:
        age_days = (time.time() - os.path.getmtime(cache_file)) / 86400
        if age_days < max_age_days:
            with open(cache_file) as f:
                return json.load(f)

    url = f'https://data.sec.gov/submissions/CIK{cik:010d}.json'
    try:
        resp = requests.get(url, headers={'User-Agent': USER_AGENT}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        time.sleep(RATE_LIMIT_SLEEP)
        return data
    except Exception as e:
        print(f"    [WARN] {ticker} (CIK {cik}): {e}")
        return None


def extract_form4_filings(submissions, since_date=SINCE_DATE):
    """
    Pulls just the Form 4 filing rows out of a submissions blob.
    Returns DataFrame with columns: filing_date, accession_no, primary_doc.
    """
    if submissions is None:
        return pd.DataFrame()

    recent = submissions.get('filings', {}).get('recent', {})
    if not recent:
        return pd.DataFrame()

    df = pd.DataFrame({
        'form':            recent.get('form', []),
        'filing_date':     recent.get('filingDate', []),
        'accession_no':    recent.get('accessionNumber', []),
        'primary_doc':     recent.get('primaryDocument', []),
    })

    df = df[df['form'] == '4']
    if df.empty:
        return df

    df['filing_date'] = pd.to_datetime(df['filing_date'])
    df = df[df['filing_date'] >= pd.Timestamp(since_date)]
    return df.reset_index(drop=True)


# ===========================================================================
# 3. TIER 1 — Build features from filing dates only (fast, no XML parsing)
# ===========================================================================

def build_filing_count_features(filings_per_ticker, all_dates):
    """
    For each (Date, Ticker), count Form 4 filings in the trailing 7 / 30 days
    and compute days-since-last-filing.

    Inputs:
        filings_per_ticker : dict {ticker -> DataFrame with 'filing_date' col}
        all_dates          : DatetimeIndex of dates we want features on
    Returns:
        DataFrame with columns:
            Date, Ticker, Form4_Count_7d, Form4_Count_30d, Days_Since_Form4
    """
    print(f"  Building filing-count features over {len(all_dates):,} dates × "
          f"{len(filings_per_ticker)} tickers ...")

    chunks = []
    for i, (ticker, fdf) in enumerate(filings_per_ticker.items()):
        if i % 50 == 0:
            print(f"    {i}/{len(filings_per_ticker)} ...")

        if fdf.empty:
            # No filings — features are all 0 / inf
            tmp = pd.DataFrame({'Date': all_dates})
            tmp['Ticker'] = ticker
            tmp['Form4_Count_7d']  = 0
            tmp['Form4_Count_30d'] = 0
            tmp['Days_Since_Form4'] = 9999
            chunks.append(tmp)
            continue

        fdates = pd.to_datetime(fdf['filing_date']).sort_values().values

        cnt7 = []
        cnt30 = []
        dsl  = []
        for d in all_dates:
            d_np = np.datetime64(d)
            n7  = int(((fdates >= d_np - np.timedelta64(7,  'D')) & (fdates <= d_np)).sum())
            n30 = int(((fdates >= d_np - np.timedelta64(30, 'D')) & (fdates <= d_np)).sum())
            past = fdates[fdates <= d_np]
            if len(past) == 0:
                ds = 9999
            else:
                ds = int((d_np - past.max()).astype('timedelta64[D]') / np.timedelta64(1, 'D'))
            cnt7.append(n7); cnt30.append(n30); dsl.append(ds)

        tmp = pd.DataFrame({
            'Date': all_dates,
            'Ticker': ticker,
            'Form4_Count_7d':   cnt7,
            'Form4_Count_30d':  cnt30,
            'Days_Since_Form4': dsl,
        })
        chunks.append(tmp)

    feat_df = pd.concat(chunks, ignore_index=True)
    feat_df['Date'] = pd.to_datetime(feat_df['Date'])
    return feat_df


# ===========================================================================
# 4. TIER 2 (optional) — Parse Form 4 XMLs for transaction details
# ===========================================================================

def fetch_form4_xml(cik, accession_no, primary_doc, ticker):
    """Download one Form 4 XML. Cached."""
    accno_clean = accession_no.replace('-', '')
    cache_file = os.path.join(FORM4_XML_DIR, f'{ticker}_{accno_clean}.xml')
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return f.read()

    url = (f'https://www.sec.gov/Archives/edgar/data/{cik}/'
           f'{accno_clean}/{primary_doc}')
    try:
        resp = requests.get(url, headers={'User-Agent': USER_AGENT}, timeout=30)
        resp.raise_for_status()
        with open(cache_file, 'w') as f:
            f.write(resp.text)
        time.sleep(RATE_LIMIT_SLEEP)
        return resp.text
    except Exception as e:
        return None


def parse_form4_xml(xml_text):
    """
    Extract key fields from a Form 4 ownership document.
    Returns list of dicts (one per non-derivative transaction).
    """
    if xml_text is None:
        return []

    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_text)
    except Exception:
        return []

    # Reporter info
    reporting_owner = root.find('.//reportingOwner')
    is_officer = director = ten_pct = False
    if reporting_owner is not None:
        rel = reporting_owner.find('.//reportingOwnerRelationship')
        if rel is not None:
            is_officer  = (rel.findtext('isOfficer',  default='0') == '1')
            director    = (rel.findtext('isDirector', default='0') == '1')
            ten_pct     = (rel.findtext('isTenPercentOwner', default='0') == '1')

    # Non-derivative transactions (actual share buys/sells)
    out = []
    for txn in root.findall('.//nonDerivativeTransaction'):
        date_str   = txn.findtext('.//transactionDate/value')
        code       = txn.findtext('.//transactionCoding/transactionCode')   # P=buy, S=sell, A=award
        shares_str = txn.findtext('.//transactionAmounts/transactionShares/value')
        price_str  = txn.findtext('.//transactionAmounts/transactionPricePerShare/value')
        if not (date_str and code and shares_str):
            continue
        try:
            shares = float(shares_str)
            price  = float(price_str) if price_str else np.nan
            value  = shares * price if not np.isnan(price) else np.nan
        except ValueError:
            continue
        out.append({
            'transaction_date':  pd.Timestamp(date_str),
            'code':              code,
            'shares':            shares,
            'price':             price,
            'value':             value,
            'is_officer':        is_officer,
            'is_director':       director,
            'is_10pct_owner':    ten_pct,
        })
    return out


def build_transaction_features(transactions_df, all_dates):
    """
    Roll up parsed transactions into per-(Date, Ticker) features.
    Adds richer signals: buy/sell counts, net dollar value, distinct insiders.
    """
    print(f"  Rolling {len(transactions_df):,} transactions into per-(Date, Ticker) features ...")

    # Code 'P' = open-market purchase (the bullish signal we care about)
    # Code 'S' = open-market sale
    transactions_df['is_buy']  = transactions_df['code'] == 'P'
    transactions_df['is_sell'] = transactions_df['code'] == 'S'
    transactions_df['Date']    = pd.to_datetime(transactions_df['transaction_date'])

    chunks = []
    for ticker, tdf in transactions_df.groupby('Ticker'):
        tdf = tdf.sort_values('Date')
        for d in all_dates:
            window = tdf[(tdf['Date'] >= d - pd.Timedelta(days=30)) & (tdf['Date'] <= d)]
            if window.empty:
                chunks.append({
                    'Date': d, 'Ticker': ticker,
                    'Insider_Buys_30d': 0, 'Insider_Sells_30d': 0,
                    'Insider_Net_Value_30d': 0,
                    'Insider_Cluster_Buyers_30d': 0,
                    'Officer_Buy_30d': 0,
                })
                continue
            buys  = window[window['is_buy']]
            sells = window[window['is_sell']]
            chunks.append({
                'Date': d, 'Ticker': ticker,
                'Insider_Buys_30d':         len(buys),
                'Insider_Sells_30d':        len(sells),
                'Insider_Net_Value_30d':    (buys['value'].sum() - sells['value'].sum()),
                'Insider_Cluster_Buyers_30d': buys['is_officer'].sum() + buys['is_director'].sum(),
                'Officer_Buy_30d':          int(buys['is_officer'].any()),
            })
    return pd.DataFrame(chunks)


# ===========================================================================
# 5. Main
# ===========================================================================

def main(all_data_df, max_tickers=MAX_TICKERS, parse_xmls=PARSE_FORM4_XMLS):
    print("=" * 70)
    print("Step 1d — SEC Form 4 Insider Features")
    print("=" * 70)

    # ---- 1. Ticker→CIK ------------------------------------------------
    print("\n[1] Loading ticker→CIK map ...")
    cik_map = get_ticker_cik_map()

    tickers = sorted(all_data_df['Ticker'].unique().tolist())
    if max_tickers is not None:
        tickers = tickers[:max_tickers]
        print(f"  Limiting to first {max_tickers} tickers (for testing)")

    matched = [(t, cik_map[t]) for t in tickers if t in cik_map]
    missing = [t for t in tickers if t not in cik_map]
    print(f"  Matched {len(matched)} of {len(tickers)} tickers to CIKs")
    if missing:
        print(f"  Missing CIK for: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    # ---- 2. Fetch submissions for each ticker -------------------------
    print(f"\n[2] Fetching submissions JSON for {len(matched)} tickers ...")
    print(f"    (~{len(matched) * RATE_LIMIT_SLEEP:.0f}s minimum due to SEC rate limit)")
    filings_per_ticker = {}
    t0 = time.time()
    for i, (ticker, cik) in enumerate(matched):
        if i % 50 == 0 and i > 0:
            elapsed = time.time() - t0
            print(f"    {i}/{len(matched)} ({elapsed:.0f}s elapsed)")
        sub = fetch_submissions(cik, ticker)
        f4  = extract_form4_filings(sub)
        filings_per_ticker[ticker] = f4
    print(f"  Done in {time.time()-t0:.0f}s")

    n_filings = sum(len(df) for df in filings_per_ticker.values())
    print(f"  Total Form 4 filings since {SINCE_DATE}: {n_filings:,}")

    # ---- 3. TIER 1 — filing-count features ----------------------------
    print(f"\n[3] Building Tier 1 (count) features ...")

    # Use all_data_df's existing dates (so the merge below is clean)
    if isinstance(all_data_df.index, pd.DatetimeIndex):
        all_dates = pd.DatetimeIndex(sorted(all_data_df.index.unique()))
    else:
        all_dates = pd.DatetimeIndex(sorted(all_data_df['Date'].unique()))

    feat_df = build_filing_count_features(filings_per_ticker, all_dates)
    feat_path = os.path.join(CACHE_DIR, 'insider_features_tier1.csv')
    feat_df.to_csv(feat_path, index=False)
    print(f"  Saved {len(feat_df):,} feature rows to {feat_path}")

    # ---- 4. TIER 2 — XML parsing for transaction details (optional) ---
    if parse_xmls:
        print(f"\n[4] Tier 2: Downloading + parsing Form 4 XMLs ...")
        print(f"    WARNING: this is slow. ~{n_filings * RATE_LIMIT_SLEEP / 60:.0f} minutes minimum.")

        all_txns = []
        t0 = time.time()
        n_done = 0
        for ticker, fdf in filings_per_ticker.items():
            if fdf.empty:
                continue
            cik = cik_map[ticker]
            for _, row in fdf.iterrows():
                xml = fetch_form4_xml(cik, row['accession_no'], row['primary_doc'], ticker)
                txns = parse_form4_xml(xml)
                for t in txns:
                    t['Ticker'] = ticker
                all_txns.extend(txns)
                n_done += 1
                if n_done % 500 == 0:
                    elapsed = time.time() - t0
                    rate = n_done / max(elapsed, 1)
                    eta = (n_filings - n_done) / max(rate, 0.1) / 60
                    print(f"    {n_done}/{n_filings} ({elapsed:.0f}s, eta {eta:.0f}min)")

        txn_df = pd.DataFrame(all_txns)
        txn_path = os.path.join(CACHE_DIR, 'insider_transactions.csv')
        txn_df.to_csv(txn_path, index=False)
        print(f"  Parsed {len(txn_df):,} transactions, saved to {txn_path}")

        feat2 = build_transaction_features(txn_df, all_dates)
        feat2_path = os.path.join(CACHE_DIR, 'insider_features_tier2.csv')
        feat2.to_csv(feat2_path, index=False)
        print(f"  Saved Tier 2 features to {feat2_path}")

        # Merge Tier 2 into Tier 1
        feat_df = feat_df.merge(feat2, on=['Date', 'Ticker'], how='left')

    # ---- 5. Merge into All_data_DF -----------------------------------
    print(f"\n[5] Merging into All_data_DF ...")
    df = all_data_df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={df.index.name or 'index': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.merge(feat_df, on=['Date', 'Ticker'], how='left')

    # Forward-fill insider features per ticker (gaps between filings are common)
    insider_cols = [c for c in df.columns if c.startswith('Form4_') or
                    c.startswith('Insider_') or c.startswith('Officer_') or
                    c == 'Days_Since_Form4']
    for c in insider_cols:
        df[c] = df.groupby('Ticker')[c].ffill().fillna(0)

    df = df.set_index('Date').sort_index()

    print(f"\n  Added {len(insider_cols)} insider columns:")
    for c in insider_cols:
        non_zero = (df[c] != 0).mean()
        print(f"    {c:30s}  non-zero rate: {non_zero:.1%}")

    return df, insider_cols


# ===========================================================================
# 6. Run it
# ===========================================================================
# Assumes All_data_DF is in scope from Step 1b/2b.

All_data_DF, insider_features = main(All_data_DF)


# ===========================================================================
# 7. Add to Technical_variables and retrain
# ===========================================================================

print(f"\nAdd these to Technical_variables in Step 4b/5b:")
for f in insider_features:
    print(f"    '{f}',")

print("\nThen re-run:  Step 4b → Step 5b → Step 6d → Step 7d")
print("Watch the per-regime CV AUC during Step 5b training. If it goes from")
print("~0.51 to 0.53+, the insider features are adding real signal. If it")
print("stays flat, the strategy doesn't have alpha even with this data —")
print("time for Option 4.")
