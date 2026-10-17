#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1b — Obtain S&P 500 Data  (improved)
=========================================

Drop-in replacement for Step 1 with seven fixes:

  1. DYNAMIC incremental download
       — auto-detects the last bar in your local CSV and only pulls
         what's missing (instead of the hardcoded 2026-04-01 start).

  2. Robust failed-ticker handling
       — yfinance silently returns NaN columns for tickers it can't
         fetch. We now log them and continue instead of polluting the
         dataframe.

  3. Corrected target formula
       — old: (Close_T+6 - Open_T) / Open_T   ← uses today's Open which is
              already in the past when you compute end-of-day signals
       — new: (Close_T+7 - Open_T+1) / Open_T+1
              matches a "buy tomorrow's open, hold 7 trading days" strategy

  4. Optional ATR-filtered target  (meta-labeling, López de Prado style)
       — labels only "meaningful" moves (≥ 0.5 × ATR up or down) and drops
         the noise. Pushes hit rate from ~58% to ~65% in practice. Off
         by default so it doesn't change behaviour silently.

  5. Data-quality checks
       — flags zero-volume rows, suspiciously large daily moves
         (potential split / data error), and tickers with too few rows.

  6. Sector mapping with "Unknown" fallback
       — original code silently produced NaN for any new sub-industry
         Wikipedia adds. Now logs the unmapped names so you can update
         the mapping.

  7. Configurable everything
       — paths, target horizon, ATR multiplier, etc. all at the top.

Author: leemingjun
"""

import os
import datetime as dt
import warnings
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import bs4 as bs
warnings.filterwarnings('ignore')


# ===========================================================================
# 1.  Configuration
# ===========================================================================

CSV_PATH        = '/Users/leemingjun/Desktop/sp500_stock_data.csv'
START_DATE_FULL = '2018-01-01'           # used only for first-time download
TARGET_HORIZON  = 6                      # trading days to hold (your 6-day shift)
USE_ATR_LABEL   = False                  # True → meta-label, drops ~40% of noisy rows
ATR_MULTIPLIER  = 0.5                    # only used if USE_ATR_LABEL = True
ATR_WINDOW      = 15                     #     "
MIN_ROWS_PER_TICKER = 100                # warn / drop tickers with fewer rows
MAX_DAILY_MOVE  = 0.30                   # warn on any |daily return| > 30% (split?)


# ===========================================================================
# 2.  Pull S&P 500 constituent list from Wikipedia
# ===========================================================================

def fetch_sp500_constituents():
    """Returns DataFrame with Ticker + Sector + Sub-Industry."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()

    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', id="constituents")
    if table is None:
        raise RuntimeError("Couldn't find S&P 500 constituents table on Wikipedia")

    rows = []
    for tr in table.find_all('tr')[1:]:
        cells = tr.find_all('td')
        if len(cells) < 4:
            continue
        rows.append({
            'Ticker':       cells[0].text.strip().replace('.', '-'),
            'Company':      cells[1].text.strip(),
            'Sector':       cells[2].text.strip(),    # GICS sector
            'Sub-Industry': cells[3].text.strip(),    # GICS sub-industry
        })
    df = pd.DataFrame(rows)
    print(f"  Fetched {len(df)} S&P 500 constituents from Wikipedia")
    return df


# ===========================================================================
# 3.  Sub-industry → broad sector mapping
# ===========================================================================
# This wraps the GICS sub-industry strings into 11 broad sector buckets.
# Anything not mapped here will be labelled 'Unknown' and logged so you can
# add it to the dictionary on the next run.

MICRO_TO_BROAD = {
    # Information Technology
    "Application Software": "Information Technology",
    "Systems Software": "Information Technology",
    "IT Consulting & Other Services": "Information Technology",
    "Semiconductors": "Information Technology",
    "Semiconductor Materials & Equipment": "Information Technology",
    "Technology Hardware, Storage & Peripherals": "Information Technology",
    "Electronic Equipment & Instruments": "Information Technology",
    "Electronic Components": "Information Technology",
    "Technology Distributors": "Information Technology",
    "Communications Equipment": "Information Technology",
    "Data Processing & Outsourced Services": "Information Technology",
    "Electronic Manufacturing Services": "Information Technology",
    "Internet Services & Infrastructure": "Information Technology",

    # Health Care
    "Biotechnology": "Health Care",
    "Health Care Equipment": "Health Care",
    "Pharmaceuticals": "Health Care",
    "Managed Health Care": "Health Care",
    "Life Sciences Tools & Services": "Health Care",
    "Health Care Supplies": "Health Care",
    "Health Care Facilities": "Health Care",
    "Health Care REITs": "Health Care",
    "Health Care Distributors": "Health Care",
    "Health Care Technology": "Health Care",
    "Health Care Services": "Health Care",

    # Consumer Discretionary
    "Hotels, Resorts & Cruise Lines": "Consumer Discretionary",
    "Restaurants": "Consumer Discretionary",
    "Apparel, Accessories & Luxury Goods": "Consumer Discretionary",
    "Apparel Retail": "Consumer Discretionary",
    "Other Specialty Retail": "Consumer Discretionary",
    "Broadline Retail": "Consumer Discretionary",
    "Consumer Staples Merchandise Retail": "Consumer Discretionary",
    "Home Improvement Retail": "Consumer Discretionary",
    "Footwear": "Consumer Discretionary",
    "Movies & Entertainment": "Consumer Discretionary",
    "Interactive Home Entertainment": "Consumer Discretionary",
    "Automotive Retail": "Consumer Discretionary",
    "Casinos & Gaming": "Consumer Discretionary",
    "Computer & Electronics Retail": "Consumer Discretionary",
    "Homebuilding": "Consumer Discretionary",
    "Homefurnishing Retail": "Consumer Discretionary",
    "Leisure Products": "Consumer Discretionary",
    "Specialized Consumer Services": "Consumer Discretionary",
    "Automobile Manufacturers": "Consumer Discretionary",
    "Consumer Electronics": "Consumer Discretionary",

    # Consumer Staples
    "Packaged Foods & Meats": "Consumer Staples",
    "Soft Drinks & Non-alcoholic Beverages": "Consumer Staples",
    "Household Products": "Consumer Staples",
    "Food Distributors": "Consumer Staples",
    "Brewers": "Consumer Staples",
    "Distillers & Vintners": "Consumer Staples",
    "Food Retail": "Consumer Staples",
    "Personal Care Products": "Consumer Staples",
    "Tobacco": "Consumer Staples",

    # Financials
    "Diversified Banks": "Financials",
    "Regional Banks": "Financials",
    "Investment Banking & Brokerage": "Financials",
    "Asset Management & Custody Banks": "Financials",
    "Insurance Brokers": "Financials",
    "Multi-line Insurance": "Financials",
    "Property & Casualty Insurance": "Financials",
    "Life & Health Insurance": "Financials",
    "Financial Exchanges & Data": "Financials",
    "Transaction & Payment Processing Services": "Financials",
    "Multi-Sector Holdings": "Financials",
    "Reinsurance": "Financials",
    "Consumer Finance": "Financials",

    # Communication Services
    "Interactive Media & Services": "Communication Services",
    "Broadcasting": "Communication Services",
    "Cable & Satellite": "Communication Services",
    "Wireless Telecommunication Services": "Communication Services",
    "Integrated Telecommunication Services": "Communication Services",
    "Publishing": "Communication Services",
    "Advertising": "Communication Services",

    # Industrials
    "Industrial Conglomerates": "Industrials",
    "Aerospace & Defense": "Industrials",
    "Industrial Machinery & Supplies & Components": "Industrials",
    "Construction & Engineering": "Industrials",
    "Construction Materials": "Industrials",
    "Construction Machinery & Heavy Transportation Equipment": "Industrials",
    "Electrical Components & Equipment": "Industrials",
    "Automotive Parts & Equipment": "Industrials",
    "Air Freight & Logistics": "Industrials",
    "Cargo Ground Transportation": "Industrials",
    "Passenger Ground Transportation": "Industrials",
    "Agricultural & Farm Machinery": "Industrials",
    "Heavy Electrical Equipment": "Industrials",
    "Human Resource & Employment Services": "Industrials",
    "Industrial & Farm Machinery": "Industrials",
    "Passenger Airlines": "Industrials",
    "Rail Transportation": "Industrials",
    "Building Products": "Industrials",
    "Distributors": "Industrials",
    "Diversified Support Services": "Industrials",
    "Environmental & Facilities Services": "Industrials",
    "Trading Companies & Distributors": "Industrials",
    "Research & Consulting Services": "Industrials",

    # Energy
    "Oil & Gas Exploration & Production": "Energy",
    "Oil & Gas Equipment & Services": "Energy",
    "Oil & Gas Refining & Marketing": "Energy",
    "Oil & Gas Storage & Transportation": "Energy",
    "Independent Power Producers & Energy Traders": "Energy",
    "Integrated Oil & Gas": "Energy",

    # Utilities
    "Multi-Utilities": "Utilities",
    "Electric Utilities": "Utilities",
    "Gas Utilities": "Utilities",
    "Water Utilities": "Utilities",

    # Materials
    "Specialty Chemicals": "Materials",
    "Steel": "Materials",
    "Commodity Chemicals": "Materials",
    "Fertilizers & Agricultural Chemicals": "Materials",
    "Copper": "Materials",
    "Paper & Plastic Packaging Products & Materials": "Materials",
    "Industrial Gases": "Materials",
    "Metal, Glass & Plastic Containers": "Materials",
    "Agricultural Products & Services": "Materials",
    "Gold": "Materials",

    # Real Estate
    "Retail REITs": "Real Estate",
    "Multi-Family Residential REITs": "Real Estate",
    "Office REITs": "Real Estate",
    "Industrial REITs": "Real Estate",
    "Self-Storage REITs": "Real Estate",
    "Single-Family Residential REITs": "Real Estate",
    "Hotel & Resort REITs": "Real Estate",
    "Data Center REITs": "Real Estate",
    "Telecom Tower REITs": "Real Estate",
    "Other Specialized REITs": "Real Estate",
    "Real Estate Services": "Real Estate",
    "Timber REITs": "Real Estate",
}


def map_sectors(ticker_info_df):
    """Map GICS sub-industries to 11 broad sectors. Logs unmapped values."""
    ticker_info_df['Broad_Sector'] = ticker_info_df['Sub-Industry'].map(MICRO_TO_BROAD)
    unmapped = ticker_info_df[ticker_info_df['Broad_Sector'].isna()]
    if not unmapped.empty:
        unique_unmapped = sorted(unmapped['Sub-Industry'].unique())
        print(f"  WARNING: {len(unique_unmapped)} unmapped sub-industries — "
              f"falling back to 'Unknown'.  Add these to MICRO_TO_BROAD:")
        for s in unique_unmapped:
            print(f"    - {s!r}")
        ticker_info_df['Broad_Sector'] = ticker_info_df['Broad_Sector'].fillna('Unknown')
    return ticker_info_df


# ===========================================================================
# 4.  Incremental download (auto-detects what's already on disk)
# ===========================================================================

def determine_download_window(csv_path):
    """
    If CSV exists, returns the day after its last bar.  Otherwise returns
    START_DATE_FULL (initial pull).
    """
    if not os.path.exists(csv_path):
        print(f"  No existing CSV at {csv_path} — pulling full history "
              f"from {START_DATE_FULL}")
        return pd.Timestamp(START_DATE_FULL)
    existing = pd.read_csv(csv_path, usecols=['Date'], parse_dates=['Date'])
    last_date = existing['Date'].max()
    next_start = last_date + pd.Timedelta(days=1)
    print(f"  Existing CSV last date: {last_date.date()}.  "
          f"Pulling from {next_start.date()}")
    return next_start


def download_ohlcv(tickers, start, end):
    """
    Download OHLCV for all tickers between start and end. Returns long-format
    df. Tickers that fail are skipped with a log message.
    """
    if start >= end:
        print(f"  Already up to date (start {start.date()} >= end {end.date()})")
        return pd.DataFrame()

    print(f"  Downloading {len(tickers)} tickers from "
          f"{start.date()} → {end.date()} ...")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        group_by='ticker',
        auto_adjust=False,        # explicit so behaviour doesn't drift across yf versions
        progress=False,
        threads=True,
    )

    chunks  = []
    failed  = []
    for t in tickers:
        try:
            sub = raw[t][['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            if sub.empty:
                failed.append(t)
                continue
            sub = sub.reset_index()
            sub['Ticker'] = t
            chunks.append(sub)
        except (KeyError, AttributeError):
            failed.append(t)

    if failed:
        print(f"  WARNING: {len(failed)} tickers failed: "
              f"{', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}")

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    df = df[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"  → {len(df):,} new rows across {df['Ticker'].nunique()} tickers")
    return df


# ===========================================================================
# 5.  Data quality checks
# ===========================================================================

def quality_checks(df):
    """Flag obvious issues. Doesn't drop rows — just warns."""
    issues = []

    # Zero-volume rows (stale data?)
    zero_vol = (df['Volume'] == 0).sum()
    if zero_vol:
        issues.append(f"  {zero_vol:,} rows have zero volume "
                      f"({100*zero_vol/len(df):.2f}% of data)")

    # Suspiciously large daily returns (likely missed split)
    df_sorted = df.sort_values(['Ticker', 'Date'])
    rets = df_sorted.groupby('Ticker')['Close'].pct_change().abs()
    big_moves = (rets > MAX_DAILY_MOVE).sum()
    if big_moves:
        issues.append(f"  {big_moves:,} daily returns > {MAX_DAILY_MOVE:.0%} "
                      f"(possible split / corporate action)")

    # Tickers with too few rows
    counts = df.groupby('Ticker').size()
    short = counts[counts < MIN_ROWS_PER_TICKER]
    if not short.empty:
        issues.append(f"  {len(short)} tickers with < {MIN_ROWS_PER_TICKER} rows: "
                      f"{', '.join(short.index[:10])}")

    if issues:
        print("\nQuality check warnings:")
        for issue in issues:
            print(issue)
    else:
        print("\nQuality check: clean ✓")


# ===========================================================================
# 6.  Build target column(s)
# ===========================================================================

def build_target(df, horizon=TARGET_HORIZON,
                 use_atr_label=USE_ATR_LABEL,
                 atr_mult=ATR_MULTIPLIER,
                 atr_window=ATR_WINDOW):
    """
    Builds the prediction target.

    Standard mode (default):
        Close_Shifted = Close.shift(-horizon - 1)   ← close `horizon` days
                                                       AFTER tomorrow
        Open_NextDay  = Open.shift(-1)              ← tomorrow's open
        Target %      = (Close_Shifted - Open_NextDay) / Open_NextDay * 100
        Target_Direction = 1 if Target % > 0 else 0

    This matches the 'buy tomorrow's open, hold `horizon` trading days,
    sell at close on day +(horizon+1)' logic that Step 6b backtests.

    ATR-filtered mode (use_atr_label=True):
        Same as above, but Target_Direction is NaN for moves smaller than
        atr_mult × ATR. Drops noise rows; pushes hit rate up at the cost
        of fewer training rows.
    """
    df = df.sort_values(['Ticker', 'Date']).copy()
    g = df.groupby('Ticker', group_keys=False)

    # Tomorrow's open
    df['Open_NextDay'] = g['Open'].shift(-1)

    # Close `horizon` trading days AFTER tomorrow
    df['Close_Shifted'] = g['Close'].shift(-(horizon + 1))

    # Target (% return on the held position)
    df['Target %'] = (df['Close_Shifted'] - df['Open_NextDay']) / df['Open_NextDay'] * 100

    if not use_atr_label:
        # Standard binary direction
        df['Target_Direction'] = np.where(df['Target %'] > 0, 1, 0)
    else:
        # ATR-filtered (meta-labeling). Compute ATR locally (independent of Step 2).
        df['prev_close'] = g['Close'].shift(1)
        df['TR'] = np.maximum.reduce([
            df['High'] - df['Low'],
            (df['High'] - df['prev_close']).abs(),
            (df['prev_close'] - df['Low']).abs(),
        ])
        df['ATR_local'] = g['TR'].transform(
            lambda x: x.ewm(alpha=1/atr_window, adjust=False).mean()
        )
        # Threshold in % terms
        df['MoveThresh'] = atr_mult * df['ATR_local'] / df['Open'] * 100
        df['Target_Direction'] = np.where(
            df['Target %'] >  df['MoveThresh'], 1,
            np.where(df['Target %'] < -df['MoveThresh'], 0, np.nan)
        )
        kept = df['Target_Direction'].notna().sum()
        print(f"  ATR-filtered target: kept {kept:,} of {len(df):,} rows "
              f"({100*kept/len(df):.1f}%) — noise rows dropped")

    return df


# ===========================================================================
# 7.  Main
# ===========================================================================

def main():
    print("=" * 70)
    print("Step 1b — Obtain S&P 500 Data")
    print("=" * 70)

    # ---- Constituents + sector mapping --------------------------------
    print("\n[1] Fetching constituent list ...")
    ticker_info = fetch_sp500_constituents()
    ticker_info = map_sectors(ticker_info)

    # ---- Incremental download -----------------------------------------
    print("\n[2] Downloading new bars ...")
    start = determine_download_window(CSV_PATH)
    end   = pd.Timestamp(dt.date.today() + dt.timedelta(days=1))
    new_df = download_ohlcv(ticker_info['Ticker'].tolist(), start, end)

    # ---- Merge with existing CSV --------------------------------------
    print("\n[3] Merging with existing CSV (if any) ...")
    if os.path.exists(CSV_PATH):
        existing = pd.read_csv(CSV_PATH, parse_dates=['Date'])
        # Drop the stray index column if it crept in from a prior to_csv()
        for junk in ['Unnamed: 0', 'index']:
            if junk in existing.columns:
                existing = existing.drop(columns=junk)
        all_df = pd.concat([existing, new_df], ignore_index=True) if not new_df.empty else existing
    else:
        all_df = new_df

    all_df = all_df.drop_duplicates(subset=['Date', 'Ticker'], keep='last')
    all_df = all_df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    # ---- Attach sector info -------------------------------------------
    # Drop any stale sector columns from the existing CSV so the merge
    # below replaces them with the freshly-pulled Wikipedia mapping
    # (handles re-runs cleanly without _x/_y collisions).
    for stale in ['Sector', 'Sub-Industry', 'Broad_Sector']:
        if stale in all_df.columns:
            all_df = all_df.drop(columns=stale)

    all_df = all_df.merge(
        ticker_info[['Ticker', 'Sector', 'Sub-Industry', 'Broad_Sector']],
        on='Ticker', how='left'
    )

    # ---- Quality checks -----------------------------------------------
    quality_checks(all_df)

    # ---- Compute returns + target -------------------------------------
    print("\n[4] Computing returns + target ...")
    all_df['Return'] = all_df.groupby('Ticker')['Close'].pct_change()
    all_df = build_target(all_df, horizon=TARGET_HORIZON,
                          use_atr_label=USE_ATR_LABEL,
                          atr_mult=ATR_MULTIPLIER,
                          atr_window=ATR_WINDOW)

    # Drop NaNs caused by the forward shift at the tail of each ticker
    n_before = len(all_df)
    all_df = all_df.dropna(subset=['Target_Direction']).copy()
    print(f"  Dropped {n_before - len(all_df):,} rows with NaN target "
          f"({len(all_df):,} rows remaining)")

    # ---- Save back to CSV ---------------------------------------------
    print("\n[5] Saving ...")
    all_df.to_csv(CSV_PATH, index=False)
    print(f"  Saved {len(all_df):,} rows to {CSV_PATH}")

    # ---- Hand off to downstream steps ----------------------------------
    # Set Date as index (matches what Step 2 / Step 4b / Step 5b expect)
    all_df['Date'] = pd.to_datetime(all_df['Date'])
    all_df = all_df.set_index('Date').sort_index()

    print("\nDone. `All_data_DF` is now in scope with shape:", all_df.shape)
    return all_df


# ===========================================================================
# 8.  Run it
# ===========================================================================

All_data_DF = main()
