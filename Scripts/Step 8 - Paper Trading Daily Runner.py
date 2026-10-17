#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 8 — Paper Trading Daily Runner
====================================

Run this ONCE per day, after US market close (≈ 4:15pm ET). It does:

  1.  Download today's OHLCV for all S&P 500 tickers (yfinance)
  2.  Recompute Step 2 technical indicators (incremental — only the latest day)
  3.  Apply the saved Step 3b regime model to label today's rows
  4.  Apply the saved Step 5b LightGBM models per regime
  5.  Filter signals (regime ∈ {1,2}, proba ≥ MIN_PROBA, ticker cap, daily cap)
  6.  Open NEW paper-trade positions via Alpaca   (BUY market, fills at next open)
  7.  Close any positions whose 7-day holding period ends tomorrow
  8.  Log everything to daily CSVs for review

You can run this script manually OR schedule it (cron / Task Scheduler — see
the SCHEDULING GUIDE at the bottom of this file).

================================================================================
ONE-TIME SETUP (do this BEFORE the first run)
================================================================================

  1.  pip install alpaca-py yfinance lightgbm hmmlearn joblib pandas numpy

  2.  Sign up for an Alpaca paper trading account (free, ~5 min):
        https://alpaca.markets/  →  Sign Up  →  Paper Trading API
      Copy your two keys: API_KEY and SECRET_KEY.

  3.  Set them as environment variables (don't hard-code in this file):
        macOS / Linux  →  add to ~/.zshrc or ~/.bashrc:
            export ALPACA_API_KEY="YOUR_KEY_HERE"
            export ALPACA_SECRET_KEY="YOUR_SECRET_HERE"
        Windows (PowerShell, run as admin once):
            [System.Environment]::SetEnvironmentVariable('ALPACA_API_KEY','YOUR_KEY','User')
            [System.Environment]::SetEnvironmentVariable('ALPACA_SECRET_KEY','YOUR_SECRET','User')

  4.  Make sure your saved model files exist:
        lightgbm_Regime_1.joblib
        lightgbm_Regime_2.joblib
      ... in MODEL_DIR (set in Configuration below).

  5.  First run is in DRY_RUN mode by default — it WILL log what it WOULD have
      done without placing real (paper) orders. Set DRY_RUN = False ONLY after
      you've verified one day looks correct.

================================================================================
"""

import os
import sys
import time
import json
import joblib
import warnings
import datetime as dt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# ===========================================================================
# Configuration — change these to match your setup
# ===========================================================================

MODEL_DIR        = '/Users/leemingjun/Desktop/Random Forest Model/2025 S&P 500 Prediction (Return Variable)'
DATA_DIR         = '/Users/leemingjun/Desktop/Random Forest Model/PaperTrading'
LOG_DIR          = os.path.join(DATA_DIR, 'logs')
POSITIONS_FILE   = os.path.join(DATA_DIR, 'open_positions.csv')
TRADES_FILE      = os.path.join(DATA_DIR, 'all_trades.csv')
TICKERS_FILE     = os.path.join(DATA_DIR, 'sp500_tickers.csv')

# Trading parameters — should match your Step 6b backtest config exactly
MIN_PROBA        = 0.80          # match MIN_PROBA_OVERRIDE from backtest
HOLDING_DAYS     = 7
MAX_NEW_PER_DAY  = 10            # cap new positions opened today
MAX_OPEN_PER_TICKER = 1
INITIAL_CAPITAL  = 5_000.0       # only used to size first trades
EXCLUDE_REGIMES  = [0, 3, 4]
GROUP_COL        = 'Regime'
MODEL_TYPE       = 'lightgbm'

# Safety knob: True = log only, no orders.  Flip to False after first dry runs.
DRY_RUN          = True

# Regime feature columns — must match what your Step 3b model was trained on
REGIME_FEATURES = [
    'ATR_Ratio', 'ADX_15', 'RSI_15', 'SMA_ratio', '%B', 'SMA_Volume_Ratio',
]

# Make output dirs
for d in [DATA_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)


# ===========================================================================
# Logging helper
# ===========================================================================

def log(msg, level='INFO'):
    stamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line  = f'[{stamp}] [{level}] {msg}'
    print(line)
    log_file = os.path.join(LOG_DIR, f'runner_{dt.date.today().isoformat()}.log')
    with open(log_file, 'a') as f:
        f.write(line + '\n')


# ===========================================================================
# 1.  Download today's bars for all S&P 500 tickers
# ===========================================================================

def get_sp500_tickers():
    """Cache the ticker list. Refresh weekly."""
    if os.path.exists(TICKERS_FILE):
        age_days = (time.time() - os.path.getmtime(TICKERS_FILE)) / 86400
        if age_days < 7:
            return pd.read_csv(TICKERS_FILE)['Ticker'].tolist()

    import requests, bs4
    log('Refreshing S&P 500 ticker list from Wikipedia ...')
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs4.BeautifulSoup(resp.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    tickers = [row.find_all('td')[0].text.strip().replace('.', '-')
               for row in table.find_all('tr')[1:]]
    pd.DataFrame({'Ticker': tickers}).to_csv(TICKERS_FILE, index=False)
    return tickers


def download_recent_bars(tickers, lookback_days=120):
    """
    Download the last `lookback_days` calendar days of OHLCV.
    We need ~3-4 months because indicators use 50-day windows.
    """
    import yfinance as yf
    end   = dt.date.today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=lookback_days)
    log(f'Downloading {lookback_days}d of bars for {len(tickers)} tickers ...')
    raw = yf.download(tickers, start=start, end=end, progress=False,
                      auto_adjust=False, group_by='ticker', threads=True)

    # Reshape to long format: Date, Ticker, Open, High, Low, Close, Volume
    chunks = []
    for t in tickers:
        try:
            sub = raw[t][['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            sub = sub.reset_index().rename(columns={'Date': 'Date'})
            sub['Ticker'] = t
            chunks.append(sub)
        except (KeyError, AttributeError):
            continue
    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    log(f'  → {len(df):,} rows across {df["Ticker"].nunique()} tickers')
    return df


# ===========================================================================
# 2.  Compute indicators (just the columns the regime + RF need)
# ===========================================================================

def compute_indicators(df, short=5, long=15):
    """Compute the same indicators Step 2 produces. Pure pandas — fast."""
    out = df.sort_values(['Ticker', 'Date']).copy()
    g   = out.groupby('Ticker', group_keys=False)

    # SMA + ratio
    out['SMA_5']     = g['Close'].transform(lambda x: x.rolling(short).mean())
    out['SMA_15']    = g['Close'].transform(lambda x: x.rolling(long ).mean())
    out['SMA_ratio'] = out['SMA_15'] / out['SMA_5']

    # SMA volume + ratio
    out['SMA5_Vol']  = g['Volume'].transform(lambda x: x.rolling(short).mean())
    out['SMA15_Vol'] = g['Volume'].transform(lambda x: x.rolling(long ).mean())
    out['SMA_Volume_Ratio'] = out['SMA5_Vol'] / out['SMA15_Vol']

    # True Range + ATR (Wilder smoothing approximated by EMA span=N)
    out['prev_close'] = g['Close'].shift(1)
    out['TR'] = np.maximum.reduce([
        out['High'] - out['Low'],
        (out['High'] - out['prev_close']).abs(),
        (out['prev_close'] - out['Low']).abs(),
    ])
    out['ATR_5']  = g['TR'].transform(lambda x: x.ewm(alpha=1/short, adjust=False).mean())
    out['ATR_15'] = g['TR'].transform(lambda x: x.ewm(alpha=1/long , adjust=False).mean())
    out['ATR_Ratio'] = out['ATR_5'] / out['ATR_15']

    # ADX (simplified — short version)
    out['prev_high'] = g['High'].shift(1)
    out['prev_low']  = g['Low'].shift(1)
    plus_dm  = np.where((out['High'] > out['prev_high']) &
                        (out['High'] - out['prev_high'] > out['prev_low'] - out['Low']),
                        out['High'] - out['prev_high'], 0)
    minus_dm = np.where((out['prev_low'] > out['Low']) &
                        (out['prev_low'] - out['Low'] > out['High'] - out['prev_high']),
                        out['prev_low'] - out['Low'], 0)
    out['+DM'] = plus_dm
    out['-DM'] = minus_dm
    plus_di_15  = (g['+DM'].transform(lambda x: x.ewm(alpha=1/long, adjust=False).mean())
                   / out['ATR_15']) * 100
    minus_di_15 = (g['-DM'].transform(lambda x: x.ewm(alpha=1/long, adjust=False).mean())
                   / out['ATR_15']) * 100
    dx_15 = (plus_di_15 - minus_di_15).abs() / (plus_di_15 + minus_di_15) * 100
    out['ADX_15'] = g['Ticker'].transform(lambda _: dx_15.ewm(alpha=1/long, adjust=False).mean())

    # Stochastic
    out['Lowest_5D']  = g['Low' ].transform(lambda x: x.rolling(short).min())
    out['High_5D']    = g['High'].transform(lambda x: x.rolling(short).max())
    out['Stochastic_5'] = ((out['Close'] - out['Lowest_5D'])
                            / (out['High_5D'] - out['Lowest_5D'])) * 100
    out['Lowest_15D'] = g['Low' ].transform(lambda x: x.rolling(long).min())
    out['High_15D']   = g['High'].transform(lambda x: x.rolling(long).max())
    out['Stochastic_15'] = ((out['Close'] - out['Lowest_15D'])
                             / (out['High_15D'] - out['Lowest_15D'])) * 100
    out['Stochastic_Ratio'] = out['Stochastic_5'] / out['Stochastic_15']

    # RSI
    diff = g['Close'].diff()
    up   = diff.clip(lower=0)
    down = (-diff).clip(lower=0)
    out['RSI_5']  = 100 - 100 / (1 + g['Ticker'].transform(lambda _:
                    up.ewm(alpha=1/short, adjust=False).mean() /
                    down.ewm(alpha=1/short, adjust=False).mean()))
    out['RSI_15'] = 100 - 100 / (1 + g['Ticker'].transform(lambda _:
                    up.ewm(alpha=1/long , adjust=False).mean() /
                    down.ewm(alpha=1/long , adjust=False).mean()))
    out['RSI_ratio'] = out['RSI_5'] / out['RSI_15']

    # MACD
    ewm_short = g['Close'].transform(lambda x: x.ewm(span=short, adjust=False).mean())
    ewm_long  = g['Close'].transform(lambda x: x.ewm(span=long , adjust=False).mean())
    out['MACD'] = ewm_long - ewm_short

    # Bollinger %B (15-day, 2 SD)
    ma  = g['Close'].transform(lambda x: x.rolling(long).mean())
    sd  = g['Close'].transform(lambda x: x.rolling(long).std())
    upper = ma + 2 * sd
    lower = ma - 2 * sd
    out['%B'] = (out['Close'] - lower) / (upper - lower)

    return out


# ===========================================================================
# 3.  Apply saved regime model
# ===========================================================================

def apply_regime_model(df, model_dir=MODEL_DIR):
    regime_path = os.path.join(model_dir, 'regime_model.joblib')
    if not os.path.exists(regime_path):
        log(f'Regime model not found at {regime_path}.  Save your trained '
            'regime model + scaler + feature list as a dict to that path.',
            level='ERROR')
        sys.exit(1)
    bundle = joblib.load(regime_path)
    rmodel  = bundle['model']
    scaler  = bundle['scaler']
    feats   = bundle['features']

    valid = df.dropna(subset=feats)
    Xz = scaler.transform(valid[feats].values.astype(np.float32))
    try:
        # HMM with per-ticker lengths
        lengths = valid.groupby('Ticker', sort=False).size().tolist()
        labels  = rmodel.predict(Xz, lengths=lengths)
    except TypeError:
        # GMM / KMeans
        labels = rmodel.predict(Xz)

    out = df.copy()
    out['Regime'] = np.nan
    out.loc[valid.index, 'Regime'] = labels
    return out


# ===========================================================================
# 4.  Generate today's signals
# ===========================================================================

def generate_today_signals(df, today, model_dir=MODEL_DIR,
                           min_proba=MIN_PROBA,
                           exclude_regimes=EXCLUDE_REGIMES):
    today_rows = df[df['Date'].dt.date == today]
    if today_rows.empty:
        log(f'No data for today ({today}).  Market closed?  Aborting.', 'WARN')
        return pd.DataFrame()

    bundles = {}
    pattern = f'{MODEL_TYPE}_{GROUP_COL}_'
    for fname in os.listdir(model_dir):
        if not (fname.startswith(pattern) and fname.endswith('.joblib')):
            continue
        try:
            r = int(fname.replace(pattern, '').replace('.joblib', ''))
        except ValueError:
            continue
        if r in exclude_regimes:
            continue
        bundles[r] = joblib.load(os.path.join(model_dir, fname))

    chunks = []
    for r, b in bundles.items():
        rows = today_rows[today_rows['Regime'] == r].dropna(subset=b['features'])
        if rows.empty:
            continue
        proba = b['model'].predict_proba(rows[b['features']].values)[:, 1]
        thresh = max(b['threshold'], min_proba)
        fired = proba >= thresh
        if fired.any():
            chunks.append(pd.DataFrame({
                'Ticker': rows['Ticker'].values[fired],
                'Regime': r,
                'Proba':  proba[fired],
            }))
    if not chunks:
        log('No signals fired today.')
        return pd.DataFrame()
    sig = pd.concat(chunks).sort_values('Proba', ascending=False).reset_index(drop=True)
    log(f'{len(sig)} signals fired today across regimes {sorted(bundles.keys())}')
    return sig


# ===========================================================================
# 5.  Position state (CSV-based — survives restarts)
# ===========================================================================

def load_open_positions():
    if os.path.exists(POSITIONS_FILE):
        df = pd.read_csv(POSITIONS_FILE, parse_dates=['buy_date', 'planned_sell_date'])
        return df
    return pd.DataFrame(columns=['ticker', 'shares', 'buy_date', 'buy_price',
                                  'planned_sell_date', 'regime', 'proba',
                                  'order_id', 'status'])


def save_open_positions(df):
    df.to_csv(POSITIONS_FILE, index=False)


def append_to_trade_log(rows):
    df_new = pd.DataFrame(rows)
    if os.path.exists(TRADES_FILE):
        df_old = pd.read_csv(TRADES_FILE)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(TRADES_FILE, index=False)


# ===========================================================================
# 6.  Alpaca order layer (paper trading)
# ===========================================================================

def get_alpaca_client():
    try:
        from alpaca.trading.client import TradingClient
    except ImportError:
        log('alpaca-py not installed.  pip install alpaca-py', 'ERROR')
        sys.exit(1)
    api_key    = os.environ.get('ALPACA_API_KEY')
    secret_key = os.environ.get('ALPACA_SECRET_KEY')
    if not api_key or not secret_key:
        log('ALPACA_API_KEY / ALPACA_SECRET_KEY env vars not set.', 'ERROR')
        sys.exit(1)
    return TradingClient(api_key, secret_key, paper=True)


def get_alpaca_cash(client):
    acct = client.get_account()
    return float(acct.cash)


def place_buy(client, ticker, shares):
    if DRY_RUN:
        log(f'  [DRY RUN] Would BUY {shares} shares of {ticker}')
        return f'dry_run_{ticker}_{int(time.time())}'
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums    import OrderSide, TimeInForce
    req = MarketOrderRequest(symbol=ticker, qty=shares, side=OrderSide.BUY,
                             time_in_force=TimeInForce.DAY)
    order = client.submit_order(req)
    log(f'  Submitted BUY order: {shares} {ticker}  (id={order.id})')
    return str(order.id)


def place_sell(client, ticker, shares):
    if DRY_RUN:
        log(f'  [DRY RUN] Would SELL {shares} shares of {ticker}')
        return f'dry_run_{ticker}_{int(time.time())}'
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums    import OrderSide, TimeInForce
    req = MarketOrderRequest(symbol=ticker, qty=shares, side=OrderSide.SELL,
                             time_in_force=TimeInForce.DAY)
    order = client.submit_order(req)
    log(f'  Submitted SELL order: {shares} {ticker}  (id={order.id})')
    return str(order.id)


# ===========================================================================
# 7.  Main daily routine
# ===========================================================================

def main():
    log('=' * 70)
    log(f'DAILY RUNNER — {dt.date.today().isoformat()}'
        f'  (DRY_RUN = {DRY_RUN})')
    log('=' * 70)

    # ---- 1. Refresh data ------------------------------------------------
    tickers = get_sp500_tickers()
    bars    = download_recent_bars(tickers, lookback_days=120)
    today   = bars['Date'].max().date()
    log(f'Latest bar date: {today}')

    # ---- 2. Compute indicators -----------------------------------------
    log('Computing indicators ...')
    df = compute_indicators(bars)

    # ---- 3. Apply regime model ----------------------------------------
    log('Applying regime model ...')
    df = apply_regime_model(df)

    # ---- 4. Generate today's signals ----------------------------------
    log('Generating today\'s signals ...')
    signals = generate_today_signals(df, today)
    if signals.empty:
        log('No signals → nothing to do today.  Done.')
        return

    # ---- 5. Connect to Alpaca + check existing positions --------------
    client = get_alpaca_client()
    cash   = get_alpaca_cash(client)
    log(f'Alpaca paper cash available: ${cash:,.2f}')

    open_df = load_open_positions()
    log(f'Currently {len(open_df)} open positions tracked locally')

    # ---- 6. Close positions whose holding period ends today -----------
    closing_today = open_df[open_df['planned_sell_date'].dt.date <= today]
    closed_log = []
    for _, pos in closing_today.iterrows():
        log(f'Closing position: {pos["ticker"]} ({pos["shares"]} sh, '
            f'bought {pos["buy_date"].date()})')
        order_id = place_sell(client, pos['ticker'], int(pos['shares']))
        closed_log.append({
            'event':         'CLOSE',
            'ticker':        pos['ticker'],
            'shares':        pos['shares'],
            'buy_date':      pos['buy_date'].isoformat(),
            'sell_date':     today.isoformat(),
            'buy_price':     pos['buy_price'],
            'sell_order_id': order_id,
            'regime':        pos.get('regime', None),
            'proba':         pos.get('proba', None),
        })
    open_df = open_df[~open_df.index.isin(closing_today.index)].reset_index(drop=True)

    # ---- 7. Open new positions from today's signals -------------------
    held = set(open_df['ticker'].tolist())
    fresh = signals[~signals['Ticker'].isin(held)].head(MAX_NEW_PER_DAY)
    if fresh.empty:
        log('No new signals after dedup → no new positions today.')
    else:
        n = len(fresh)
        # Equal-weight allocation against current cash
        alloc = cash / n if cash > 0 else 0
        log(f'Allocating ${alloc:,.2f} per new trade across {n} signals')

        opened_log = []
        for _, sig in fresh.iterrows():
            tk = sig['Ticker']
            # Use today's close as a rough sizing reference (real fill = next open)
            try:
                last_close = float(df[(df['Ticker'] == tk) &
                                      (df['Date'].dt.date == today)]['Close'].iloc[-1])
            except IndexError:
                log(f'  Skip {tk}: no close price', 'WARN')
                continue
            shares = int(alloc // last_close)
            if shares < 1:
                log(f'  Skip {tk}: alloc too small for 1 share at ${last_close:.2f}', 'WARN')
                continue
            order_id = place_buy(client, tk, shares)

            # Plan the sell date: today + HOLDING_DAYS trading days
            future = pd.bdate_range(start=today + dt.timedelta(days=1),
                                    periods=HOLDING_DAYS)
            sell_date = future[-1].date()

            new_pos = {
                'ticker':            tk,
                'shares':            shares,
                'buy_date':          today,
                'buy_price':         last_close,
                'planned_sell_date': sell_date,
                'regime':            int(sig['Regime']),
                'proba':             float(sig['Proba']),
                'order_id':          order_id,
                'status':            'OPEN',
            }
            open_df = pd.concat([open_df, pd.DataFrame([new_pos])],
                                ignore_index=True)
            opened_log.append({**new_pos, 'event': 'OPEN'})

        if opened_log:
            append_to_trade_log(opened_log)

    if closed_log:
        append_to_trade_log(closed_log)

    # ---- 8. Save updated state + summary ------------------------------
    save_open_positions(open_df)
    log('-' * 70)
    log(f'Open positions: {len(open_df)}  |  Cash: ${cash:,.2f}'
        f'  |  Closed today: {len(closed_log)}  |  Opened today: '
        f'{len(opened_log) if "opened_log" in dir() else 0}')
    log('Done.\n')


if __name__ == '__main__':
    main()


# ===========================================================================
# SCHEDULING GUIDE — make this run automatically every day
# ===========================================================================
#
# macOS / Linux  →  cron
# ------------------------------------------------------------------
# 1. Open your crontab:    crontab -e
# 2. Add a line that runs this script at 4:30pm ET every weekday:
#
#      30 16 * * 1-5  cd "/Users/leemingjun/Desktop/Random Forest Model" && \
#                     /usr/local/bin/python3 "Drafts/Step 8 - Paper Trading Daily Runner.py" \
#                     >> "PaperTrading/logs/cron.log" 2>&1
#
#    (Adjust 16 → 21 if your machine is on UTC; cron uses your machine's local time.)
# 3. Save + exit.  Check it ran:  ls -lt PaperTrading/logs/
#
# Windows  →  Task Scheduler
# ------------------------------------------------------------------
# 1. Open Task Scheduler (search in Start menu).
# 2. Create Basic Task → name "Daily Runner" → trigger: Daily at 4:30 PM
#    → action: Start a program
#    → program/script:   C:\Python311\python.exe
#    → arguments:        "C:\Users\<you>\Desktop\Random Forest Model\Drafts\Step 8 - Paper Trading Daily Runner.py"
#    → start in:         C:\Users\<you>\Desktop\Random Forest Model
# 3. Check the "Run with highest privileges" box if it complains about the env vars.
#
# Either way, after the first 1-2 days check PaperTrading/logs/ to confirm
# the script ran and nothing errored.
