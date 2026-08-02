"""
Unified market data refresh script.
Downloads COT, FRED macro, and yfinance cross-asset data.
Designed to be called daily at midnight by the live execution system.

Usage:
    python scripts/refresh_market_data.py

Saves to: data/cot/, data/fred/, data/yfinance/
"""
import sys
import time
import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("MarketDataRefresh")

# --- COT Configuration ---
COT_SLUGS = {
    'EURUSD': 'euro-fx',
    'GBPUSD': 'british-pound-sterling',
    'USDJPY': 'japanese-yen',
    'USDCHF': 'swiss-franc',
}

# --- FRED Configuration ---
FRED_SERIES = {
    'GS10': '10-Year Treasury Constant Maturity Rate',
    'GS2': '2-Year Treasury Constant Maturity Rate',
    'CPIAUCSL': 'Consumer Price Index for All Urban Consumers',
    'FEDFUNDS': 'Federal Funds Effective Rate',
    'PAYEMS': 'All Employees, Total Nonfarm',
    'DFII10': 'Market Yield on 10-Year TIPS',
    'DTWEXBGS': 'Trade Weighted U.S. Dollar Index: Broad',
}

# --- yfinance Configuration ---
YF_TICKERS = {
    '^VIX': 'VIX',
    'GC=F': 'GC_F',
    '^GSPC': 'GSPC',
    'CL=F': 'CL_F',
    '^TNX': 'TNX',
}


def download_cot(out_dir):
    """Download COT data from Xoomar API."""
    import requests
    log.info("=== Downloading COT data ===")
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    for asset, slug in COT_SLUGS.items():
        try:
            url = f'https://xoomar.com/api/markets/cot/{slug}'
            params = {'from': '2010-01-01', 'to': '2026-12-31'}
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()['data']

            df = pd.DataFrame(data)
            df['reportDate'] = pd.to_datetime(df['reportDate'])
            df.set_index('reportDate', inplace=True)
            df.sort_index(inplace=True)

            out_path = out_dir / f'{asset}_cot.parquet'
            df.to_parquet(out_path)
            log.info(f"  {asset}: {len(df)} rows saved")
            downloaded += 1
            time.sleep(2)  # Rate limit
        except Exception as e:
            log.warning(f"  {asset} FAILED: {e}")

    log.info(f"COT download complete: {downloaded}/{len(COT_SLUGS)}")
    return downloaded


def download_fred(out_dir):
    """Download FRED macro data."""
    log.info("=== Downloading FRED macro data ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from fredapi import Fred
        from dotenv import load_dotenv
        import os
        load_dotenv(PROJECT_ROOT / '.env')
        api_key = os.environ.get('FRED_API_KEY')
        if not api_key:
            log.error("  FRED_API_KEY not set. Skipping FRED download.")
            return 0
    except ImportError:
        log.error("  fredapi not installed. Run: pip install fredapi")
        return 0

    fred = Fred(api_key=api_key)
    downloaded = 0

    for series_id, description in FRED_SERIES.items():
        try:
            series = fred.get_series(series_id, observation_start='2015-12-01')
            df = series.to_frame(name=series_id)
            df.index = pd.to_datetime(df.index)
            df.index.name = 'date'

            out_path = out_dir / f'{series_id}.parquet'
            df.to_parquet(out_path)
            log.info(f"  {series_id}: {len(df)} rows saved")
            downloaded += 1
        except Exception as e:
            log.warning(f"  {series_id} FAILED: {e}")

    log.info(f"FRED download complete: {downloaded}/{len(FRED_SERIES)}")
    return downloaded


def download_yfinance(out_dir):
    """Download yfinance cross-asset data."""
    log.info("=== Downloading yfinance data ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import yfinance as yf
    except ImportError:
        log.error("  yfinance not installed. Run: pip install yfinance")
        return 0

    downloaded = 0

    for ticker, filename in YF_TICKERS.items():
        try:
            data = yf.download(ticker, start='2015-12-01', end='2026-12-31',
                               interval='1d', auto_adjust=True, progress=False)
            if data.empty:
                log.warning(f"  {ticker}: no data returned")
                continue
            data.index = pd.to_datetime(data.index).tz_localize(None)
            data.index.name = 'date'

            out_path = out_dir / f'{filename}.parquet'
            data.to_parquet(out_path)
            log.info(f"  {ticker}: {len(data)} rows saved")
            downloaded += 1
        except Exception as e:
            log.warning(f"  {ticker} FAILED: {e}")

    log.info(f"yfinance download complete: {downloaded}/{len(YF_TICKERS)}")
    return downloaded


def main():
    log.info("=" * 60)
    log.info("MARKET DATA REFRESH STARTED")
    log.info("=" * 60)

    data_dir = PROJECT_ROOT / 'data'

    cot_count = download_cot(data_dir / 'cot')
    fred_count = download_fred(data_dir / 'fred')
    yf_count = download_yfinance(data_dir / 'yfinance')

    total = cot_count + fred_count + yf_count
    log.info("=" * 60)
    log.info(f"REFRESH COMPLETE: {total} files downloaded")
    log.info(f"  COT: {cot_count}/{len(COT_SLUGS)}")
    log.info(f"  FRED: {fred_count}/{len(FRED_SERIES)}")
    log.info(f"  yfinance: {yf_count}/{len(YF_TICKERS)}")
    log.info("=" * 60)

    return total


if __name__ == '__main__':
    main()
