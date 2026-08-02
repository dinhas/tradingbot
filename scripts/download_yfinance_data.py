"""
Download all required yfinance series for Tier 3 cross-asset features.
Run once. Output: data/yfinance/*.parquet

Usage:
    python scripts/download_yfinance_data.py
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TICKERS = {
    '^VIX': 'VIX',
    'GC=F': 'GC_F',
    '^GSPC': 'GSPC',
    'CL=F': 'CL_F',
    '^TNX': 'TNX',
}


def main():
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    out_dir = PROJECT_ROOT / 'data' / 'yfinance'
    out_dir.mkdir(parents=True, exist_ok=True)

    for ticker, filename in TICKERS.items():
        print(f"Downloading {ticker}...")
        try:
            data = yf.download(ticker, start='2015-12-01', end='2026-01-01',
                               interval='1d', auto_adjust=True)
            data.index = pd.to_datetime(data.index).tz_localize(None)
            data.index.name = 'date'
            out_path = out_dir / f'{filename}.parquet'
            data.to_parquet(out_path)
            print(f"  Saved {len(data)} rows to {out_path}")
        except Exception as e:
            print(f"  WARNING: Failed to download {ticker}: {e}")
            continue

    print("\nDone. All yfinance series downloaded.")


if __name__ == '__main__':
    main()
