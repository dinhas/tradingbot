"""
Download all required FRED series for Tier 3 macro features.
Run once. Output: data/fred/*.parquet

Usage:
    python scripts/download_fred_data.py --api-key YOUR_KEY
    or set FRED_API_KEY environment variable
"""
import os
import sys
import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / '.env')
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SERIES = {
    'GS10': '10-Year Treasury Constant Maturity Rate',
    'GS2': '2-Year Treasury Constant Maturity Rate',
    'CPIAUCSL': 'Consumer Price Index for All Urban Consumers',
    'FEDFUNDS': 'Federal Funds Effective Rate',
    'PAYEMS': 'All Employees, Total Nonfarm',
    'DFII10': 'Market Yield on 10-Year TIPS',
    'DTWEXBGS': 'Trade Weighted U.S. Dollar Index: Broad',
}


def main():
    parser = argparse.ArgumentParser(description="Download FRED macro data")
    parser.add_argument('--api-key', type=str, default=None,
                        help='FRED API key (or set FRED_API_KEY env var)')
    parser.add_argument('--start', type=str, default='2015-12-01',
                        help='Start date (default: 2015-12-01)')
    parser.add_argument('--output-dir', type=str, default='data/fred',
                        help='Output directory (default: data/fred)')
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get('FRED_API_KEY')
    if not api_key:
        print("ERROR: No FRED API key provided.")
        print("  Set FRED_API_KEY in .env file or pass --api-key argument.")
        print("  Register free at: https://fred.stlouisfed.org/docs/api/api_key.html")
        sys.exit(1)

    try:
        from fredapi import Fred
    except ImportError:
        print("ERROR: fredapi not installed. Run: pip install fredapi")
        sys.exit(1)

    fred = Fred(api_key=api_key)
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for series_id, description in SERIES.items():
        print(f"Downloading {series_id}: {description}...")
        try:
            series = fred.get_series(series_id, observation_start=args.start)
            df = series.to_frame(name=series_id)
            df.index = pd.to_datetime(df.index)
            df.index.name = 'date'
            out_path = out_dir / f'{series_id}.parquet'
            df.to_parquet(out_path)
            print(f"  Saved {len(df)} rows to {out_path}")
        except Exception as e:
            print(f"  WARNING: Failed to download {series_id}: {e}")
            continue

    print("\nDone. All FRED series downloaded.")


if __name__ == '__main__':
    main()
