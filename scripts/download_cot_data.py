"""
Download COT positioning data from Xoomar API for all 4 FX pairs.
Run once. Output: data/cot/*_cot.parquet

Usage:
    python scripts/download_cot_data.py
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

COT_SLUGS = {
    'EURUSD': 'euro-fx',
    'GBPUSD': 'british-pound-sterling',
    'USDJPY': 'japanese-yen',
    'USDCHF': 'swiss-franc',
}


def download_cot(slug, start='2010-01-01'):
    """Download COT history from Xoomar API."""
    url = f'https://xoomar.com/api/markets/cot/{slug}'
    params = {'from': start, 'to': '2026-12-31'}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()['data']

    df = pd.DataFrame(data)
    df['reportDate'] = pd.to_datetime(df['reportDate'])
    df.set_index('reportDate', inplace=True)
    df.sort_index(inplace=True)
    return df


def main():
    out_dir = PROJECT_ROOT / 'data' / 'cot'
    out_dir.mkdir(parents=True, exist_ok=True)

    for asset, slug in COT_SLUGS.items():
        print(f"Downloading COT for {asset} (slug: {slug})...")
        try:
            df = download_cot(slug)
            out_path = out_dir / f'{asset}_cot.parquet'
            df.to_parquet(out_path)
            print(f"  Saved {len(df)} rows to {out_path}")
            print(f"  Columns: {list(df.columns[:10])}...")
        except Exception as e:
            print(f"  WARNING: Failed to download {asset}: {e}")
            continue
        time.sleep(2)  # Rate limit: 30 req/min

    print("\nDone. All COT data downloaded.")


if __name__ == '__main__':
    main()
