import os
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("LiveExecution")

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


def is_cot_stale(latest_date: datetime) -> bool:
    """
    Determines if COT report cache is stale.
    CFTC publishes COT data weekly on Fridays around 20:30 UTC.
    The report represents the Tuesday snapshot of that week.
    """
    now = datetime.utcnow()
    # Find most recent Tuesday for which a report should be published
    days_since_tuesday = (now.weekday() - 1) % 7
    tuesday_this_week = (now - timedelta(days=days_since_tuesday)).replace(hour=0, minute=0, second=0, microsecond=0)

    # If before Friday 21:00 UTC, the latest report available is for Tuesday of the PREVIOUS week.
    # If after Friday 21:00 UTC, the latest report available is for Tuesday of THIS week.
    friday_this_week = now + timedelta(days=((4 - now.weekday()) % 7))
    friday_release = friday_this_week.replace(hour=21, minute=0, second=0, microsecond=0)

    # Adjust if we are in the same week but before the Friday release
    if now < friday_release:
        # Expected latest Tuesday is the previous week's Tuesday
        expected_tuesday = tuesday_this_week - timedelta(days=7)
    else:
        # Expected latest Tuesday is this week's Tuesday
        expected_tuesday = tuesday_this_week

    return latest_date < expected_tuesday


def is_macro_stale(latest_date: datetime) -> bool:
    """
    Determines if daily macro data (FRED / yfinance) cache is stale.
    Checks against market close at 22:00 UTC.
    """
    now = datetime.utcnow()
    # Sat, Sun -> expect Friday's data
    if now.weekday() in [5, 6]:
        days_since_friday = (now.weekday() - 4) % 7
        expected_date = (now - timedelta(days=days_since_friday)).replace(hour=0, minute=0, second=0, microsecond=0)
    # Mon morning -> expect last Friday's data
    elif now.weekday() == 0 and now.hour < 22:
        expected_date = (now - timedelta(days=3)).replace(hour=0, minute=0, second=0, microsecond=0)
    # Weekdays after market close -> expect today's data
    elif now.hour >= 22:
        expected_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
    # Weekdays before market close -> expect yesterday's data
    else:
        expected_date = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    # Let's allow a small 1-day slack for FRED data which can have reporting lags
    return latest_date < (expected_date - timedelta(days=1))


class MarketDataCacheManager:
    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir) if data_dir else PROJECT_ROOT / 'data'
        self.cot_dir = self.data_dir / 'cot'
        self.fred_dir = self.data_dir / 'fred'
        self.yf_dir = self.data_dir / 'yfinance'

    def check_cache_status(self):
        """
        Scans data directories and determines cache path.
        Returns:
            status: dict containing 'has_cache', 'stale_cot', 'stale_macro', 'missing_files'
        """
        status = {
            'has_cache': True,
            'stale_cot': False,
            'stale_macro': False,
            'missing_files': []
        }

        # Check COT
        for asset in COT_SLUGS.keys():
            path = self.cot_dir / f"{asset}_cot.parquet"
            if not path.exists():
                status['has_cache'] = False
                status['missing_files'].append(f"cot/{asset}_cot.parquet")
            else:
                try:
                    df = pd.read_parquet(path)
                    if df.empty:
                        status['has_cache'] = False
                        status['missing_files'].append(f"cot/{asset}_cot.parquet")
                    else:
                        latest_date = df.index.max()
                        if isinstance(latest_date, pd.Timestamp):
                            latest_date = latest_date.to_pydatetime()
                        if is_cot_stale(latest_date):
                            status['stale_cot'] = True
                except Exception as e:
                    logger.warning(f"Error reading COT cache for {asset}: {e}")
                    status['has_cache'] = False
                    status['missing_files'].append(f"cot/{asset}_cot.parquet")

        # Check FRED
        for series_id in FRED_SERIES.keys():
            path = self.fred_dir / f"{series_id}.parquet"
            if not path.exists():
                status['has_cache'] = False
                status['missing_files'].append(f"fred/{series_id}.parquet")
            else:
                try:
                    df = pd.read_parquet(path)
                    if df.empty:
                        status['has_cache'] = False
                        status['missing_files'].append(f"fred/{series_id}.parquet")
                    else:
                        latest_date = df.index.max()
                        if isinstance(latest_date, pd.Timestamp):
                            latest_date = latest_date.to_pydatetime()
                        if is_macro_stale(latest_date):
                            status['stale_macro'] = True
                except Exception as e:
                    logger.warning(f"Error reading FRED cache for {series_id}: {e}")
                    status['has_cache'] = False
                    status['missing_files'].append(f"fred/{series_id}.parquet")

        # Check yfinance
        for ticker, filename in YF_TICKERS.items():
            path = self.yf_dir / f"{filename}.parquet"
            if not path.exists():
                status['has_cache'] = False
                status['missing_files'].append(f"yfinance/{filename}.parquet")
            else:
                try:
                    df = pd.read_parquet(path)
                    if df.empty:
                        status['has_cache'] = False
                        status['missing_files'].append(f"yfinance/{filename}.parquet")
                    else:
                        latest_date = df.index.max()
                        if isinstance(latest_date, pd.Timestamp):
                            latest_date = latest_date.to_pydatetime()
                        if is_macro_stale(latest_date):
                            status['stale_macro'] = True
                except Exception as e:
                    logger.warning(f"Error reading yfinance cache for {ticker}: {e}")
                    status['has_cache'] = False
                    status['missing_files'].append(f"yfinance/{filename}.parquet")

        return status

    def sync(self):
        """
        Performs the complete cache validation and synchronization process.
        """
        logger.info("Initializing market data cache synchronization...")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cot_dir.mkdir(parents=True, exist_ok=True)
        self.fred_dir.mkdir(parents=True, exist_ok=True)
        self.yf_dir.mkdir(parents=True, exist_ok=True)

        status = self.check_cache_status()

        if not status['has_cache']:
            logger.info(f"Cold start detected. Missing files: {status['missing_files']}. Performing full pull...")
            self.download_all(full=True)
        elif status['stale_cot'] or status['stale_macro']:
            logger.info(f"Stale cache detected (stale_cot={status['stale_cot']}, stale_macro={status['stale_macro']}). Performing incremental refresh...")
            self.download_all(full=False)
        else:
            logger.info("Market data cache is valid and fresh. Cache hit!")

        self.validate_data_integrity()

    def download_all(self, full=True):
        """Downloads FRED, yfinance, and COT data."""
        self.download_cot(full=full)
        self.download_fred(full=full)
        self.download_yfinance(full=full)

    def download_cot(self, full=True):
        import requests
        logger.info(f"Downloading COT positioning data (full={full})...")
        downloaded = 0

        for asset, slug in COT_SLUGS.items():
            try:
                out_path = self.cot_dir / f"{asset}_cot.parquet"
                start_date = "2010-01-01"
                existing_df = pd.DataFrame()

                if not full and out_path.exists():
                    try:
                        existing_df = pd.read_parquet(out_path)
                        if not existing_df.empty:
                            last_dt = existing_df.index.max()
                            start_date = (last_dt + timedelta(days=1)).strftime("%Y-%m-%d")
                    except Exception as e:
                        logger.warning(f"Error reading {asset} COT during incremental download: {e}")

                url = f'https://xoomar.com/api/markets/cot/{slug}'
                params = {'from': start_date, 'to': '2026-12-31'}
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json().get('data', [])

                if not data:
                    logger.info(f"  {asset}: No new COT data available.")
                    continue

                df = pd.DataFrame(data)
                df['reportDate'] = pd.to_datetime(df['reportDate'])
                df.set_index('reportDate', inplace=True)
                df.sort_index(inplace=True)

                if not existing_df.empty:
                    df = pd.concat([existing_df, df])
                    df = df[~df.index.duplicated(keep='last')]
                    df.sort_index(inplace=True)

                df.to_parquet(out_path)
                logger.info(f"  {asset}: {len(df)} rows saved to disk. Path taken: {'full pull' if full else 'incremental refresh'}")
                downloaded += 1
                time.sleep(1)
            except Exception as e:
                logger.error(f"  {asset} COT download failed: {e}")

        logger.info(f"COT Download Complete: {downloaded}/{len(COT_SLUGS)}")

    def download_fred(self, full=True):
        try:
            from fredapi import Fred
            api_key = os.environ.get('FRED_API_KEY')
            if not api_key:
                logger.error("FRED_API_KEY is missing. FRED download cannot proceed.")
                return
            fred = Fred(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize FRED API client: {e}")
            return

        logger.info(f"Downloading FRED macroeconomic indicators (full={full})...")
        downloaded = 0

        for series_id, desc in FRED_SERIES.items():
            try:
                out_path = self.fred_dir / f"{series_id}.parquet"
                start_date = "2015-12-01"
                existing_df = pd.DataFrame()

                if not full and out_path.exists():
                    try:
                        existing_df = pd.read_parquet(out_path)
                        if not existing_df.empty:
                            last_dt = existing_df.index.max()
                            start_date = (last_dt + timedelta(days=1)).strftime("%Y-%m-%d")
                    except Exception as e:
                        logger.warning(f"Error reading FRED series {series_id} cache: {e}")

                # Avoid unnecessary requests if start_date is in the future
                if datetime.strptime(start_date, "%Y-%m-%d") >= datetime.utcnow():
                    logger.info(f"  {series_id}: Up to date.")
                    downloaded += 1
                    continue

                series = fred.get_series(series_id, observation_start=start_date)
                if series.empty:
                    logger.info(f"  {series_id}: No new FRED data available.")
                    downloaded += 1
                    continue

                df = series.to_frame(name=series_id)
                df.index = pd.to_datetime(df.index)
                df.index.name = 'date'

                if not existing_df.empty:
                    df = pd.concat([existing_df, df])
                    df = df[~df.index.duplicated(keep='last')]
                    df.sort_index(inplace=True)

                df.to_parquet(out_path)
                logger.info(f"  {series_id}: {len(df)} rows saved. Path taken: {'full pull' if full else 'incremental refresh'}")
                downloaded += 1
            except Exception as e:
                logger.error(f"  FRED Series {series_id} download failed: {e}")

        logger.info(f"FRED Download Complete: {downloaded}/{len(FRED_SERIES)}")

    def download_yfinance(self, full=True):
        try:
            import yfinance as yf
        except Exception as e:
            logger.error(f"yfinance library not installed: {e}")
            return

        logger.info(f"Downloading yfinance market indicators (full={full})...")
        downloaded = 0

        for ticker, filename in YF_TICKERS.items():
            try:
                out_path = self.yf_dir / f"{filename}.parquet"
                start_date = "2015-12-01"
                existing_df = pd.DataFrame()

                if not full and out_path.exists():
                    try:
                        existing_df = pd.read_parquet(out_path)
                        if not existing_df.empty:
                            last_dt = existing_df.index.max()
                            # Fetch with 3-day overlap to handle timezone adjustments
                            start_date = (last_dt - timedelta(days=3)).strftime("%Y-%m-%d")
                    except Exception as e:
                        logger.warning(f"Error reading yfinance ticker {ticker} cache: {e}")

                data = yf.download(ticker, start=start_date, end='2026-12-31',
                                   interval='1d', auto_adjust=True, progress=False)

                if data.empty:
                    logger.info(f"  {ticker}: No new data returned.")
                    downloaded += 1
                    continue

                data.index = pd.to_datetime(data.index).tz_localize(None)
                data.index.name = 'date'

                if not existing_df.empty:
                    data = pd.concat([existing_df, data])
                    data = data[~data.index.duplicated(keep='last')]
                    data.sort_index(inplace=True)

                data.to_parquet(out_path)
                logger.info(f"  {ticker} ({filename}): {len(data)} rows saved. Path taken: {'full pull' if full else 'incremental refresh'}")
                downloaded += 1
            except Exception as e:
                logger.error(f"  yfinance Ticker {ticker} download failed: {e}")

        logger.info(f"yfinance Download Complete: {downloaded}/{len(YF_TICKERS)}")

    def validate_data_integrity(self):
        """
        Validates that after sync, the feature engine has enough history
        to compute valid features. Blocks system if validation fails.
        """
        logger.info("Running post-download data integrity validation...")
        from Alpha.src.data_loader import DataLoader
        loader = DataLoader(data_dir=self.data_dir)

        macro_df = loader.load_macro_data()
        cot_df = loader.load_cot_data()

        logger.info(f"Validated Macro Rows: {len(macro_df)}, Columns: {list(macro_df.columns)}")
        logger.info(f"Validated COT Rows: {len(cot_df)}, Columns: {list(cot_df.columns)}")

        # Ensure we have at least 100 macro rows to calculate returns and moving averages
        if len(macro_df) < 100:
            msg = f"Data validation FAILED: Macro data has only {len(macro_df)} rows. Need at least 100 rows for proper indicator calculation."
            logger.critical(msg)
            raise ValueError(msg)

        # Ensure we have at least 52 COT rows to compute rolling 52-week indicators
        if len(cot_df) < 52:
            msg = f"Data validation FAILED: COT positioning data has only {len(cot_df)} rows. Need at least 52 weeks of positioning history."
            logger.critical(msg)
            raise ValueError(msg)

        # Validate that essential columns are present
        required_macro = ['dxy_return_5d', 'sp500_return_5d']
        missing_macro = [c for c in required_macro if c not in macro_df.columns]
        if missing_macro:
            msg = f"Data validation FAILED: Macro data is missing critical features: {missing_macro}"
            logger.critical(msg)
            raise ValueError(msg)

        logger.info("Data integrity validation PASSED. All baseline requirements satisfied.")
