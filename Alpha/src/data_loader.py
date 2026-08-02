import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Alpha.src.feature_engine import FeatureEngine
from shared_constants import FX_ALPHA_ASSETS

class DataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.assets = FX_ALPHA_ASSETS

    def load_raw_data(self, max_rows: int = 0) -> Dict[str, pd.DataFrame]:
        """
        Loads OHLCV data for all assets from Parquet files.
        Ensures sorted timestamps and no duplicates.
        Args:
            max_rows: If > 0, take only the last N rows per asset (for fast iteration).
        """
        data_dict = {}
        for asset in self.assets:
            candidate_paths = [
                self.data_dir / f"{asset}_5m_backtest.parquet",
                self.data_dir / f"{asset}_5m.parquet",
                self.data_dir / f"{asset}_5m_2025.parquet",
                self.data_dir / asset / f"{asset}_5m.parquet",
            ]

            file_path = next((p for p in candidate_paths if p.exists()), None)
            if file_path is None:
                print(f"Warning: Data file for {asset} not found in {[str(p) for p in candidate_paths]}")
                continue

            df = pd.read_parquet(file_path)

            # Ensure timestamp is index and sorted
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

            df.sort_index(inplace=True)

            # Limit rows for fast iteration (before dedup to save memory)
            if max_rows > 0 and len(df) > max_rows:
                df = df.tail(max_rows)

            # Remove duplicates
            df = df[~df.index.duplicated(keep='first')]

            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: {asset} data is missing some OHLCV columns.")

            data_dict[asset] = df

        return data_dict

    def load_macro_data(self) -> pd.DataFrame:
        """
        Load and align all macro/cross-asset data to a daily index.
        Returns DataFrame with date index and columns for each macro feature.
        Returns empty DataFrame if data directories don't exist (graceful degradation).
        """
        macro_dfs = []

        # --- FRED Series ---
        fred_dir = self.data_dir / 'fred'
        if fred_dir.exists():
            gs10_path = fred_dir / 'GS10.parquet'
            gs2_path = fred_dir / 'GS2.parquet'

            if gs10_path.exists() and gs2_path.exists():
                gs10 = pd.read_parquet(gs10_path)['GS10']
                gs2 = pd.read_parquet(gs2_path)['GS2']
                macro_dfs.append((gs10 - gs2).rename('yield_curve_slope'))
                macro_dfs.append(gs10.pct_change().rename('us10y_change'))

            dfii10_path = fred_dir / 'DFII10.parquet'
            if dfii10_path.exists():
                dfii10 = pd.read_parquet(dfii10_path)['DFII10']
                macro_dfs.append(dfii10.rename('dollar_real_rate'))

            cpi_path = fred_dir / 'CPIAUCSL.parquet'
            if cpi_path.exists():
                cpi = pd.read_parquet(cpi_path)['CPIAUCSL']
                cpi_yoy = cpi.pct_change(12, fill_method=None)
                macro_dfs.append(cpi_yoy.rename('cpi_yoy'))

            fed_path = fred_dir / 'FEDFUNDS.parquet'
            if fed_path.exists():
                fed = pd.read_parquet(fed_path)['FEDFUNDS']
                macro_dfs.append(fed.rename('fed_rate_level'))

            payems_path = fred_dir / 'PAYEMS.parquet'
            if payems_path.exists():
                nfp = pd.read_parquet(payems_path)['PAYEMS']
                nfp_mom = nfp.diff() / nfp.rolling(3).mean()
                macro_dfs.append(nfp_mom.rename('nfp_momentum'))

        # --- yfinance Series ---
        yf_dir = self.data_dir / 'yfinance'
        if yf_dir.exists():
            for ticker, col_name in [
                ('VIX', 'vix_level'),
                ('GC_F', 'gold_price'),
                ('GSPC', 'sp500_price'),
                ('CL_F', 'oil_price'),
            ]:
                path = yf_dir / f'{ticker}.parquet'
                if path.exists():
                    df = pd.read_parquet(path)
                    series = df['Close']
                    if isinstance(series, pd.DataFrame):
                        series = series.iloc[:, 0]
                    macro_dfs.append(series.rename(col_name))

            # DXY from FRED is preferred, but yfinance TNX as fallback for 10Y
            tnx_path = yf_dir / 'TNX.parquet'
            if tnx_path.exists() and not (fred_dir.exists() and (fred_dir / 'GS10.parquet').exists()):
                df = pd.read_parquet(tnx_path)
                tnx = df['Close']
                if isinstance(tnx, pd.DataFrame):
                    tnx = tnx.iloc[:, 0]
                macro_dfs.append(tnx.rename('us10y_level'))

        # --- FRED DXY (consistent source) ---
        if fred_dir.exists():
            dxy_path = fred_dir / 'DTWEXBGS.parquet'
            if dxy_path.exists():
                dxy = pd.read_parquet(dxy_path)['DTWEXBGS']
                macro_dfs.append(dxy.rename('dxy_index'))

        if not macro_dfs:
            return pd.DataFrame()

        daily = pd.concat(macro_dfs, axis=1)
        daily.index = pd.to_datetime(daily.index)
        daily.index.name = 'date'
        daily.sort_index(inplace=True)

        # Compute derived features
        if 'dxy_index' in daily.columns:
            daily['dxy_return_5d'] = daily['dxy_index'].pct_change(5, fill_method=None)
        if 'gold_price' in daily.columns:
            daily['gold_return_5d'] = daily['gold_price'].pct_change(5, fill_method=None)
        if 'sp500_price' in daily.columns:
            daily['sp500_return_5d'] = daily['sp500_price'].pct_change(5, fill_method=None)
        if 'oil_price' in daily.columns:
            daily['oil_return_5d'] = daily['oil_price'].pct_change(5, fill_method=None)
        if 'vix_level' in daily.columns:
            daily['vix_regime'] = pd.cut(
                daily['vix_level'],
                bins=[0, 15, 25, 100],
                labels=[0, 1, 2]
            ).astype(float).fillna(1.0)

        keep_cols = [
            'vix_level', 'vix_regime', 'yield_curve_slope', 'us10y_change',
            'dxy_return_5d', 'gold_return_5d', 'sp500_return_5d', 'oil_return_5d',
            'cpi_yoy', 'fed_rate_level', 'dollar_real_rate', 'nfp_momentum',
        ]
        daily = daily[[c for c in keep_cols if c in daily.columns]]
        return daily

    def load_cot_data(self) -> pd.DataFrame:
        """
        Load and align all COT positioning data to a weekly index.
        Returns DataFrame with date index and columns for each pair's COT features.
        Returns empty DataFrame if data directory doesn't exist.
        """
        cot_dir = self.data_dir / 'cot'
        if not cot_dir.exists():
            return pd.DataFrame()

        cot_dfs = []
        for asset in self.assets:
            path = cot_dir / f'{asset}_cot.parquet'
            if not path.exists():
                continue

            df = pd.read_parquet(path)
            if 'levFundLong' in df.columns and 'levFundShort' in df.columns:
                net = df['levFundLong'] - df['levFundShort']
            elif 'non_commercial_long' in df.columns and 'non_commercial_short' in df.columns:
                net = df['non_commercial_long'] - df['non_commercial_short']
            else:
                continue

            cot_features = pd.DataFrame(index=net.index)
            cot_features[f'{asset}_cot_net'] = net
            cot_features[f'{asset}_cot_net_zscore'] = (
                (net - net.rolling(52).mean()) / (net.rolling(52).std() + 1e-8)
            )
            cot_features[f'{asset}_cot_index'] = net.rolling(52).rank(pct=True)
            cot_features[f'{asset}_cot_change_4w'] = net.diff(4)
            cot_features[f'{asset}_cot_extreme'] = (
                (cot_features[f'{asset}_cot_index'] > 0.90) |
                (cot_features[f'{asset}_cot_index'] < 0.10)
            ).astype(float)

            cot_dfs.append(cot_features)

        if not cot_dfs:
            return pd.DataFrame()

        combined = pd.concat(cot_dfs, axis=1)
        combined.index = pd.to_datetime(combined.index)
        combined.index.name = 'date'
        combined.sort_index(inplace=True)
        return combined

    def get_features(self, engine=None, max_rows: int = 0,
                     macro_df: Optional[pd.DataFrame] = None,
                     cot_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates features using FeatureEngine.
        Returns (aligned_df, normalized_df).
        """
        data_dict = self.load_raw_data(max_rows=max_rows)
        if not data_dict:
            raise FileNotFoundError(
                f"No parquet files found under data_dir={self.data_dir}."
            )

        if engine is None:
            engine = FeatureEngine()

        if macro_df is None:
            macro_df = self.load_macro_data()
        if cot_df is None:
            cot_df = self.load_cot_data()

        aligned_df, normalized_df = engine.preprocess_data(
            data_dict, macro_df=macro_df, cot_df=cot_df
        )
        return aligned_df, normalized_df

if __name__ == "__main__":
    loader = DataLoader()
    aligned_df, normalized_df = loader.get_features()
    print(f"Aligned DF shape: {aligned_df.shape}")
    print(f"Normalized DF shape: {normalized_df.shape}")
    print(f"Columns: {normalized_df.columns[:10]}...")
