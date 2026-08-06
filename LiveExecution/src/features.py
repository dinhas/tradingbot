import pandas as pd
import numpy as np
import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
from twisted.internet.task import LoopingCall
from Alpha.src.feature_engine import FeatureEngine as AlphaFeatureEngine
from Alpha.src.data_loader import DataLoader
from shared_constants import FX_ALPHA_ASSETS

# Add Filter/src to path for feature engine
import sys
from pathlib import Path
_filter_src = Path(__file__).resolve().parent.parent.parent / "Filter" / "src"
if str(_filter_src) not in sys.path:
    sys.path.insert(0, str(_filter_src))

from Filter.src.feature_engine import FeatureEngine as FilterFeatureEngine


class FeatureManager:
    """
    Coordinates feature calculation for Alpha and Filter models.
    Handles macro/COT data with daily refresh at midnight.
    """
    def __init__(self, data_dir='data'):
        self.logger = logging.getLogger("LiveExecution")
        self.assets = FX_ALPHA_ASSETS
        self.alpha_fe = AlphaFeatureEngine()
        self.filter_fe = FilterFeatureEngine()

        # Pre-load macro and COT data once at startup
        self._data_dir = data_dir
        self.macro_df = pd.DataFrame()
        self.cot_df = pd.DataFrame()
        self._load_macro_cot_data()

        # History buffers for each asset
        self.history = {asset: pd.DataFrame() for asset in self.assets}
        self.max_history = 300

        # Daily macro/COT refresh scheduler
        self._refresh_loop = None

    def initialize_market_data(self):
        """
        Initializes market data by validating and syncing cache, then reloading it.
        Runs off the main reactor thread to avoid blocking.
        """
        self.logger.info("MarketDataCache: Starting cache sync and data warm-up...")
        from LiveExecution.src.market_data_cache import MarketDataCacheManager
        manager = MarketDataCacheManager(data_dir=self._data_dir)
        manager.sync()
        self.logger.info("MarketDataCache: Sync complete. Reloading cache into memory...")
        self._load_macro_cot_data()

    def _load_macro_cot_data(self):
        """Load macro and COT data from disk."""
        try:
            _loader = DataLoader(data_dir=self._data_dir)
            self.macro_df = _loader.load_macro_data()
            self.cot_df = _loader.load_cot_data()
            self.logger.info(
                f"Macro data loaded: {len(self.macro_df)} rows, "
                f"COT data loaded: {len(self.cot_df)} rows"
            )
        except Exception:
            self.logger.warning("Could not load macro/COT data. Features will be zero.")
            self.macro_df = pd.DataFrame()
            self.cot_df = pd.DataFrame()

    def start_daily_refresh(self):
        """Start the daily macro/COT/yfinance data refresh at 00:00:00."""
        if self._refresh_loop is not None:
            return

        def _do_refresh():
            self.logger.info("Daily refresh: Triggering off-thread market data sync...")
            from twisted.internet import threads
            d = threads.deferToThread(self.initialize_market_data)
            d.addCallback(lambda _: self.logger.info("Daily refresh complete."))
            d.addErrback(lambda f: self.logger.error(f"Daily refresh failed: {f.getErrorMessage()}"))

        def _schedule_next():
            now = datetime.now()
            tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            delay = (tomorrow - now).total_seconds()
            self.logger.info(f"Next market data refresh in {delay:.0f}s (at {tomorrow})")
            from twisted.internet import reactor as _reactor
            _reactor.callLater(delay, _do_refresh_and_reschedule)

        def _do_refresh_and_reschedule():
            _do_refresh()
            _schedule_next()

        _schedule_next()
        self.logger.info("Daily market data refresh scheduler started (00:00:00).")

    def stop_daily_refresh(self):
        """Stop the daily refresh scheduler."""
        if self._refresh_loop is not None:
            self._refresh_loop.stop()
            self._refresh_loop = None

    def push_candle(self, asset, candle_data):
        """
        Pushes a new candle to the history buffer.
        If the timestamp exists, updates the row. If not, appends.
        """
        ts = candle_data.pop('timestamp')
        ts = pd.Timestamp(ts)
        ts = ts.floor('min')

        new_row = pd.Series(candle_data, name=ts)

        if ts in self.history[asset].index:
            self.history[asset].loc[ts] = new_row
        else:
            self.history[asset] = pd.concat([self.history[asset], pd.DataFrame([new_row])])
            self.history[asset] = self.history[asset].sort_index()

        if len(self.history[asset]) > self.max_history:
            self.history[asset] = self.history[asset].iloc[-self.max_history:]

        self.logger.debug(f"Pushed candle for {asset}. Buffer size: {len(self.history[asset])}")

    def update_data(self, symbol_id, ohlcv_res):
        """
        Updates internal history from cTrader trendbars response.
        """
        from datetime import datetime as dt

        asset = self._get_asset_name_from_id(symbol_id)
        if not asset:
            self.logger.error(f"Unknown symbol_id: {symbol_id}")
            return

        new_rows = []
        for bar in ohlcv_res.trendbar:
            divisor = 100000.0
            low = bar.low / divisor
            new_rows.append({
                'timestamp': dt.fromtimestamp(bar.utcTimestampInMinutes * 60),
                'open': low + (bar.deltaOpen / divisor),
                'high': low + (bar.deltaHigh / divisor),
                'low': low,
                'close': low + (bar.deltaClose / divisor),
                'volume': bar.volume
            })

        if new_rows:
            new_rows.sort(key=lambda x: x['timestamp'])
            for row in new_rows:
                self.push_candle(asset, row)

    def update_from_trendbar(self, asset, bar):
        """
        Updates history from a single Protobuf Trendbar object.
        """
        from datetime import datetime as dt

        divisor = 100000.0
        low = bar.low / divisor
        row = {
            'timestamp': dt.fromtimestamp(bar.utcTimestampInMinutes * 60),
            'open': low + (bar.deltaOpen / divisor),
            'high': low + (bar.deltaHigh / divisor),
            'low': low,
            'close': low + (bar.deltaClose / divisor),
            'volume': bar.volume
        }
        self.push_candle(asset, row)

    def _get_asset_name_from_id(self, symbol_id):
        if hasattr(self, 'client') and self.client:
            if hasattr(self.client, 'broker_symbol_map') and symbol_id in self.client.broker_symbol_map:
                raw_name = self.client.broker_symbol_map[symbol_id]
                for asset in ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']:
                    if asset in raw_name:
                        return asset
                return raw_name
            if hasattr(self.client, 'symbol_ids'):
                inv_map = {v: k for k, v in self.client.symbol_ids.items()}
                if symbol_id in inv_map:
                    raw_name = inv_map[symbol_id]
                    for asset in ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']:
                        if asset in raw_name:
                            return asset
                    return raw_name

        # Fallback to hardcoded mapping
        mapping = {1: 'EURUSD', 2: 'GBPUSD', 6: 'USDCHF', 4: 'USDJPY'}
        return mapping.get(symbol_id)

    def get_atr(self, asset):
        """Calculates the current 14-period ATR for the given asset."""
        if asset not in self.history or self.history[asset].empty or len(self.history[asset]) < 15:
            return 0.0

        from ta.volatility import AverageTrueRange
        df = self.history[asset]
        atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        return float(atr.iloc[-1])

    def get_alpha_sequence(self, asset, sequence_length):
        """Builds the ordered 5m feature sequence expected by the Alpha LSTM."""
        # Always include all 4 assets — feature engine processes self.assets (all 4)
        data_dict = {}
        for a in self.assets:
            if not self.history[a].empty:
                data_dict[a] = self.history[a]
            else:
                # Empty placeholder so _align_data can still build the full column set
                data_dict[a] = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        _, normalized_df = self.alpha_fe.preprocess_data(data_dict, macro_df=self.macro_df, cot_df=self.cot_df)
        obs = self.alpha_fe.get_observation_vectorized(normalized_df, asset)
        if len(obs) < sequence_length:
            return None
        return obs[-sequence_length:].astype(np.float32)

    def get_filter_features(self):
        """Build the 26-feature vector for the RF filter from current OHLCV data.

        Returns numpy array of shape (n_rows, 26) from FilterFeatureEngine.
        """
        # Same approach as get_alpha_sequence: all assets must be present
        data_dict = {}
        for a in self.assets:
            if not self.history[a].empty:
                data_dict[a] = self.history[a]
            else:
                data_dict[a] = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        if not any(not df.empty for df in data_dict.values()):
            return None

        try:
            _, normalized_df = self.filter_fe.preprocess_data(data_dict)
            asset = "EURUSD"
            obs = self.filter_fe.get_observation_vectorized(normalized_df, asset)
            return obs
        except Exception as e:
            self.logger.error(f"Filter feature build error: {e}")
            return None

    def is_ready(self):
        """Checks if enough history is collected for all assets."""
        min_required = 200  # For MA200
        for asset in self.assets:
            if len(self.history[asset]) < min_required:
                return False
        return True
