import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
import json
import os
from datetime import datetime, timedelta
import traceback
from twisted.internet import reactor, defer
from twisted.internet.defer import inlineCallbacks

# --- cTrader Open API Imports ---
from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
from ctrader_open_api.messages.OpenApiMessages_pb2 import *
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import *

# --- Configuration ---
CT_APP_ID = "17481_ejoPRnjMkFdEkcZTHbjYt5n98n6wRE2wESCkSSHbLIvdWzRkRp"
CT_APP_SECRET = "AaIrnTNyz47CC9t5nsCXU67sCXtKOm7samSkpNFIvqKOaz1vJ1"
CT_ACCOUNT_ID = 44663862  # Must be an integer
CT_ACCESS_TOKEN = "INnzhrurLIS2OSQDgzzckzZr1IbSf10VkS0sDx-cEVU"
CT_HOST_TYPE = "demo"     # "live" or "demo"

# Assets to Fetch (RPD v3.0)
ASSETS = {
    'Crypto': ['BTC', 'ETH', 'SOL'],
    'Forex': ['EUR', 'GBP', 'JPY']
}

# --- Manual Symbol ID Configuration ---
# User must fill these in!
SYMBOL_IDS = {
    'BTC': 1310,   # BITCOIN
    'ETH': 1311,   # ETHEREUM
    'SOL': 1438,   # SOLANA
    'EUR': 1,      # EURUSD
    'GBP': 2,      # GBPUSD
    'JPY': 4       # USDJPY
}

START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2024, 12, 31)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Data Processor (RPD v3.0 Logic) ---
class DataProcessor:
    def __init__(self):
        self.stats = {} 

    def calculate_technical_features(self, df):
        """Generates the 97 features specified in RPD v3.0"""
        df = df.copy()
        
        # Check if enough data for indicators (max window = 50 for EMA)
        if len(df) < 50:
            logging.warning("Not enough data for technical indicators (need > 50 rows). Skipping.")
            return pd.DataFrame()

        # 1. Base 15-min Features
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        
        ema50 = ta.ema(df['close'], length=50)
        df['dist_ema50'] = (df['close'] - ema50) / ema50
        
        df['atr_14_norm'] = ta.atr(df['high'], df['low'], df['close'], length=14) / df['close']
        
        bb = ta.bbands(df['close'], length=20, std=2)
        if bb is not None and 'BBU_20_2.0' in bb.columns:
            df['bb_width'] = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / bb['BBM_20_2.0']
        else:
            df['bb_width'] = 0.0
        
        df['rsi_14_norm'] = ta.rsi(df['close'], length=14) / 100.0
        
        macd = ta.macd(df['close'])
        df['macd_norm'] = macd['MACDh_12_26_9'] / df['close']
        
        vol_sma = ta.sma(df['volume'], length=20)
        df['vol_ratio'] = df['volume'] / (vol_sma + 1e-9)
        
        df['adx_norm'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14'] / 100.0

        # 2. Multi-Timeframe Context
        df_4h = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
        df_4h['rsi_4h'] = ta.rsi(df_4h['close'], length=14) / 100.0
        ema50_4h = ta.ema(df_4h['close'], length=50)
        df_4h['dist_ema50_4h'] = (df_4h['close'] - ema50_4h) / ema50_4h
        df_4h['atr_4h_norm'] = ta.atr(df_4h['high'], df_4h['low'], df_4h['close'], length=14) / df_4h['close']

        df_1d = df.resample('1D').agg({'close': 'last'})
        ema200_1d = ta.ema(df_1d['close'], length=200)
        df_1d['dist_ema200_1d'] = (df_1d['close'] - ema200_1d) / ema200_1d
        df_1d['rsi_1d'] = ta.rsi(df_1d['close'], length=14) / 100.0

        df = df.join(df_4h[['rsi_4h', 'dist_ema50_4h', 'atr_4h_norm']], how='left')
        df = df.join(df_1d[['dist_ema200_1d', 'rsi_1d']], how='left')
        df.fillna(method='ffill', inplace=True)

        # 3. Temporal Features
        df['sin_hour'] = np.sin(df.index.hour * 2 * np.pi / 24)
        df['cos_hour'] = np.cos(df.index.hour * 2 * np.pi / 24)
        df['day_of_week'] = df.index.dayofweek / 6.0

        # 4. Session Status
        df['is_btc_tradeable'] = 1.0
        df['is_eth_tradeable'] = 1.0
        df['is_sol_tradeable'] = 1.0
        
        hours = df.index.hour
        # Simplified session logic (UTC assumed)
        df['is_eur_tradeable'] = np.where(((hours >= 8) & (hours < 21)) & (df['day_of_week'] < 5), 1.0, 0.0)
        df['is_gbp_tradeable'] = np.where(((hours >= 8) & (hours < 21)) & (df['day_of_week'] < 5), 1.0, 0.0)
        df['is_jpy_tradeable'] = np.where(((hours >= 0) & (hours < 16)) & (df['day_of_week'] < 5), 1.0, 0.0)

        return df.dropna()

    def normalize_data(self, df_dict, train_end_date):
        logging.info("Normalizing data (Z-Score on Training Split)...")
        normalized_dfs = {}
        
        # Use the first key in the dictionary to determine columns
        first_asset = next(iter(df_dict))
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'sin_hour', 'cos_hour', 'day_of_week']
        # Also exclude boolean columns dynamically if they start with 'is_'
        exclude_cols += [c for c in df_dict[first_asset].columns if c.startswith('is_')]

        feature_cols = [c for c in df_dict[first_asset].columns if c not in exclude_cols]
        
        for asset, df in df_dict.items():
            train_df = df[df.index <= train_end_date]
            mean = train_df[feature_cols].mean()
            std = train_df[feature_cols].std()
            self.stats[asset] = {'mean': mean, 'std': std}
            
            df_norm = df.copy()
            for col in feature_cols:
                df_norm[col] = (df[col] - mean[col]) / (std[col] + 1e-9)
            normalized_dfs[asset] = df_norm
            
        return normalized_dfs

# --- cTrader Twisted Downloader ---
class CTraderDownloader:
    def __init__(self, assets_to_fetch):
        host = EndPoints.PROTOBUF_LIVE_HOST if CT_HOST_TYPE.lower() == "live" else EndPoints.PROTOBUF_DEMO_HOST
        self.client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)
        self.symbol_map = SYMBOL_IDS # Load from config
        self.downloaded_data = {} # {'BTC': df, ...}
        self.request_delay = 0.25 # Slight increase to be safe
        self.assets_to_fetch = assets_to_fetch
        self.processor = DataProcessor()

    def start(self):
        self.client.setConnectedCallback(self.on_connected)
        self.client.setDisconnectedCallback(self.on_disconnected)
        self.client.setMessageReceivedCallback(self.on_message)
        self.client.startService()
        reactor.run()

    def on_disconnected(self, client, reason):
        logging.info(f"Disconnected: {reason}")

    def on_message(self, client, message):
        pass

    def send_proto_request(self, request):
        return self.client.send(request)

    @inlineCallbacks
    def on_connected(self, client):
        logging.info("Connected to cTrader. Authenticating...")
        
        try:
            # 1. Application Auth
            auth_req = ProtoOAApplicationAuthReq()
            auth_req.clientId = CT_APP_ID
            auth_req.clientSecret = CT_APP_SECRET
            yield self.send_proto_request(auth_req)
            logging.info("App Auth Success.")

            # 2. Account Auth
            acc_auth_req = ProtoOAAccountAuthReq()
            acc_auth_req.ctidTraderAccountId = CT_ACCOUNT_ID
            acc_auth_req.accessToken = CT_ACCESS_TOKEN
            yield self.send_proto_request(acc_auth_req)
            logging.info("Account Auth Success.")
            
            # 3. Fetch Data for All Assets
            logging.info(f"Starting data fetch for: {self.assets_to_fetch}")
            
            for asset in self.assets_to_fetch:
                yield self.fetch_asset_history(asset)
                
                # INCREMENTAL PROCESSING & SAVING
                if asset in self.downloaded_data:
                    try:
                        df = self.downloaded_data[asset]
                        logging.info(f"Processing {asset}...")
                        
                        processed_df = self.processor.calculate_technical_features(df)
                        
                        # 1. Volatility
                        train_df = processed_df[processed_df.index <= datetime(2023, 12, 31)]
                        # Check if log_ret exists
                        if 'log_ret' in train_df.columns:
                            vol = train_df['log_ret'].std()
                        else:
                            vol = 0.01
                            
                        if pd.isna(vol) or vol == 0: vol = 0.01
                        
                        with open(f"volatility_{asset}.json", "w") as f:
                            json.dump({asset: float(vol)}, f)
                        logging.info(f"Saved volatility_{asset}.json")
                        
                        # 2. Normalize
                        norm_dict = self.processor.normalize_data({asset: processed_df}, datetime(2023, 12, 31))
                        final_df = norm_dict[asset]
                        
                        # 3. Save Parquet
                        fname = f"data_{asset}_final.parquet"
                        final_df.to_parquet(fname)
                        logging.info(f"✅ Saved {fname}")
                        
                        # Cleanup to save RAM
                        del self.downloaded_data[asset]
                        
                    except Exception as e:
                        logging.error(f"Error processing {asset}: {e}")
                
            logging.info("All downloads complete. Stopping reactor.")
            reactor.stop()

        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            reactor.stop()

    @inlineCallbacks
    def fetch_asset_history(self, asset):
        if asset not in self.symbol_map:
            logging.warning(f"❌ Skipping {asset}: Could not find Symbol ID in server response.")
            return

        # RETRIEVE THE INTEGER ID
        symbol_id = self.symbol_map[asset]
        logging.info(f"📥 Starting fetch for {asset} using Symbol ID: {symbol_id}")
        
        current_start = START_DATE
        all_bars = []
        
        while current_start < END_DATE:
            # Clamp end time to avoid requesting future data
            chunk_end = current_start + timedelta(days=20)
            if chunk_end > END_DATE:
                chunk_end = END_DATE
            
            # Construct Request using the ID
            req = ProtoOAGetTrendbarsReq()
            req.ctidTraderAccountId = CT_ACCOUNT_ID
            req.symbolId = int(symbol_id) # Explicit int cast for safety
            req.period = ProtoOATrendbarPeriod.M15
            req.fromTimestamp = int(current_start.timestamp() * 1000)
            req.toTimestamp = int(chunk_end.timestamp() * 1000)
            
            try:
                # Rate limit delay (CORRECTED)
                d = defer.Deferred()
                reactor.callLater(self.request_delay, d.callback, None)
                yield d
                
                res_msg = yield self.send_proto_request(req)
                payload = Protobuf.extract(res_msg)
                
                if not hasattr(payload, 'trendbar') or not payload.trendbar:
                    # logging.info(f"   {asset} (ID {symbol_id}): No data between {current_start} and {chunk_end}")
                    current_start = chunk_end
                    continue
                
                bars_data = []
                # Process payload.trendbar (list of ProtoOATrendbar)
                for bar in payload.trendbar:
                    # cTrader Price = value / 100,000 (standard for most pairs)
                    DIVISOR = 100000.0
                    
                    low = bar.low / DIVISOR
                    open_p = (bar.low + bar.deltaOpen) / DIVISOR
                    high = (bar.low + bar.deltaHigh) / DIVISOR
                    close = (bar.low + bar.deltaClose) / DIVISOR
                    
                    bars_data.append({
                        'open': open_p, 
                        'high': high, 
                        'low': low, 
                        'close': close, 
                        'volume': bar.volume
                    })
                
                if not bars_data:
                    current_start = chunk_end
                    continue

                df_chunk = pd.DataFrame(bars_data)
                
                # Create time index based on request range (Approximation)
                time_index = pd.date_range(start=current_start, periods=len(df_chunk), freq='15min')
                df_chunk.index = time_index
                
                all_bars.append(df_chunk)
                
                # Advance time
                current_start = chunk_end
                logging.info(f"   {asset}: Fetched {len(df_chunk)} bars. Next start: {current_start}")

            except Exception as e:
                logging.error(f"Error fetching chunk for {asset}: {e}")
                # Wait a bit before retrying or skipping
                d_err = defer.Deferred()
                reactor.callLater(1.0, d_err.callback, None)
                yield d_err
                # Move on to avoid infinite loop on error
                current_start = chunk_end
        
        if all_bars:
            full_df = pd.concat(all_bars)
            full_df = full_df[~full_df.index.duplicated(keep='first')]
            self.downloaded_data[asset] = full_df
            logging.info(f"✅ Completed Download {asset}: {len(full_df)} total rows.")
        else:
            logging.warning(f"⚠️ No data fetched for {asset}!")

# --- Main Entry Point ---
if __name__ == "__main__":
    print("DEBUG: Script started")
    all_assets = ASSETS['Crypto'] + ASSETS['Forex']
    assets_to_fetch = []
    
    # 1. Check for existing data
    for asset in all_assets:
        fname = f"data_{asset}_final.parquet"
        if os.path.exists(fname):
            logging.info(f"Found existing data for {asset}. Skipping.")
        else:
            assets_to_fetch.append(asset)
            
    # 2. Fetch missing data
    if assets_to_fetch:
        logging.info(f"Need to fetch: {assets_to_fetch}")
        downloader = CTraderDownloader(assets_to_fetch)
        print("Starting cTrader Downloader... (Press Ctrl+C to stop manually)")
        try:
            downloader.start()
        except KeyboardInterrupt:
            logging.info("Interrupted.")
    else:
        logging.info("All assets already downloaded.")

    # 4. Aggregate Volatility Baselines
    final_vol_map = {}
    for asset in all_assets:
        vol_file = f"volatility_{asset}.json"
        if os.path.exists(vol_file):
            with open(vol_file, "r") as f:
                data = json.load(f)
                final_vol_map.update(data)
        else:
            logging.warning(f"Missing volatility file for {asset}. Defaulting to 0.01")
            final_vol_map[asset] = 0.01
            
    with open("volatility_baseline.json", "w") as f:
        json.dump(final_vol_map, f, indent=4)
    logging.info("Saved aggregated volatility_baseline.json")
    
    print("Done.")