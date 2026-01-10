import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import logging
from scipy.signal import argrelextrema

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints", "alpha")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOOKBACK_WINDOW = 500  # How far back to search for structure
SWING_ORDER = 5        # Number of points on each side to define a local extrema

def get_latest_model_files(models_dir):
    if not os.path.exists(models_dir): return None, None
    files = [f for f in os.listdir(models_dir) if f.endswith(".zip")]
    if not files: return None, None
    files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
    latest_zip = files[0]
    return os.path.join(models_dir, latest_zip), os.path.join(models_dir, f"{os.path.splitext(latest_zip)[0]}_vecnormalize.pkl")

def find_swings(df, order=5):
    """
    Identify indices of local highs and lows (fractals).
    """
    # Find local maxima (Swing Highs)
    highs = df['high'].values
    high_idx = argrelextrema(highs, np.greater, order=order)[0]
    
    # Find local minima (Swing Lows)
    lows = df['low'].values
    low_idx = argrelextrema(lows, np.less, order=order)[0]
    
    return high_idx, low_idx

def inspect_structure_rr():
    model_path, vec_norm_path = get_latest_model_files(MODELS_DIR)
    if not model_path:
        logger.error("No model found.")
        return

    logger.info(f"Loading Model: {os.path.basename(model_path)}")
    model = PPO.load(model_path, device='cpu')
    
    norm_env = None
    if os.path.exists(vec_norm_path):
        dummy_venv = DummyVecEnv([lambda: None])
        norm_env = VecNormalize.load(vec_norm_path, dummy_venv)
        norm_env.training = False
        norm_env.norm_reward = False

    test_file = os.path.join(DATA_DIR, "EURUSD_5m.parquet")
    if not os.path.exists(test_file):
        logger.error(f"Data file not found: {test_file}")
        return

    logger.info(f"Loading data: {test_file}")
    df = pd.read_parquet(test_file)
    
    # Setup Feature Engine
    sys.path.append(os.path.join(PROJECT_ROOT, "RiskLayer"))
    from src.feature_engine import FeatureEngine
    
    engine = FeatureEngine()
    
    # We will inspect the last 5000 points
    inspect_len = 10000
    if len(df) > inspect_len:
        df_slice = df.iloc[-inspect_len:].copy()
    else:
        df_slice = df.copy()

    # Preprocess
    data_dict = {'EURUSD': df_slice}
    # Mock other assets
    for asset in ['GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']:
        if asset == 'EURUSD': continue
        p = os.path.join(DATA_DIR, f"{asset}_5m.parquet")
        if os.path.exists(p):
            d = pd.read_parquet(p)
            common = df_slice.index.intersection(d.index)
            data_dict[asset] = d.loc[common]
            data_dict['EURUSD'] = data_dict['EURUSD'].loc[common]
        else:
             data_dict[asset] = pd.DataFrame(index=df_slice.index)

    logger.info("Generating features...")
    _, processed_df = engine.preprocess_data(data_dict)
    
    # Identify Structure (Swing Points) on the raw OHLC data
    # We use the raw 'EURUSD' slice aligned with processed_df
    raw_df = data_dict['EURUSD']
    swing_highs_idx, swing_lows_idx = find_swings(raw_df, order=SWING_ORDER)
    
    # Convert absolute indices to timestamp-based lookups or align with current slice iteration
    # It's easier to work with integer positions relative to the slice.
    
    logger.info("Running inference...")
    
    # Mock Portfolio
    portfolio_state = {
        'equity': 10000.0, 'margin_usage_pct': 0.0, 'drawdown': 0.0, 'num_open_positions': 0,
        'EURUSD': {'has_position': 0, 'position_size': 0, 'unrealized_pnl': 0, 'position_age': 0, 'entry_price': 0, 'current_sl': 0, 'current_tp': 0}
    }
    for a in ['GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']:
        portfolio_state[a] = portfolio_state.get('EURUSD').copy()

    obs_list = []
    
    # Generate observations
    for i in range(len(processed_df)):
        row = processed_df.iloc[i]
        obs = engine.get_observation(row, portfolio_state)
        obs_list.append(obs)
        
    obs_array_full = np.array(obs_list, dtype=np.float32)
    
    # Slice for Single Pair Model (40 features)
    eurusd_feats = obs_array_full[:, 0:25]
    global_feats = obs_array_full[:, 125:140]
    obs_array_sliced = np.concatenate([eurusd_feats, global_feats], axis=1)
    
    if norm_env:
        obs_array_sliced = norm_env.normalize_obs(obs_array_sliced)
        
    actions, _ = model.predict(obs_array_sliced, deterministic=True)
    
    results = []
    
    raw_close = raw_df['close'].values
    raw_high = raw_df['high'].values
    raw_low = raw_df['low'].values
    
    logger.info("Analyzing Structural Risk/Reward...")
    
    for i in range(len(processed_df)):
        # Skip beginning where we can't look back enough
        if i < LOOKBACK_WINDOW: continue
        
        # Action
        action = actions[i]
        act = action.item() if isinstance(action, np.ndarray) else action
        
        signal = 0
        if act > 0.33: signal = 1
        elif act < -0.33: signal = -1
        
        if signal == 0: continue
        
        entry_price = raw_close[i]
        current_idx = i
        
        # FIND STRUCTURE
        # We look at swings strictly BEFORE current_idx
        
        # Filter valid past swings
        past_highs = swing_highs_idx[swing_highs_idx < current_idx]
        past_lows = swing_lows_idx[swing_lows_idx < current_idx]
        
        # Filter by window (optional, but good for relevance)
        # past_highs = past_highs[past_highs > (current_idx - LOOKBACK_WINDOW)]
        # past_lows = past_lows[past_lows > (current_idx - LOOKBACK_WINDOW)]
        
        if len(past_highs) == 0 or len(past_lows) == 0:
            continue
            
        sl_price = None
        tp_price = None
        
        if signal == 1: # BUY
            # SL: Nearest Swing Low BELOW Entry
            valid_sls = raw_low[past_lows]
            valid_sls = valid_sls[valid_sls < entry_price]
            if len(valid_sls) > 0:
                sl_price = valid_sls[-1] # The most recent one
            
            # TP: Nearest Swing High ABOVE Entry
            valid_tps = raw_high[past_highs]
            valid_tps = valid_tps[valid_tps > entry_price]
            if len(valid_tps) > 0:
                tp_price = valid_tps[-1] # The most recent one
                
        else: # SELL
            # SL: Nearest Swing High ABOVE Entry
            valid_sls = raw_high[past_highs]
            valid_sls = valid_sls[valid_sls > entry_price]
            if len(valid_sls) > 0:
                sl_price = valid_sls[-1] # The most recent one
                
            # TP: Nearest Swing Low BELOW Entry
            valid_tps = raw_low[past_lows]
            valid_tps = valid_tps[valid_tps < entry_price]
            if len(valid_tps) > 0:
                tp_price = valid_tps[-1] # The most recent one
        
        # Calculate Ratios
        if sl_price is not None and tp_price is not None:
            risk = abs(entry_price - sl_price)
            reward = abs(tp_price - entry_price)
            
            # Safety for nearly zero risk
            if risk < 0.00005: risk = 0.00005 
            
            rr = reward / risk
            
            results.append({
                'timestamp': raw_df.index[i],
                'signal': 'BUY' if signal == 1 else 'SELL',
                'entry': entry_price,
                'sl': sl_price,
                'tp': tp_price,
                'risk_pips': risk,
                'reward_pips': reward,
                'rr_ratio': rr
            })
            
    # Summary
    res_df = pd.DataFrame(results)
    if res_df.empty:
        logger.info("No trades with valid structure found.")
        return
        
    print("\n--- Structural Risk/Reward Analysis (Nearest BOS) ---")
    print(f"Total Trades Analyzed: {len(res_df)}")
    print(f"Avg RR: {res_df['rr_ratio'].mean():.2f}")
    print(f"Median RR: {res_df['rr_ratio'].median():.2f}")
    
    print("\nRR Distribution:")
    print(res_df['rr_ratio'].describe())
    
    print("\nSurvival Rates (Based on Structure):")
    for cutoff in [0.5, 1.0, 1.5, 2.0, 3.0]:
        count = len(res_df[res_df['rr_ratio'] >= cutoff])
        print(f"RR >= {cutoff}: {count} ({count/len(res_df)*100:.1f}%)")
        
    print("\nSample Trades:")
    print(res_df.head(10)[['timestamp', 'signal', 'entry', 'sl', 'tp', 'rr_ratio']])
    
    res_df.to_csv("structure_rr_results.csv", index=False)
    logger.info("Saved to structure_rr_results.csv")

if __name__ == "__main__":
    inspect_structure_rr()
