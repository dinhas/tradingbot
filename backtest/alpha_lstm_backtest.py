import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Alpha.src.model import AlphaSLModel
from Alpha.src.data_loader import DataLoader as AlphaDataLoader
from RiskLayer.src.feature_engine import FeatureEngine
from backtest.rl_backtest import BacktestMetrics, generate_all_charts, NumpyEncoder
from shared_constants import FX_ALPHA_ASSETS, DEFAULT_SPREADS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _infer_lstm_input_dim(state_dict: dict) -> int:
    """Infer the expected input dimension from an LSTM checkpoint."""
    key = "lstm.weight_ih_l0"
    if key not in state_dict:
        raise ValueError(f"Checkpoint is missing {key}; cannot infer input dimension.")
    return int(state_dict[key].shape[1])


def _load_state_dict(model_path: Path) -> dict:
    """Load a PyTorch checkpoint from .pth or .zip serialization."""
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format in {model_path}")
    return checkpoint

class AlphaLSTMVectorizedBacktester:
    """
    Ultra-fast vectorized backtester for Alpha LSTM models.
    Achieves 100x+ speed boost by batching model inference and using NumPy for trade logic.
    """
    def __init__(
        self,
        model: AlphaSLModel,
        aligned_df: pd.DataFrame,
        normalized_df: pd.DataFrame,
        sequence_length: int,
        confidence_thresh: float,
        initial_equity: float,
        position_size_pct: float,
        sl_mult: float,
        tp_mult: float,
        adx_thresh: float = 25.0,
        max_hold_bars: int = 6,
        leverage: float = 100.0,
        batch_size: int = 1024
    ):
        self.model = model.to(DEVICE)
        self.aligned_df = aligned_df
        self.normalized_df = normalized_df
        self.sequence_length = sequence_length
        self.confidence_thresh = confidence_thresh
        self.initial_equity = initial_equity
        self.position_size_pct = position_size_pct
        self.sl_mult = sl_mult
        self.tp_mult = tp_mult
        self.adx_thresh = adx_thresh
        self.max_hold_bars = max_hold_bars
        self.leverage = leverage
        self.batch_size = batch_size
        
        self.assets = FX_ALPHA_ASSETS
        self.spreads = DEFAULT_SPREADS
        self.feature_engine = FeatureEngine()
        
    def _precompute_predictions(self):
        """Precomputes model predictions for all assets in batches."""
        logger.info(f"Precomputing predictions for {len(self.assets)} assets...")
        all_probs = {}
        
        n_steps = len(self.normalized_df)
        
        for asset in self.assets:
            logger.info(f"Predicting for {asset}...")
            obs = self.feature_engine.get_observation_vectorized(self.normalized_df, asset)
            
            # Use sliding_window_view for zero-copy sequence creation (NumPy 1.20+)
            # Shape: (n_sequences, sequence_length, n_features)
            sequences = np.lib.stride_tricks.sliding_window_view(
                obs, (self.sequence_length, obs.shape[1])
            ).squeeze(1)
            
            asset_probs = np.zeros((n_steps, 3), dtype=np.float32)
            
            self.model.eval()
            with torch.no_grad():
                for i in range(0, len(sequences), self.batch_size):
                    batch_seq = sequences[i:i+self.batch_size]
                    batch_tensor = torch.from_numpy(batch_seq.copy()).to(DEVICE)
                    logits = self.model(batch_tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    
                    # Align probabilities back to original indices
                    # sequences[i] corresponds to normalized_df.index[i + sequence_length - 1]
                    start_idx = i + self.sequence_length - 1
                    end_idx = start_idx + len(probs)
                    asset_probs[start_idx:end_idx] = probs
            
            all_probs[asset] = asset_probs
            
        return all_probs

    def run(self, max_steps: int | None = None) -> BacktestMetrics:
        metrics = BacktestMetrics()
        
        # 1. Vectorized Inference
        all_probs = self._precompute_predictions()
        
        # 2. Sequential Trade Logic (Still sequential in time, but extremely fast due to precomputed probs)
        n_steps = len(self.normalized_df)
        if max_steps:
            n_steps = min(n_steps, max_steps)
            
        equity = self.initial_equity
        positions = {asset: None for asset in self.assets}
        
        # Extract necessary arrays for speed
        close_prices = {asset: self.aligned_df[f"{asset}_close"].values for asset in self.assets}
        high_prices = {asset: self.aligned_df[f"{asset}_high"].values for asset in self.assets}
        low_prices = {asset: self.aligned_df[f"{asset}_low"].values for asset in self.assets}
        atrs = {asset: self.aligned_df[f"{asset}_atr"].values for asset in self.assets}
        adxs = {asset: self.aligned_df[f"{asset}_adx"].values for asset in self.assets}
        timestamps = self.normalized_df.index
        
        logger.info(f"Starting vectorized backtest execution (ADX Threshold: {self.adx_thresh})...")
        metrics.add_equity_point(timestamps[self.sequence_length - 1], float(equity))
        
        # We start from sequence_length - 1
        for idx in tqdm(range(self.sequence_length - 1, n_steps - 1)):
            ts = timestamps[idx]
            
            for asset in self.assets:
                current_adx = adxs[asset][idx]
                
                # ADX Filter: Only consider model decisions if ADX is above threshold
                can_act = current_adx >= self.adx_thresh

                # Prediction logic
                probs = all_probs[asset][idx]
                max_prob = np.max(probs)
                
                direction = 0
                if can_act and max_prob >= self.confidence_thresh:
                    direction = int(np.argmax(probs) - 1)
                
                current_pos = positions[asset]
                mid_close = close_prices[asset][idx]
                atr = atrs[asset][idx]
                half_spread = self.spreads.get(asset, 0.0) / 2.0
                
                # Close Logic / Management
                if current_pos is not None:
                    # SL/TP Evaluation on the current candle (since we entered at idx-1 or earlier)
                    # Spread-aware: longs exit at Bid, shorts exit at Ask
                    exit_price = None
                    reason = ""
                    
                    p = current_pos
                    if p['direction'] == 1: # Long -> exits at Bid
                        bid_low = low_prices[asset][idx] - half_spread
                        bid_high = high_prices[asset][idx] - half_spread
                        if bid_low <= p['sl']:
                            exit_price = p['sl']
                            reason = "SL"
                        elif bid_high >= p['tp']:
                            exit_price = p['tp']
                            reason = "TP"
                    else: # Short -> exits at Ask
                        ask_high = high_prices[asset][idx] + half_spread
                        ask_low = low_prices[asset][idx] + half_spread
                        if ask_high >= p['sl']:
                            exit_price = p['sl']
                            reason = "SL"
                        elif ask_low <= p['tp']:
                            exit_price = p['tp']
                            reason = "TP"

                    # Vertical barrier: mirrors the 6-bar timeout used in labeling
                    if exit_price is None and (idx - p['entry_idx']) >= self.max_hold_bars:
                        exit_price = mid_close - (p['direction'] * half_spread)
                        reason = "Timeout"

                    # Model Flip Signal (Only if ADX allows acting)
                    if exit_price is None and can_act and (direction == 0 or direction != p['direction']):
                        exit_price = mid_close - (p['direction'] * half_spread)
                        reason = "Signal Flip"
                        
                    if exit_price is not None:
                        # Process Close
                        price_change_pct = (exit_price - p['entry_price']) / p['entry_price'] * p['direction']
                        pnl = price_change_pct * (p['size'] * self.leverage)

                        fee = p['entry_fee'] + (p['size'] * 0.00002)
                        net_pnl = pnl - fee
                        equity += pnl - (p['size'] * 0.00002)
                        
                        metrics.add_trade({
                            'timestamp': ts,
                            'asset': asset,
                            'pnl': float(pnl),
                            'fees': float(fee),
                            'net_pnl': float(net_pnl),
                            'entry_price': float(p['entry_price']),
                            'exit_price': float(exit_price),
                            'size': float(p['size']),
                            'equity_before': float(p['equity_before']),
                            'equity_after': float(equity),
                            'hold_time': (ts - p['entry_timestamp']).total_seconds() / 60.0,
                            'reason': reason
                        })
                        positions[asset] = None
                        current_pos = None

                # Open Logic (Only if ADX allows acting)
                if positions[asset] is None and can_act and direction != 0:
                    # Open position (spread-aware: buy at Ask, sell at Bid)
                    size = self.position_size_pct * equity
                    entry_price = mid_close + (direction * half_spread)
                    sl_dist = self.sl_mult * atr
                    tp_dist = self.tp_mult * atr
                    
                    sl = entry_price - (direction * sl_dist)
                    tp = entry_price + (direction * tp_dist)
                    
                    entry_fee = size * 0.00002
                    positions[asset] = {
                        'direction': direction,
                        'entry_price': entry_price,
                        'size': size,
                        'entry_fee': entry_fee,
                        'sl': sl,
                        'tp': tp,
                        'entry_timestamp': ts,
                        'entry_idx': idx,
                        'equity_before': equity
                    }
                    equity -= entry_fee

            metrics.add_equity_point(ts, float(equity))


        # Force close at end
        final_idx = n_steps - 1
        for asset in self.assets:
            p = positions[asset]
            if p:
                half_spread = self.spreads.get(asset, 0.0) / 2.0
                exit_price = close_prices[asset][final_idx] - (p['direction'] * half_spread)
                price_change_pct = (exit_price - p['entry_price']) / p['entry_price'] * p['direction']
                pnl = price_change_pct * (p['size'] * self.leverage)
                fee = p['entry_fee'] + (p['size'] * 0.00002)
                net_pnl = pnl - fee
                equity += pnl - (p['size'] * 0.00002)
                metrics.add_trade({
                    'timestamp': timestamps[final_idx],
                    'asset': asset,
                    'pnl': float(pnl),
                    'fees': float(fee),
                    'net_pnl': float(net_pnl),
                    'entry_price': float(p['entry_price']),
                    'exit_price': float(exit_price),
                    'size': float(p['size']),
                    'equity_before': float(p['equity_before']),
                    'equity_after': float(equity),
                    'hold_time': (timestamps[final_idx] - p['entry_timestamp']).total_seconds() / 60.0,
                    'reason': "End of Backtest"
                })
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description="Ultra-Fast Vectorized Alpha LSTM Backtester")
    parser.add_argument("--model-path", type=str, default="Alpha/models/alpha_model.zip")
    parser.add_argument("--data-dir", type=str, default="backtest/data")
    parser.add_argument("--output-dir", type=str, default="backtest/results")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--confidence-thresh", type=float, default=0.45)
    parser.add_argument("--initial-equity", type=float, default=10000.0)
    parser.add_argument("--pos-size", type=float, default=0.1, help="Position size as % of equity (0.1 = 10%)")
    # Defaults now MIRROR the Labeler geometry (tp=1.0 ATR, sl=0.5 ATR, 6-bar timeout, ADX 25)
    parser.add_argument("--sl-mult", type=float, default=0.5)
    parser.add_argument("--tp-mult", type=float, default=1.0)
    parser.add_argument("--adx-thresh", type=float, default=25.0)
    parser.add_argument("--max-hold-bars", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--no-charts", action="store_true", help="Skip chart generation after saving metrics/trades")
    args = parser.parse_args()

    model_path = PROJECT_ROOT / args.model_path
    data_dir = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the RiskLayer FeatureEngine to ensure causal (no-lookahead) features
    rk_engine = FeatureEngine()
    loader = AlphaDataLoader(data_dir=str(data_dir))
    aligned_df, normalized_df = loader.get_features(engine=rk_engine)

    # Model parameters (derived from the feature engine so train/backtest never drift)
    input_dim = len(rk_engine.feature_names)
    model = AlphaSLModel(input_dim=input_dim, lstm_units=64, dense_units=32, dropout=0.3)
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return

    checkpoint = _load_state_dict(model_path)
    checkpoint_input_dim = _infer_lstm_input_dim(checkpoint)
    if checkpoint_input_dim != input_dim:
        logger.error(
            "Model/input mismatch: checkpoint expects %d features but backtest engine builds %d.",
            checkpoint_input_dim,
            input_dim,
        )
        return
    logger.info("Model/input feature check passed: checkpoint=%d, backtest_engine=%d", checkpoint_input_dim, input_dim)

    model.load_state_dict(checkpoint)
    model.eval()

    bt = AlphaLSTMVectorizedBacktester(
        model=model,
        aligned_df=aligned_df,
        normalized_df=normalized_df,
        sequence_length=50,
        confidence_thresh=args.confidence_thresh,
        initial_equity=args.initial_equity,
        position_size_pct=args.pos_size,
        sl_mult=args.sl_mult,
        tp_mult=args.tp_mult,
        adx_thresh=args.adx_thresh,
        max_hold_bars=args.max_hold_bars,
        batch_size=args.batch_size
    )

    start_time = datetime.now()
    metrics = bt.run(max_steps=args.steps)
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    logger.info(f"Backtest completed in {duration:.2f} seconds.")

    results = metrics.calculate_metrics()
    per_asset = metrics.get_per_asset_metrics()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results
    metrics_file = output_dir / f"metrics_alpha_lstm_vectorized_{timestamp}.json"
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    if metrics.trades:
        trades_file = output_dir / f"trades_alpha_lstm_vectorized_{timestamp}.csv"
        pd.DataFrame(metrics.trades).to_csv(trades_file, index=False)

    if per_asset:
        asset_file = output_dir / f"asset_breakdown_alpha_lstm_vectorized_{timestamp}.csv"
        pd.DataFrame(per_asset).T.to_csv(asset_file)

    if metrics.equity_curve and metrics.trades and not args.no_charts:
        generate_all_charts(metrics, per_asset, "AlphaLSTM_Vectorized", output_dir, timestamp)

    logger.info("\n=== RESULTS ===")
    for k, v in results.items():
        if isinstance(v, float):
            logger.info(f"{k:<25}: {v:.4f}")
        else:
            logger.info(f"{k:<25}: {v}")

if __name__ == "__main__":
    main()
