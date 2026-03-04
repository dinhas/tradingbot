import os
import sys
import numpy as np
import pandas as pd
import torch
from collections import deque
from datetime import timedelta
from tqdm import tqdm
import logging
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RiskLayer.src.risk_model_rl import RiskModelRL, SL_CHOICES, NUM_SL_CHOICES
from Alpha.src.feature_engine import FeatureEngine as AlphaFeatureEngine
from RiskLayer.src.frozen_alpha_env import TradingEnv

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ASSET_BASE_SPREADS = {
    "EURUSD": 1.5,
    "GBPUSD": 1.8,
    "USDJPY": 2.0,
    "USDCHF": 1.5,
    "XAUUSD": 20.0,
}

SESSION_MULTIPLIERS = {"asian": 1.5, "london": 1.0, "ny": 1.0, "overlap": 0.8}


def get_dynamic_spread(asset, hour, atr, atr_ma):
    base_spread = ASSET_BASE_SPREADS.get(asset, 1.5)

    if 0 <= hour < 7:
        session = "asian"
    elif 7 <= hour < 12:
        session = "overlap"
    elif 12 <= hour < 17:
        session = "ny"
    else:
        session = "london"

    session_mult = SESSION_MULTIPLIERS.get(session, 1.0)
    vol_mult = 1.3 if atr > atr_ma * 1.2 else 1.0

    spread = base_spread * session_mult * vol_mult

    pip_value = 0.0001 if "JPY" not in asset else 0.01
    return spread * pip_value


class BacktestMetrics:
    def __init__(self):
        self.trades = []
        self.equity_curve = []

    def add_trade(self, trade):
        self.trades.append(trade)

    def add_equity_point(self, timestamp, equity):
        self.equity_curve.append({"timestamp": timestamp, "equity": equity})

    def get_summary(self):
        if not self.trades:
            return {"num_trades": 0}

        pnls = [t["net_pnl"] for t in self.trades]
        return {
            "num_trades": len(self.trades),
            "total_pnl": sum(pnls),
            "avg_pnl": np.mean(pnls),
            "win_rate": sum(1 for p in pnls if p > 0) / len(pnls),
            "max_drawdown": min(pnls) if pnls else 0,
        }


class RLBacktester:
    def __init__(self, risk_model_path, scaler_path, data_dir, initial_equity=10000.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading RL Risk Model from {risk_model_path}...")
        checkpoint = torch.load(risk_model_path, map_location=self.device)
        self.model = RiskModelRL(input_dim=40)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        logger.info(f"Loading Scaler from {scaler_path}...")
        self.scaler = joblib.load(scaler_path)

        logger.info("Initializing environment...")
        self.env = TradingEnv(data_dir=data_dir, stage=3, is_training=False)
        self.env.feature_engine = AlphaFeatureEngine()
        self.env.raw_data, self.env.processed_data = (
            self.env.feature_engine.preprocess_data(self.env.data)
        )
        self.env._cache_data_arrays()

        self.env.equity = initial_equity
        self.env.start_equity = initial_equity
        self.env.peak_equity = initial_equity

        self.initial_equity = initial_equity
        self.equity = initial_equity

        self.assets = self.env.assets
        self._precompute_risk_predictions()

    def _precompute_risk_predictions(self):
        logger.info("Precomputing RL Risk predictions...")

        df = self.env.processed_data
        N = len(df)
        num_assets = len(self.assets)

        all_obs = []
        for i in range(N):
            row_obs = []
            for asset in self.assets:
                obs = self.env.feature_engine.get_observation_vectorized(
                    df.iloc[i : i + 1], asset
                )
                row_obs.append(obs[0])
            all_obs.append(row_obs)

        obs_flat = np.array([obs for row in all_obs for obs in row])

        batch_size = 1024
        sl_idx_list, tp_list, size_list = [], [], []

        for i in tqdm(range(0, len(obs_flat), batch_size), desc="Risk Inference"):
            batch = obs_flat[i : i + batch_size]
            batch_scaled = self.scaler.transform(batch).astype(np.float32)
            batch_tensor = torch.from_numpy(batch_scaled).to(self.device)

            with torch.no_grad():
                preds = self.model(batch_tensor)
                sl_idx = torch.argmax(preds["sl_probs"], dim=-1).cpu().numpy()
                tp = preds["tp"].cpu().numpy().flatten()
                size = preds["size"].cpu().numpy().flatten()

            sl_idx_list.append(sl_idx)
            tp_list.append(tp)
            size_list.append(size)

        sl_idx_all = np.concatenate(sl_idx_list)
        tp_all = np.concatenate(tp_list)
        size_all = np.concatenate(size_list)

        self.sl_idx_matrix = sl_idx_all.reshape(N, num_assets)
        self.tp_matrix = tp_all.reshape(N, num_assets)
        self.size_matrix = size_all.reshape(N, num_assets)

        logger.info("Risk predictions precomputed.")

    def _get_spread(self, asset, idx):
        hour = self.env.raw_data[asset].index[idx].hour
        atr = self.env.atr_arrays[asset][idx]

        atr_ma = np.mean(self.env.atr_arrays[asset][max(0, idx - 20) : idx])
        if atr_ma == 0:
            atr_ma = 1e-10

        return get_dynamic_spread(asset, hour, atr, atr_ma)

    def run_backtest(self, start_idx=500, end_idx=None, size_threshold=0.30):
        if end_idx is None:
            end_idx = self.env.max_steps

        logger.info(f"Running backtest from {start_idx} to {end_idx}")

        metrics = BacktestMetrics()

        for current_idx in tqdm(range(start_idx, end_idx), desc="Backtest"):
            self.env.current_step = current_idx

            combined_actions = {}
            open_pos_count = 0

            for i, asset in enumerate(self.assets):
                sl_idx = self.sl_idx_matrix[current_idx, i]
                tp_mult = self.tp_matrix[current_idx, i]
                size_out = self.size_matrix[current_idx, i]

                if size_out < size_threshold:
                    continue

                sl_mult = SL_CHOICES[sl_idx]

                entry_price = self.env.close_arrays[asset][current_idx]
                spread = self._get_spread(asset, current_idx)

                position_size = self.equity * size_out * 0.6
                leverage = 100

                is_usd_quote = asset in ["EURUSD", "GBPUSD", "XAUUSD"]
                contract_size = 100000

                if is_usd_quote:
                    lot_size = position_size * contract_size / self.equity
                else:
                    quote_price = entry_price
                    lot_size = position_size * contract_size / quote_price / self.equity

                lot_size = min(lot_size, leverage)

                if lot_size > 0.01:
                    combined_actions[asset] = {
                        "direction": 1,
                        "size": size_out,
                        "sl_mult": sl_mult,
                        "tp_mult": tp_mult,
                        "lots": lot_size,
                        "spread": spread,
                    }

            for asset, act in combined_actions.items():
                current_pos = self.env.positions[asset]
                price_raw = self.env.close_arrays[asset][current_idx]
                spread = act["spread"]

                entry_price_buy = price_raw + spread
                entry_price_sell = price_raw - spread

                direction = act["direction"]
                price = entry_price_buy if direction == 1 else entry_price_sell

                atr = self.env.atr_arrays[asset][current_idx]

                if current_pos is None:
                    self._open_position(asset, direction, act, price, atr)
                elif current_pos["direction"] != direction:
                    self._close_position(asset, price, spread)
                    self._open_position(asset, direction, act, price, atr)

            self.env.current_step += 1
            if self.env.current_step >= self.env.max_steps:
                break

            self._update_positions()

            self.equity = self.env.equity

            if self.env.completed_trades:
                for trade in self.env.completed_trades:
                    metrics.add_trade(trade)

        summary = metrics.get_summary()
        logger.info(f"Backtest complete: {summary}")
        return metrics

    def _open_position(self, asset, direction, act, price, atr):
        size = act["size"] * 0.6 * self.equity
        atr = max(atr, price * 0.0001)
        sl_dist = act["sl_mult"] * atr
        tp_dist = act["tp_mult"] * atr

        if direction == 1:
            sl = price - sl_dist
            tp = price + tp_dist
        else:
            sl = price + sl_dist
            tp = price - tp_dist

        self.env.positions[asset] = {
            "direction": direction,
            "entry_price": price,
            "size": size,
            "sl": sl,
            "tp": tp,
            "entry_step": self.env.current_step,
            "sl_dist": sl_dist,
            "tp_dist": tp_dist,
        }

        self.equity -= size * 0.00002

    def _close_position(self, asset, price, spread):
        pos = self.env.positions[asset]
        if pos is None:
            return

        price_change_pct = (
            (price - pos["entry_price"]) / pos["entry_price"] * pos["direction"]
        )
        pnl = price_change_pct * (pos["size"] * 100)

        self.equity += pnl
        self.equity -= pos["size"] * 0.00002

        self.env.completed_trades.append(
            {
                "timestamp": self.env._get_current_timestamp(),
                "asset": asset,
                "pnl": pnl,
                "net_pnl": pnl - (pos["size"] * 0.00004),
                "entry_price": pos["entry_price"],
                "exit_price": price,
                "size": pos["size"],
            }
        )
        self.env.positions[asset] = None

    def _update_positions(self):
        current_prices = self._get_current_prices()

        for asset, pos in list(self.env.positions.items()):
            if pos is None:
                continue

            price = current_prices[asset]

            if pos["direction"] == 1:
                if price <= pos["sl"]:
                    self._close_position(asset, price, 0)
                elif price >= pos["tp"]:
                    self._close_position(asset, price, 0)
            else:
                if price >= pos["sl"]:
                    self._close_position(asset, price, 0)
                elif price <= pos["tp"]:
                    self._close_position(asset, price, 0)

    def _get_current_prices(self):
        return {
            asset: self.env.close_arrays[asset][self.env.current_step]
            for asset in self.assets
        }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="Risklayer/models/risk_model_rl_best.pth"
    )
    parser.add_argument(
        "--scaler", type=str, default="Risklayer/models/rl_risk_scaler.pkl"
    )
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--start", type=int, default=500)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    backtester = RLBacktester(args.model, args.scaler, args.data)
    metrics = backtester.run_backtest(args.start, args.end)

    print("\n=== Backtest Results ===")
    print(f"Total Trades: {metrics.get_summary()['num_trades']}")
    print(f"Total PnL: {metrics.get_summary()['total_pnl']:.2f}")
    print(f"Win Rate: {metrics.get_summary()['win_rate']:.2%}")


if __name__ == "__main__":
    main()
