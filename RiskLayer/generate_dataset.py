import argparse
import logging
import os
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Alpha.src.data_loader import DataLoader as AlphaDataLoader
from Alpha.src.model import AlphaSLModel
from RiskLayer.src.feature_engine import FeatureEngine

LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _build_windows(values: np.ndarray, end_indices: np.ndarray, seq_len: int) -> np.ndarray:
    windows = []
    for end_idx in end_indices:
        start_idx = end_idx - seq_len + 1
        windows.append(values[start_idx : end_idx + 1])
    return np.asarray(windows, dtype=np.float32)


def _load_alpha_model(model_path: Path, input_dim: int, device: torch.device) -> AlphaSLModel:
    model = AlphaSLModel(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def generate_dataset(
    data_dir: str,
    output_dir: str,
    alpha_model_path: str,
    seq_len: int = 50,
    adx_threshold: float = 20.0,
    alpha_threshold: float = 0.50,
    lookahead_candles: int = 24,
    batch_size: int = 2048,
) -> tuple[str, str]:
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")
    if lookahead_candles < 1:
        raise ValueError("lookahead_candles must be >= 1")
    if adx_threshold < 0:
        raise ValueError("adx_threshold must be >= 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = AlphaDataLoader(data_dir=data_dir)
    engine = FeatureEngine()

    aligned_df, normalized_df = loader.get_features()

    input_dim = len(engine.get_observation_vectorized(normalized_df.head(1), loader.assets[0])[0])
    alpha_model = _load_alpha_model(Path(alpha_model_path), input_dim=input_dim, device=device)

    all_sequences = []
    all_actions = []
    all_confidences = []
    all_future_high_max = []
    all_future_low_min = []
    all_future_high_paths = []
    all_future_low_paths = []
    all_entry_prices = []
    all_signal_atr = []
    all_asset_ids = []
    all_signal_steps = []

    for asset_id, asset in enumerate(loader.assets):
        obs = engine.get_observation_vectorized(normalized_df, asset)

        atr = aligned_df[f"{asset}_atr"].values
        adx = aligned_df[f"{asset}_adx"].values
        highs = aligned_df[f"{asset}_high"].values
        lows = aligned_df[f"{asset}_low"].values
        closes = aligned_df[f"{asset}_close"].values

        max_end_idx = len(obs) - lookahead_candles
        if max_end_idx < seq_len - 1:
            LOGGER.warning("Skipping %s: not enough rows for seq_len + lookahead.", asset)
            continue

        candidate_indices = [
            end_idx
            for end_idx in range(seq_len - 1, max_end_idx + 1)
            if adx[end_idx] >= adx_threshold
        ]

        if not candidate_indices:
            LOGGER.info(
                "No ADX-qualified rows for %s (threshold=%.2f).",
                asset,
                adx_threshold,
            )
            continue

        candidate_indices = np.asarray(candidate_indices, dtype=np.int32)
        sequences = _build_windows(obs, candidate_indices, seq_len)

        probs_accum = []
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = torch.from_numpy(sequences[i : i + batch_size]).to(device)
                logits = alpha_model(batch)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs_accum.append(probs)

        probs = np.concatenate(probs_accum, axis=0)
        pred_class = probs.argmax(axis=1)
        pred_conf = probs.max(axis=1)

        # Map class ids [0,1,2] -> actions [-1,0,1]
        actions = pred_class.astype(np.int8) - 1
        
        # --- APPLY THRESHOLD: Signals below alpha_threshold are Neutral (0) ---
        actions = np.where(pred_conf >= alpha_threshold, actions, 0).astype(np.int8)
        
        # --- FILTER: Only keep Buy/Sell signals. Drop everything else. ---
        trade_mask = actions != 0
        
        if not np.any(trade_mask):
            LOGGER.info("No signals passed the %.2f threshold for %s. Skipping.", alpha_threshold, asset)
            continue
            
        # Apply mask to all relevant data
        candidate_indices = candidate_indices[trade_mask]
        sequences = sequences[trade_mask]
        probs = probs[trade_mask]
        pred_conf = pred_conf[trade_mask]
        actions = actions[trade_mask]

        # --- NEW: Append Alpha Model Outputs as Features for Risk Model ---
        # probs (3 features) + final action (1 feature) = 4 extra features
        # Broadcast (N, 4) -> (N, seq_len, 4) and concatenate to sequences (N, seq_len, 17)
        
        N = sequences.shape[0]
        seq_len = sequences.shape[1]
        
        # Create alpha features array: [prob_sell, prob_neutral, prob_buy, final_action]
        alpha_features = np.zeros((N, 4), dtype=np.float32)
        alpha_features[:, 0:3] = probs
        alpha_features[:, 3] = actions.astype(np.float32)
        
        # Broadcast and concatenate
        # alpha_features_expanded shape: (N, seq_len, 4)
        alpha_features_expanded = np.tile(alpha_features[:, np.newaxis, :], (1, seq_len, 1))
        
        # New sequences shape: (N, seq_len, 21)
        sequences_with_alpha = np.concatenate([sequences, alpha_features_expanded], axis=2)

        future_high_paths = np.asarray(
            [highs[idx : idx + lookahead_candles] for idx in candidate_indices],
            dtype=np.float32,
        )
        future_low_paths = np.asarray(
            [lows[idx : idx + lookahead_candles] for idx in candidate_indices],
            dtype=np.float32,
        )

        all_sequences.append(sequences_with_alpha)
        all_actions.append(actions)
        all_confidences.append(pred_conf.astype(np.float32))
        all_future_high_paths.append(future_high_paths)
        all_future_low_paths.append(future_low_paths)
        all_future_high_max.append(future_high_paths.max(axis=1))
        all_future_low_min.append(future_low_paths.min(axis=1))
        all_entry_prices.append(closes[candidate_indices].astype(np.float32))
        all_signal_atr.append(atr[candidate_indices].astype(np.float32))
        all_asset_ids.append(np.full(len(candidate_indices), asset_id, dtype=np.int8))
        all_signal_steps.append(candidate_indices)

        LOGGER.info(
            "%s: generated %d ADX-qualified sequences (threshold=%.2f).",
            asset,
            len(candidate_indices),
            adx_threshold,
        )

    if not all_sequences:
        raise RuntimeError("No dataset rows were generated. Check thresholds and input data.")

    X = np.concatenate(all_sequences, axis=0).astype(np.float32)
    y_action = np.concatenate(all_actions, axis=0).astype(np.int8)
    y_conf = np.concatenate(all_confidences, axis=0).astype(np.float32)
    future_high_max = np.concatenate(all_future_high_max, axis=0).astype(np.float32)
    future_low_min = np.concatenate(all_future_low_min, axis=0).astype(np.float32)
    future_high_paths = np.concatenate(all_future_high_paths, axis=0).astype(np.float32)
    future_low_paths = np.concatenate(all_future_low_paths, axis=0).astype(np.float32)
    entry_price = np.concatenate(all_entry_prices, axis=0).astype(np.float32)
    signal_atr = np.concatenate(all_signal_atr, axis=0).astype(np.float32)
    asset_ids = np.concatenate(all_asset_ids, axis=0).astype(np.int8)
    signal_steps = np.concatenate(all_signal_steps, axis=0).astype(np.int32)

    os.makedirs(output_dir, exist_ok=True)
    seq_path = os.path.join(output_dir, "sequences.npy")
    labels_path = os.path.join(output_dir, "labels.npz")

    np.save(seq_path, X)
    np.savez(
        labels_path,
        action=y_action,
        confidence=y_conf,
        future_high_max=future_high_max,
        future_low_min=future_low_min,
        future_high_path=future_high_paths,
        future_low_path=future_low_paths,
        entry_price=entry_price,
        signal_atr=signal_atr,
        asset_id=asset_ids,
        signal_step=signal_steps,
        asset_names=np.asarray(loader.assets),
        adx_threshold=np.float32(adx_threshold),
        alpha_threshold=np.float32(alpha_threshold),
        lookahead_candles=np.int32(lookahead_candles),
        sequence_length=np.int32(seq_len),
    )

    LOGGER.info("Saved %d sequences to %s", len(X), seq_path)
    LOGGER.info("Saved labels/metadata to %s", labels_path)
    return seq_path, labels_path


def main() -> None:
    _configure_logging()

    parser = argparse.ArgumentParser(description="Generate risk dataset with Alpha-identical feature stack.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="RiskLayer/data/training_set")
    parser.add_argument("--alpha-model-path", type=str, default="Alpha/models/alpha_model.pth")
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--adx-threshold", type=float, default=20.0)
    parser.add_argument("--alpha-threshold", type=float, default=0.50)
    parser.add_argument("--lookahead-candles", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    generate_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        alpha_model_path=args.alpha_model_path,
        seq_len=args.seq_len,
        adx_threshold=args.adx_threshold,
        alpha_threshold=args.alpha_threshold,
        lookahead_candles=args.lookahead_candles,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
