"""
Analyze VSN (Variable Selection Network) feature importance weights
from the trained Alpha V7 model.

Focuses on the asian_range_pos feature (index 19) to determine how
heavily the model depends on it.

Usage:
    cd Alpha && python scripts/analyze_feature_importance.py
    cd Alpha && python scripts/analyze_feature_importance.py --max-rows 50000
    cd Alpha && python scripts/analyze_feature_importance.py --asset EURUSD
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from Alpha.src.data_loader import DataLoader
from Alpha.src.feature_engine import FeatureEngine
from Alpha.src.model import AlphaSLModelV7, AlphaSLModel
from shared_constants import FX_ALPHA_ASSETS

ASSET_MAP = {name: i for i, name in enumerate(FX_ALPHA_ASSETS)}
BAR_MINUTES = 5
SEQ_LEN = 25


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    version = checkpoint.get("format_version", 6)
    cfg = checkpoint["model_config"]
    feature_names = checkpoint.get("feature_names", [])

    if version == 7:
        model = AlphaSLModelV7(**cfg)
    else:
        model = AlphaSLModel(**cfg)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model: version={version}, features={len(feature_names)}, "
          f"input_dim={cfg.get('input_dim', '?')}")
    return model, checkpoint, feature_names


def build_simple_sequences(features, seq_len):
    """Build contiguous sliding windows (no labels needed)."""
    if len(features) < seq_len:
        return np.empty((0, seq_len, features.shape[1]), dtype=np.float32)
    indices = np.arange(seq_len - 1, len(features))
    return np.stack([features[i - seq_len + 1:i + 1] for i in indices]).astype(np.float32)


def get_hour_from_index(timestamps, indices):
    """Get UTC hour for each end-index timestamp."""
    hours = []
    for idx in indices:
        ts = pd.Timestamp(timestamps[int(idx)])
        hours.append(ts.hour)
    return np.array(hours)


def main():
    parser = argparse.ArgumentParser(description="Analyze VSN feature importance")
    parser.add_argument("--model-path", type=str,
                        default=os.path.join(PROJECT_ROOT, "Alpha", "models", "alpha_model.pth"))
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Max raw rows per asset (0=all)")
    parser.add_argument("--asset", type=str, default=None,
                        help="Single asset to analyze (default: all)")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model, checkpoint, feature_names = load_model(args.model_path, device)
    is_v7 = checkpoint.get("format_version", 6) == 7

    if not is_v7:
        print("\nWARNING: Model is V6 (no VSN). Feature importance via VSN is not available.")
        print("V6 uses per-head attention weights, but they are not exposed by default.")
        print("Proceeding to load data anyway for basic inspection.\n")

    # Load data and compute features
    print("Loading data and computing features...")
    loader = DataLoader()
    engine = FeatureEngine()

    assets_to_analyze = [args.asset.upper()] if args.asset else FX_ALPHA_ASSETS
    if args.asset and args.asset.upper() not in ASSET_MAP:
        print(f"Error: Unknown asset '{args.asset}'. Must be one of {FX_ALPHA_ASSETS}")
        return

    data_dict = loader.load_raw_data(max_rows=args.max_rows)
    if not data_dict:
        print("Error: No data files found. Check data/ directory.")
        return

    _, normalized_df = engine.preprocess_data(data_dict)
    print(f"Feature matrix shape: {normalized_df.shape}")

    # Collect VSN weights per asset
    all_vsn_weights = {}  # asset -> (N, T, n_features)
    all_hours = {}        # asset -> (N,) hours

    for asset in assets_to_analyze:
        if asset not in data_dict:
            print(f"Skipping {asset}: no data")
            continue

        print(f"\nProcessing {asset}...")
        obs = engine.get_observation_vectorized(normalized_df, asset)
        n_features = obs.shape[1]
        print(f"  Observations: {len(obs)}, features: {n_features}")

        # Get timestamps for hour extraction
        timestamps = normalized_df.index.to_numpy()

        # Build sequences
        seqs = build_simple_sequences(obs, SEQ_LEN)
        if len(seqs) == 0:
            print(f"  Not enough data for sequences (need >= {SEQ_LEN} bars)")
            continue

        # Build corresponding timestamps for end-indices
        end_indices = np.arange(SEQ_LEN - 1, len(obs))
        hours = get_hour_from_index(timestamps, end_indices)

        print(f"  Sequences: {len(seqs)}, hours range: {hours.min()}-{hours.max()}")

        # Run inference in batches to extract VSN weights
        vsn_list = []
        asset_id = ASSET_MAP.get(asset, 0)
        asset_ids_tensor = torch.full((seqs.shape[0],), asset_id, dtype=torch.long, device=device)

        with torch.no_grad():
            for start in range(0, len(seqs), args.batch_size):
                end = min(start + args.batch_size, len(seqs))
                batch = torch.from_numpy(seqs[start:end]).to(device)
                batch_aids = asset_ids_tensor[start:end]

                if is_v7:
                    out = model(batch, asset_ids=batch_aids, return_dict=True)
                    vsn = out["vsn_weights"].cpu().numpy()  # (B, T, n_features)
                else:
                    # V6: no VSN, run forward and skip
                    model(batch, asset_ids=batch_aids)
                    continue

                vsn_list.append(vsn)

        if is_v7 and vsn_list:
            vsn_all = np.concatenate(vsn_list, axis=0)  # (N, T, n_features)
            all_vsn_weights[asset] = vsn_all
            all_hours[asset] = hours
            print(f"  VSN weights collected: {vsn_all.shape}")

    if not all_vsn_weights:
        print("\nNo VSN weights collected. Model may not be V7.")
        return

    # ============================================================
    # REPORT
    # ============================================================
    target_feature = "asian_range_pos"
    target_idx = None
    if target_feature in feature_names:
        target_idx = feature_names.index(target_feature)
    else:
        # Fallback: hard-coded index
        target_idx = 19
        print(f"Warning: '{target_feature}' not in checkpoint feature_names, using index {target_idx}")

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS (VSN Softmax Weights)")
    print("=" * 70)

    # --- 1. Overall Feature Ranking ---
    print("\n--- 1. Overall Feature Ranking (Mean VSN Weight) ---\n")

    # Aggregate across all assets
    combined_vsn = np.concatenate(list(all_vsn_weights.values()), axis=0)  # (N_total, T, F)
    mean_per_feat = combined_vsn.mean(axis=(0, 1))  # (F,)
    std_per_feat = combined_vsn.std(axis=(0, 1))

    # Sort by importance
    ranking = np.argsort(mean_per_feat)[::-1]
    print(f"{'Rank':<5} {'Feature':<30} {'Mean Weight':>12} {'Std':>10} {'% of Total':>12}")
    print("-" * 70)
    for rank, feat_idx in enumerate(ranking):
        name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"feat_{feat_idx}"
        mean_w = mean_per_feat[feat_idx]
        std_w = std_per_feat[feat_idx]
        pct = mean_w * 100
        marker = " <--- TARGET" if feat_idx == target_idx else ""
        print(f"{rank+1:<5} {name:<30} {mean_w:>12.6f} {std_w:>10.6f} {pct:>10.2f}%{marker}")

    # --- 2. Target Feature Focus ---
    print(f"\n--- 2. '{target_feature}' Focus ---\n")
    target_mean = mean_per_feat[target_idx]
    target_rank = np.where(ranking == target_idx)[0][0] + 1
    n_features = len(mean_per_feat)
    uniform_weight = 1.0 / n_features

    print(f"  Feature index:        {target_idx}")
    print(f"  Mean VSN weight:      {target_mean:.6f}")
    print(f"  Std:                  {std_per_feat[target_idx]:.6f}")
    print(f"  Rank:                 {target_rank} / {n_features}")
    print(f"  Uniform baseline:     {uniform_weight:.6f} (1/{n_features})")
    print(f"  Ratio vs uniform:     {target_mean / uniform_weight:.2f}x")
    print(f"  Percentile:           {(1 - target_rank / n_features) * 100:.1f}%")

    if target_mean / uniform_weight > 1.5:
        print(f"  >>> Model DEPENDS on this feature ({target_mean/uniform_weight:.1f}x above uniform)")
    elif target_mean / uniform_weight > 0.8:
        print(f"  >>> Model uses this feature at roughly average levels")
    else:
        print(f"  >>> Model largely IGNORES this feature ({target_mean/uniform_weight:.2f}x uniform)")

    # --- 3. Asian Range Pos Weight by Hour ---
    print(f"\n--- 3. '{target_feature}' Weight by Hour of Day ---\n")

    combined_hours = np.concatenate(list(all_hours.values()))
    target_per_sample = combined_vsn[:, :, target_idx]  # (N, T)
    target_last_step = target_per_sample[:, -1]          # weight at final timestep

    hour_bins = defaultdict(list)
    for h, w in zip(combined_hours, target_last_step):
        hour_bins[int(h)].append(float(w))

    print(f"{'Hour':<6} {'Mean Weight':>12} {'Std':>10} {'Count':>8} {'Session':>12}")
    print("-" * 55)
    for hour in range(24):
        if hour in hour_bins:
            vals = np.array(hour_bins[hour])
            session = "Asian" if hour < 8 else ("London" if hour < 14 else ("NY" if hour < 20 else "Late"))
            mean_v = vals.mean()
            bar = "#" * int(mean_v / target_last_step.max() * 30) if target_last_step.max() > 0 else ""
            print(f"{hour:<6} {mean_v:>12.6f} {vals.std():>10.6f} {len(vals):>8} {session:>12}  {bar}")

    # Compare Asian vs Non-Asian
    asian_hours_vals = []
    non_asian_hours_vals = []
    for h, w in zip(combined_hours, target_last_step):
        if h < 8:
            asian_hours_vals.append(float(w))
        else:
            non_asian_hours_vals.append(float(w))

    if asian_hours_vals and non_asian_hours_vals:
        asian_mean = np.mean(asian_hours_vals)
        non_asian_mean = np.mean(non_asian_hours_vals)
        print(f"\n  Asian session (00-08) mean weight: {asian_mean:.6f}  (n={len(asian_hours_vals)})")
        print(f"  Non-Asian session mean weight:      {non_asian_mean:.6f}  (n={len(non_asian_hours_vals)})")
        print(f"  Ratio (Asian / Non-Asian):          {asian_mean / non_asian_mean:.2f}x" if non_asian_mean > 0 else "")

    # --- 4. Per-Asset Breakdown ---
    if len(all_vsn_weights) > 1:
        print(f"\n--- 4. Per-Asset '{target_feature}' Weight ---\n")
        print(f"{'Asset':<12} {'Mean Weight':>12} {'Rank':>6} {'Uniform Ratio':>14}")
        print("-" * 50)
        for asset, vsn in all_vsn_weights.items():
            asset_target = vsn[:, :, target_idx].mean()
            asset_mean = vsn.mean(axis=(0, 1))
            asset_rank = np.where(np.argsort(asset_mean)[::-1] == target_idx)[0][0] + 1
            print(f"{asset:<12} {asset_target:>12.6f} {asset_rank:>6}/{len(feature_names)} "
                  f"{asset_target/uniform_weight:>12.2f}x")

    # --- 5. Temporal Attention Summary ---
    print(f"\n--- 5. Summary ---\n")
    print(f"Total sequences analyzed: {len(combined_vsn)}")
    print(f"Sequence length: {SEQ_LEN} bars ({SEQ_LEN * BAR_MINUTES} minutes)")
    print(f"Total features: {n_features}")
    print()

    if target_mean / uniform_weight > 1.5:
        print("RECOMMENDATION: The model assigns above-average importance to asian_range_pos.")
        print("  After fixing the look-ahead bug, RETRAINING is recommended.")
    elif target_mean / uniform_weight > 0.8:
        print("RECOMMENDATION: The model uses asian_range_pos at average levels.")
        print("  Retraining is recommended for correctness, but impact may be modest.")
    else:
        print("RECOMMENDATION: The model largely ignores asian_range_pos.")
        print("  Fix the bug for correctness, but retraining urgency is low.")

    print("\nDone.")


if __name__ == "__main__":
    main()
