"""Alpha Model Feature Behavior Analysis

Loads the trained model + data, runs 10 diagnostic metrics on the test split
to find features that hurt predictions and should be removed.

Outputs:
  - feature_ranking.csv       (all features ranked by importance)
  - harmful_features.json     (features with negative permutation delta)
  - feature_correlation.png   (heatmap of inter-feature correlations)
  - importance_bar_chart.png  (permutation importance bar chart)
  - attention_heatmap.png     (attention weight distribution)
  - analysis_report.json      (full JSON report with all metrics)
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- matplotlib (non-interactive backend) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, log_loss

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from Alpha.src.model import AlphaSLModel, AlphaSLModelV7
from Alpha.src.feature_engine import FeatureEngine

# ---------------------------------------------------------------------------
# Constants (must match run_pipeline.py)
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 25
LABEL_MAX_BARS = 18
BAR_MINUTES = 5
PURGE_TD = np.timedelta64(LABEL_MAX_BARS * BAR_MINUTES, "m")
EMBARGO_TD = np.timedelta64((SEQUENCE_LENGTH + LABEL_MAX_BARS) * BAR_MINUTES, "m")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "training_set")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "alpha_model.pth")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_output")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def date_based_split(timestamps, train_ratio=0.70, val_ratio=0.15):
    ts = np.asarray(timestamps, dtype="datetime64[ns]")
    unique_ts = np.unique(ts)
    train_cut = unique_ts[max(0, min(len(unique_ts) - 2, int(len(unique_ts) * train_ratio) - 1))]
    val_cut = unique_ts[max(1, min(len(unique_ts) - 1, int(len(unique_ts) * (train_ratio + val_ratio)) - 1))]
    train_idx = np.flatnonzero(ts <= train_cut - PURGE_TD)
    val_idx = np.flatnonzero((ts > train_cut + EMBARGO_TD) & (ts <= val_cut - PURGE_TD))
    test_idx = np.flatnonzero(ts > val_cut + EMBARGO_TD)
    return train_idx, val_idx, test_idx


def load_model_and_data():
    """Load trained model checkpoint, sequences, and labels."""
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    config = ckpt["model_config"]
    fmt_version = ckpt.get("format_version", 6)

    if fmt_version == 7:
        model = AlphaSLModelV7(
            input_dim=config["input_dim"],
            lstm_units=config["lstm_units"],
            dense_units=config["dense_units"],
            dropout=config["dropout"],
            num_assets=config["num_assets"],
            asset_embedding_dim=config["asset_embedding_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            bidirectional=config["bidirectional"],
        ).to(DEVICE)
    else:
        model = AlphaSLModel(
            input_dim=config["input_dim"],
            lstm_units=config["lstm_units"],
            dense_units=config["dense_units"],
            dropout=config["dropout"],
            num_assets=config["num_assets"],
            asset_embedding_dim=config["asset_embedding_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            bidirectional=config["bidirectional"],
        ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sequences = np.load(os.path.join(DATA_DIR, "sequences.npy"))
    labels = np.load(os.path.join(DATA_DIR, "labels.npz"))
    feature_names = ckpt["feature_names"]
    print(f"  Model format version: {fmt_version}")
    return model, sequences, labels, feature_names


def get_test_indices(labels):
    _, _, test_idx = date_based_split(labels["timestamp"])
    return test_idx


def run_inference(model, sequences, asset_ids, indices, batch_size=128):
    """Run model inference, return logits, attention weights.
    sequences: local array (already extracted from mmap).
    indices: local indices (0-based into sequences).
    """
    all_logits = []
    all_attn_weights = []

    is_v7 = isinstance(model, AlphaSLModelV7)

    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i : i + batch_size]
            batch_x = torch.from_numpy(np.asarray(sequences[batch_idx], dtype=np.float32)).to(DEVICE)
            batch_assets = torch.from_numpy(asset_ids[batch_idx].astype(np.int64)).to(DEVICE)

            if is_v7:
                outputs = model(batch_x, batch_assets, return_dict=True)
                all_logits.append(outputs["action_logits"].detach().cpu().numpy())
                if "attn_weights" in outputs:
                    all_attn_weights.append(outputs["attn_weights"].detach().cpu().numpy())
                else:
                    all_attn_weights.append(np.zeros((len(batch_idx), 1, 1)))
            else:
                # V6: manual forward to capture attention
                lstm_out, _ = model.lstm(batch_x)
                lstm_out_norm = model.lstm_norm(lstm_out)

                head_outputs = []
                attn_weights_per_head = []
                for head_attn in model.attention_heads:
                    attn_scores = head_attn(lstm_out_norm)
                    attn_w = torch.softmax(attn_scores, dim=1)
                    attn_weights_per_head.append(attn_w.detach().cpu().numpy())
                    context = torch.sum(attn_w * lstm_out_norm, dim=1)
                    head_outputs.append(context)
                multi_head_context = torch.cat(head_outputs, dim=1)
                context_vector = model.attn_proj(multi_head_context)

                asset_context = model.asset_embedding(batch_assets)
                x = F.relu(model.fc1(torch.cat([context_vector, asset_context], dim=1)))
                x = model.dropout(x)
                residual = x
                x = F.relu(model.fc2(x))
                x = model.dropout(x)
                x = x + residual
                logits = model.action_head(x)

                all_logits.append(logits.detach().cpu().numpy())
                all_attn_weights.append(np.stack(attn_weights_per_head, axis=1))  # (B, n_heads, seq_len, 1)

    return np.concatenate(all_logits), np.concatenate(all_attn_weights)


# ---------------------------------------------------------------------------
# Metric 1 & 2: Permutation Importance (accuracy delta + loss delta)
# ---------------------------------------------------------------------------
def permutation_importance(model, sequences, asset_ids, targets, feature_names,
                           indices, n_repeats=5):
    """For each feature, shuffle it and measure accuracy/loss change."""
    test_seqs = sequences[indices].copy()
    test_assets = asset_ids[indices]
    local_idx = np.arange(len(indices))

    base_logits, _ = run_inference(model, test_seqs, test_assets, local_idx)
    base_probs = 1 / (1 + np.exp(-base_logits.squeeze(-1)))
    base_preds = (base_probs > 0.5).astype(int)
    base_acc = accuracy_score(targets, base_preds)
    base_loss = log_loss(targets, np.column_stack([1 - base_probs, base_probs]))

    n_features = test_seqs.shape[2]
    results = []
    rng = np.random.RandomState(42)

    for feat_i in range(n_features):
        acc_deltas = []
        loss_deltas = []
        for _ in range(n_repeats):
            perm = test_seqs.copy()
            perm[:, :, feat_i] = rng.permutation(perm[:, :, feat_i])

            logits, _ = run_inference(model, perm, test_assets, local_idx)
            probs = 1 / (1 + np.exp(-logits.squeeze(-1)))
            preds = (probs > 0.5).astype(int)
            perm_acc = accuracy_score(targets, preds)
            perm_loss = log_loss(targets, probs)

            acc_deltas.append(perm_acc - base_acc)
            loss_deltas.append(base_loss - perm_loss)  # positive = harmful

        results.append({
            "feature": feature_names[feat_i],
            "feature_idx": feat_i,
            "acc_delta_mean": float(np.mean(acc_deltas)),
            "acc_delta_std": float(np.std(acc_deltas)),
            "loss_delta_mean": float(np.mean(loss_deltas)),
            "loss_delta_std": float(np.std(loss_deltas)),
            "base_acc": base_acc,
            "base_loss": base_loss,
        })

    return results, base_acc, base_loss


# ---------------------------------------------------------------------------
# Metric 3: Feature-Label Pearson Correlation
# ---------------------------------------------------------------------------
def feature_label_correlation(features_last_bar, labels, feature_names):
    results = []
    for i, name in enumerate(feature_names):
        r = np.corrcoef(features_last_bar[:, i], labels)[0, 1]
        results.append({"feature": name, "pearson_r": float(r) if np.isfinite(r) else 0.0})
    return sorted(results, key=lambda x: -abs(x["pearson_r"]))


# ---------------------------------------------------------------------------
# Metric 4: Mutual Information
# ---------------------------------------------------------------------------
def mutual_information(features_last_bar, labels, feature_names):
    mi_scores = mutual_info_classif(features_last_bar, labels, random_state=42, n_neighbors=5)
    results = [{"feature": name, "mi": float(mi_scores[i])} for i, name in enumerate(feature_names)]
    return sorted(results, key=lambda x: -x["mi"])


# ---------------------------------------------------------------------------
# Metric 5: Gradient Saliency
# ---------------------------------------------------------------------------
def gradient_saliency(model, sequences, asset_ids, targets, feature_names, indices):
    model.eval()
    all_grads = []

    test_seqs = sequences[indices].copy()
    test_assets = asset_ids[indices]
    all_idx = np.arange(len(indices))

    for i in range(0, len(all_idx), 128):
        batch_idx = all_idx[i : i + 128]
        batch_x = torch.from_numpy(np.asarray(test_seqs[batch_idx], dtype=np.float32)).to(DEVICE)
        batch_x.requires_grad_(True)
        batch_assets = torch.from_numpy(test_assets[batch_idx].astype(np.int64)).to(DEVICE)

        outputs = model(batch_x, batch_assets, return_dict=True)
        logits = outputs["action_logits"].squeeze(-1)

        # Sum of logit magnitudes as scalar
        loss = logits.abs().sum()
        model.zero_grad()
        if batch_x.grad is not None:
            batch_x.grad.zero_()
        loss.backward()

        grads = batch_x.grad.abs().mean(dim=(0, 1)).detach().cpu().numpy()
        all_grads.append(grads)

    avg_grads = np.mean(all_grads, axis=0)
    results = [{"feature": feature_names[i], "mean_abs_grad": float(avg_grads[i])}
               for i in range(len(feature_names))]
    return sorted(results, key=lambda x: -x["mean_abs_grad"])


# ---------------------------------------------------------------------------
# Metric 6: Feature Variance
# ---------------------------------------------------------------------------
def feature_variance(features_last_bar, feature_names):
    variances = features_last_bar.var(axis=0)
    return [{"feature": feature_names[i], "variance": float(variances[i])}
            for i in range(len(feature_names))]


# ---------------------------------------------------------------------------
# Metric 7: Inter-Feature Correlation
# ---------------------------------------------------------------------------
def inter_feature_correlation(features_last_bar, feature_names, threshold=0.95):
    corr = np.corrcoef(features_last_bar.T)
    redundant_pairs = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            if abs(corr[i, j]) > threshold:
                redundant_pairs.append({
                    "feature_a": feature_names[i],
                    "feature_b": feature_names[j],
                    "correlation": float(corr[i, j]),
                })
    return sorted(redundant_pairs, key=lambda x: -abs(x["correlation"])), corr


# ---------------------------------------------------------------------------
# Metric 8: Drop-one-feature accuracy
# ---------------------------------------------------------------------------
def drop_one_accuracy(model, sequences, asset_ids, targets, feature_names, indices):
    """Drop one feature (zero it out) and measure accuracy change."""
    test_seqs = sequences[indices].copy()
    test_assets = asset_ids[indices]
    all_idx = np.arange(len(indices))

    base_logits, _ = run_inference(model, test_seqs, test_assets, all_idx)
    base_probs = 1 / (1 + np.exp(-base_logits.squeeze(-1)))
    base_acc = accuracy_score(targets, (base_probs > 0.5).astype(int))

    n_features = test_seqs.shape[2]
    results = []
    for feat_i in range(n_features):
        modified = test_seqs.copy()
        modified[:, :, feat_i] = 0.0
        logits, _ = run_inference(model, modified, test_assets, all_idx)
        probs = 1 / (1 + np.exp(-logits.squeeze(-1)))
        acc = accuracy_score(targets, (probs > 0.5).astype(int))
        results.append({
            "feature": feature_names[feat_i],
            "acc_with_drop": float(acc),
            "acc_drop_delta": float(acc - base_acc),  # positive = feature helped, negative = removing helped
            "base_acc": base_acc,
        })
    return sorted(results, key=lambda x: x["acc_drop_delta"])


# ---------------------------------------------------------------------------
# Metric 9: Attention Weight Distribution
# ---------------------------------------------------------------------------
def attention_distribution(attn_weights, feature_names):
    """Analyze which timesteps attention heads focus on.
    attn_weights shape: (n_samples, n_heads, seq_len, 1)
    """
    n_heads = attn_weights.shape[1]
    seq_len = attn_weights.shape[2]
    head_stats = []
    for h in range(n_heads):
        head_attn = attn_weights[:, h, :, 0]  # (n_samples, seq_len)
        mean_per_position = head_attn.mean(axis=0)
        head_stats.append({
            "head": h,
            "mean_per_position": [round(float(x), 6) for x in mean_per_position],
            "entropy": round(float(-np.sum(mean_per_position * np.log(mean_per_position + 1e-12))), 4),
            "max_position": int(np.argmax(mean_per_position)),
            "concentration": round(float(np.max(mean_per_position) / mean_per_position.mean()), 4),
        })
    return head_stats


# ---------------------------------------------------------------------------
# Metric 10: Regime-Conditional Importance
# ---------------------------------------------------------------------------
def regime_conditional_importance(features_last_bar, labels, feature_names, sequences, indices):
    """Split test samples into trending (regime=1) vs ranging (regime=0) and
    compute per-feature F-score within each regime."""
    # regime is feature index 3 (feature_names[3] == "regime")
    regime_idx = feature_names.index("regime") if "regime" in feature_names else 3
    regime_vals = features_last_bar[:, regime_idx]

    results = {"trending": [], "ranging": []}
    classes = np.unique(labels)

    for regime_val, regime_name in [(1.0, "trending"), (0.0, "ranging")]:
        mask = regime_vals == regime_val
        if mask.sum() < 5:
            results[regime_name] = [{"feature": fn, "f_score": 0.0, "n": 0} for fn in feature_names]
            continue

        sub_features = features_last_bar[mask]
        sub_labels = labels[mask]
        overall_mean = sub_features.mean(axis=0)

        ssb = np.zeros(sub_features.shape[1])
        ssw = np.zeros(sub_features.shape[1])
        for c in classes:
            grp = sub_features[sub_labels == c]
            if len(grp) == 0:
                continue
            grp_mean = grp.mean(axis=0)
            ssb += len(grp) * (grp_mean - overall_mean) ** 2
            ssw += ((grp - grp_mean) ** 2).sum(axis=0)

        dfb = max(1, len(classes) - 1)
        dfw = max(1, len(sub_labels) - len(classes))
        f_scores = (ssb / dfb) / (ssw / dfw + 1e-12)

        results[regime_name] = [
            {"feature": feature_names[i], "f_score": round(float(f_scores[i]), 4), "n": int(mask.sum())}
            for i in range(len(feature_names))
        ]
        results[regime_name].sort(key=lambda x: -x["f_score"])

    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_importance_bar_chart(ranking, output_path):
    features = [r["feature"] for r in ranking]
    deltas = [r["perm_acc_delta"] for r in ranking]
    colors = ["#e74c3c" if d > 0 else "#2ecc71" for d in deltas]

    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.35)))
    y_pos = np.arange(len(features))
    ax.barh(y_pos, deltas, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=8)
    ax.set_xlabel("Accuracy Delta (shuffle) — Red=harmful, Green=helpful")
    ax.set_title("Permutation Importance: Features Ranked by Accuracy Impact")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def plot_correlation_heatmap(corr_matrix, feature_names, output_path):
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix, mask=mask, annot=False, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        xticklabels=feature_names, yticklabels=feature_names,
        ax=ax, square=True, linewidths=0.5,
    )
    ax.set_title("Inter-Feature Correlation Matrix")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def plot_attention_heatmap(head_stats, output_path):
    n_heads = len(head_stats)
    seq_len = len(head_stats[0]["mean_per_position"])
    data = np.array([h["mean_per_position"] for h in head_stats])

    fig, ax = plt.subplots(figsize=(10, max(3, n_heads * 0.8)))
    sns.heatmap(data, annot=True, fmt=".4f", cmap="YlOrRd",
                xticklabels=[f"t-{seq_len - 1 - i}" for i in range(seq_len)],
                yticklabels=[f"Head {i}" for i in range(n_heads)],
                ax=ax)
    ax.set_title("Attention Weight Distribution by Head and Timestep")
    ax.set_xlabel("Timestep (relative to prediction)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("ALPHA MODEL FEATURE BEHAVIOR ANALYSIS")
    print("=" * 70)

    # 1. Load
    print("\n[1/10] Loading model and data...")
    model, sequences, labels, feature_names = load_model_and_data()
    test_idx = get_test_indices(labels)
    targets = labels["action_classes"][test_idx].astype(int)
    asset_ids = labels["asset_id"]

    # Last bar features for test set
    features_last_bar = sequences[test_idx, -1, :]  # (N_test, 48)
    n_features = sequences.shape[2]
    n_test = len(test_idx)

    print(f"  Test samples: {n_test}")
    print(f"  Features: {n_features}")
    print(f"  Class distribution: hold={int((targets == 0).sum())}, buy={int((targets == 1).sum())}")
    print(f"  Model on device: {DEVICE}")

    report = {
        "generated_at": datetime.now().isoformat(),
        "n_test_samples": int(n_test),
        "n_features": int(n_features),
        "feature_names": feature_names,
    }

    # 2. Permutation Importance
    print("\n[2/10] Permutation Importance (accuracy + loss delta)...")
    perm_results, base_acc, base_loss = permutation_importance(
        model, sequences, asset_ids, targets, feature_names, test_idx
    )
    report["baseline_accuracy"] = round(base_acc, 4)
    report["baseline_log_loss"] = round(base_loss, 4)
    report["permutation_importance"] = perm_results
    print(f"  Baseline accuracy: {base_acc:.4f}")
    print(f"  Baseline log loss: {base_loss:.4f}")

    # Identify harmful features (shuffling improved accuracy)
    harmful_acc = [r for r in perm_results if r["acc_delta_mean"] > 0]
    harmful_loss = [r for r in perm_results if r["loss_delta_mean"] > 0]
    print(f"  Harmful features (accuracy improves on shuffle): {len(harmful_acc)}")
    for h in sorted(harmful_acc, key=lambda x: -x["acc_delta_mean"]):
        print(f"    {h['feature']:30s}  acc_delta=+{h['acc_delta_mean']:.4f}")
    print(f"  Harmful features (loss drops on shuffle): {len(harmful_loss)}")

    # 3. Pearson Correlation
    print("\n[3/10] Feature-Label Pearson Correlation...")
    corr_results = feature_label_correlation(features_last_bar, targets, feature_names)
    report["pearson_correlation"] = corr_results
    for r in corr_results[:5]:
        print(f"  {r['feature']:30s}  r={r['pearson_r']:+.4f}")
    print("  ...")
    for r in corr_results[-5:]:
        print(f"  {r['feature']:30s}  r={r['pearson_r']:+.4f}")

    # 4. Mutual Information
    print("\n[4/10] Mutual Information...")
    mi_results = mutual_information(features_last_bar, targets, feature_names)
    report["mutual_information"] = mi_results
    for r in mi_results[:5]:
        print(f"  {r['feature']:30s}  MI={r['mi']:.6f}")
    print("  ...")
    for r in mi_results[-3:]:
        print(f"  {r['feature']:30s}  MI={r['mi']:.6f}")

    # 5. Gradient Saliency
    print("\n[5/10] Gradient Saliency...")
    grad_results = gradient_saliency(model, sequences, asset_ids, targets, feature_names, test_idx)
    report["gradient_saliency"] = grad_results
    for r in grad_results[:5]:
        print(f"  {r['feature']:30s}  |grad|={r['mean_abs_grad']:.6f}")

    # 6. Feature Variance
    print("\n[6/10] Feature Variance...")
    var_results = feature_variance(features_last_bar, feature_names)
    report["feature_variance"] = var_results
    low_var = [r for r in var_results if r["variance"] < 1e-6]
    if low_var:
        print(f"  WARNING: {len(low_var)} near-constant features:")
        for r in low_var:
            print(f"    {r['feature']:30s}  var={r['variance']:.2e}")
    else:
        print(f"  All features have non-zero variance.")

    # 7. Inter-Feature Correlation
    print("\n[7/10] Inter-Feature Correlation (threshold > 0.95)...")
    redundant, corr_matrix = inter_feature_correlation(features_last_bar, feature_names)
    report["redundant_pairs"] = redundant
    report["correlation_matrix"] = corr_matrix.tolist()
    if redundant:
        print(f"  Found {len(redundant)} redundant pairs:")
        for r in redundant:
            print(f"    {r['feature_a']:30s} <-> {r['feature_b']:30s}  r={r['correlation']:.4f}")
    else:
        print("  No highly redundant feature pairs found.")

    # 8. Drop-one-feature Accuracy
    print("\n[8/10] Drop-One-Feature Accuracy...")
    drop_results = drop_one_accuracy(model, sequences, asset_ids, targets, feature_names, test_idx)
    report["drop_one_accuracy"] = drop_results
    print("  Features where dropping improves accuracy:")
    for r in drop_results:
        if r["acc_drop_delta"] > 0:
            print(f"    {r['feature']:30s}  acc_delta=+{r['acc_drop_delta']:.4f}")
    print("  Features where dropping hurts accuracy (most important):")
    for r in drop_results[:5]:
        print(f"    {r['feature']:30s}  acc_delta={r['acc_drop_delta']:.4f}")

    # 9. Attention Distribution
    print("\n[9/10] Attention Weight Distribution...")
    test_seqs_for_attn = sequences[test_idx].copy()
    test_assets_for_attn = asset_ids[test_idx]
    all_local_idx = np.arange(len(test_idx))
    _, attn_weights = run_inference(model, test_seqs_for_attn, test_assets_for_attn, all_local_idx)
    attn_stats = attention_distribution(attn_weights, feature_names)
    report["attention_distribution"] = attn_stats
    for h in attn_stats:
        print(f"  Head {h['head']}: entropy={h['entropy']:.3f}, "
              f"max_at_t-{24 - h['max_position']}, concentration={h['concentration']:.2f}")

    # 10. Regime-Conditional Importance
    print("\n[10/10] Regime-Conditional Importance...")
    regime_results = regime_conditional_importance(features_last_bar, targets, feature_names, sequences, test_idx)
    report["regime_conditional"] = regime_results
    print("  Top features in TRENDING regime:")
    for r in regime_results["trending"][:5]:
        print(f"    {r['feature']:30s}  F={r['f_score']:.4f}  (n={r['n']})")
    print("  Top features in RANGING regime:")
    for r in regime_results["ranging"][:5]:
        print(f"    {r['feature']:30s}  F={r['f_score']:.4f}  (n={r['n']})")

    # -----------------------------------------------------------------------
    # Aggregate: Final Feature Ranking
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("AGGREGATE FEATURE RANKING")
    print("=" * 70)

    # Normalize each metric to 0-1 scale, then combine
    perm_lookup = {r["feature"]: r["acc_delta_mean"] for r in perm_results}
    mi_lookup = {r["feature"]: r["mi"] for r in mi_results}
    grad_lookup = {r["feature"]: r["mean_abs_grad"] for r in grad_results}
    corr_lookup = {r["feature"]: abs(r["pearson_r"]) for r in corr_results}
    drop_lookup = {r["feature"]: r["acc_drop_delta"] for r in drop_results}

    # For permutation: higher positive delta = more harmful (bad for model)
    # We want to rank: harmful features first (to remove them)
    perm_vals = np.array([perm_lookup[f] for f in feature_names])
    mi_vals = np.array([mi_lookup.get(f, 0) for f in feature_names])
    grad_vals = np.array([grad_lookup.get(f, 0) for f in feature_names])
    corr_vals = np.array([corr_lookup.get(f, 0) for f in feature_names])
    drop_vals = np.array([drop_lookup.get(f, 0) for f in feature_names])

    def safe_normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    # Harmfulness score: high permutation delta + high drop-one delta = bad
    # Helpfulness score: high MI + high gradient + high correlation = good
    perm_norm = safe_normalize(perm_vals)          # high = harmful
    drop_norm = safe_normalize(drop_vals)          # high = helps (so low = harmful)
    mi_norm = safe_normalize(mi_vals)              # high = useful
    grad_norm_s = safe_normalize(grad_vals)        # high = used
    corr_norm = safe_normalize(corr_vals)          # high = correlated

    # Harmfulness = features to REMOVE
    # (high perm delta) + (low drop delta) + (low MI) + (low gradient) + (low correlation)
    harm_score = perm_norm + (1 - drop_norm) + (1 - mi_norm) + (1 - grad_norm_s) + (1 - corr_norm)

    final_ranking = []
    for i, f in enumerate(feature_names):
        final_ranking.append({
            "feature": f,
            "harm_score": round(float(harm_score[i]), 4),
            "perm_acc_delta": round(float(perm_vals[i]), 6),
            "mi": round(float(mi_vals[i]), 6),
            "grad_saliency": round(float(grad_vals[i]), 6),
            "abs_pearson_r": round(float(corr_vals[i]), 4),
            "drop_acc_delta": round(float(drop_vals[i]), 4),
        })
    final_ranking.sort(key=lambda x: -x["harm_score"])

    report["final_ranking"] = final_ranking

    # Identify features to remove:
    # 1. Features with zero variance (dead constants)
    # 2. Features with zero MI AND zero Pearson correlation
    # 3. Features with positive perm_acc_delta (shuffling helps = harmful)
    # 4. Features with positive drop_acc_delta (dropping helps = harmful)
    features_to_remove = set()
    features_to_remove_weak = set()

    for r in final_ranking:
        reasons = []
        # Dead constant
        if r["mi"] == 0.0 and r["abs_pearson_r"] == 0.0 and r["grad_saliency"] < 0.0004:
            reasons.append("zero_variance_or_constant")
        # Harmful by permutation
        if r["perm_acc_delta"] > 0:
            reasons.append("permutation_harmful")
        # Harmful by drop-one
        if r["drop_acc_delta"] > 0:
            reasons.append("drop_one_harmful")

        if len(reasons) >= 2:
            features_to_remove.add(r["feature"])
        elif len(reasons) >= 1 and r["harm_score"] > 3.5:
            features_to_remove_weak.add(r["feature"])

    # Also remove features that are perfectly correlated with kept features
    # (from redundant pairs where one is already flagged)
    redundant_to_remove = set()
    for pair in redundant:
        a, b = pair["feature_a"], pair["feature_b"]
        if abs(pair["correlation"]) >= 0.999:
            if a in features_to_remove or a in features_to_remove_weak:
                redundant_to_remove.add(b)
            elif b in features_to_remove or b in features_to_remove_weak:
                redundant_to_remove.add(a)

    all_remove = sorted(features_to_remove | redundant_to_remove)
    all_remove_weak = sorted(features_to_remove_weak - features_to_remove - redundant_to_remove)

    report["features_to_remove"] = all_remove
    report["features_to_remove_weak"] = all_remove_weak
    report["redundant_to_remove"] = sorted(redundant_to_remove)

    print(f"\n{'Feature':30s} {'HarmScore':>10s} {'PermD':>10s} {'MI':>10s} {'Grad':>10s} {'AbsR':>8s} {'DropD':>10s}")
    print("-" * 90)
    for r in final_ranking:
        marker = ""
        if r["feature"] in features_to_remove:
            marker = " <<< REMOVE"
        elif r["feature"] in redundant_to_remove:
            marker = " <<< REDUNDANT"
        elif r["feature"] in features_to_remove_weak:
            marker = " <<< WEAK"
        print(f"{r['feature']:30s} {r['harm_score']:10.4f} {r['perm_acc_delta']:+10.6f} "
              f"{r['mi']:10.6f} {r['grad_saliency']:10.6f} {r['abs_pearson_r']:8.4f} "
              f"{r['drop_acc_delta']:+10.4f}{marker}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    print("\n[Plotting] Generating charts...")
    chart1 = plot_importance_bar_chart(final_ranking, os.path.join(OUTPUT_DIR, "importance_bar_chart.png"))
    chart2 = plot_correlation_heatmap(corr_matrix, feature_names, os.path.join(OUTPUT_DIR, "feature_correlation_matrix.png"))
    chart3 = plot_attention_heatmap(attn_stats, os.path.join(OUTPUT_DIR, "attention_heatmap.png"))
    print(f"  {chart1}")
    print(f"  {chart2}")
    print(f"  {chart3}")

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    # JSON report (without numpy arrays)
    report_clean = {k: v for k, v in report.items() if k != "correlation_matrix"}
    report_path = os.path.join(OUTPUT_DIR, "analysis_report.json")
    with open(report_path, "w") as f:
        json.dump(report_clean, f, indent=2, default=str)

    # Feature ranking CSV
    ranking_df = pd.DataFrame(final_ranking)
    ranking_csv = os.path.join(OUTPUT_DIR, "feature_ranking.csv")
    ranking_df.to_csv(ranking_csv, index=False)

    # Harmful features JSON
    harmful_path = os.path.join(OUTPUT_DIR, "harmful_features.json")
    with open(harmful_path, "w") as f:
        json.dump({
            "features_to_remove": all_remove,
            "features_to_remove_weak": all_remove_weak,
            "redundant_to_remove": sorted(redundant_to_remove),
            "total_strong": len(all_remove),
            "total_weak": len(all_remove_weak),
            "details_strong": [r for r in final_ranking if r["feature"] in features_to_remove],
            "details_weak": [r for r in final_ranking if r["feature"] in features_to_remove_weak],
        }, f, indent=2)

    print(f"\n[Saved] {report_path}")
    print(f"[Saved] {ranking_csv}")
    print(f"[Saved] {harmful_path}")

    print("\n" + "=" * 70)
    print(f"ANALYSIS COMPLETE")
    print(f"  Strong removal candidates: {len(all_remove)}")
    print(f"  Weak removal candidates:   {len(all_remove_weak)}")
    print(f"  Redundant to remove:       {len(redundant_to_remove)}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("=" * 70)

    return report


if __name__ == "__main__":
    main()
