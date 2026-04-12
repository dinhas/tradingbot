"""
Pipeline diagnostic — run after fixes to verify:
1. features_full.npy exists and is consecutive (no gaps)
2. labels.npz contains full_idx
3. AlphaDataset windows are real consecutive M5 bars
4. Label distribution is balanced enough to learn from
5. Window shape is correct for LSTM
"""
import os, sys
import numpy as np
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "training_set")
FULL_PATH   = os.path.join(DATASET_DIR, "features_full.npy")
LABELS_PATH = os.path.join(DATASET_DIR, "labels.npz")
SEQ_LEN     = 50
PASS = "✅ PASS"
FAIL = "❌ FAIL"

print("\n========== PIPELINE DIAGNOSTIC ==========\n")
all_passed = True

# ---- TEST 1: files exist ----
print("TEST 1 — Required files exist")
for path, name in [(FULL_PATH, "features_full.npy"), (LABELS_PATH, "labels.npz")]:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1e6
        print(f"  {PASS}  {name}  ({size_mb:.1f} MB)")
    else:
        print(f"  {FAIL}  {name} NOT FOUND — regenerate dataset")
        all_passed = False

if not all_passed:
    print("\nStop here — regenerate dataset before continuing.")
    sys.exit(1)

# ---- TEST 2: features_full row count is plausible ----
print("\nTEST 2 — features_full.npy row count (expect ~650K+ rows)")
features_full = np.load(FULL_PATH, mmap_mode='r')
n_rows, n_cols = features_full.shape
print(f"  Shape: {features_full.shape}")
if n_rows > 300_000:
    print(f"  {PASS}  {n_rows:,} rows — full bar matrix confirmed")
else:
    print(f"  {FAIL}  Only {n_rows:,} rows — this looks like the sparse labeled file, not the full matrix")
    all_passed = False

# ---- TEST 3: labels.npz has full_idx and asset_idx ----
print("\nTEST 3 — labels.npz contains full_idx and asset_idx")
labels = np.load(LABELS_PATH)
keys = list(labels.keys())
print(f"  Keys found: {keys}")
if 'full_idx' in keys and 'asset_idx' in keys:
    print(f"  {PASS}  Indices present — {len(labels['full_idx']):,} entries")
else:
    print(f"  {FAIL}  Indices missing — pipeline fix was not applied")
    all_passed = False

# ---- TEST 4: full_idx values point into features_full correctly ----
print("\nTEST 4 — full_idx values are valid (in range, >= SEQ_LEN)")
full_idx = labels['full_idx']
out_of_range = np.sum(full_idx >= n_rows)
below_seqlen = np.sum(full_idx < SEQ_LEN)
print(f"  full_idx range: {full_idx.min()} → {full_idx.max()}")
print(f"  Out of range indices: {out_of_range}")
print(f"  Below SEQ_LEN ({SEQ_LEN}): {below_seqlen}")
if out_of_range == 0 and below_seqlen < 100:
    print(f"  {PASS}  Indices look valid")
else:
    print(f"  {FAIL}  Bad indices detected")
    all_passed = False

# ---- TEST 5: window is real consecutive bars (the core gappy window test) ----
print("\nTEST 5 — Window consecutive bar check (THE KEY TEST)")
print("  Sampling 5 windows and checking they are consecutive rows in features_full...")
sample_label_positions = [100, 500, 1100, 2100, 4100] # Smoke test has 5000 samples (1000 per asset)
sample_label_positions = [p for p in sample_label_positions if p < len(full_idx)]
gap_bug_detected = False

asset_idx = labels['asset_idx']

for label_pos in sample_label_positions:
    full_row = full_idx[label_pos]
    a_idx    = asset_idx[label_pos]

    start_col = a_idx * 40
    end_col   = start_col + 40
    window    = features_full[full_row - SEQ_LEN : full_row, start_col : end_col]

    # Check: the rows in the window should be drawn from the dense matrix.
    if len(window) != SEQ_LEN:
        print(f"  {FAIL}  label_pos={label_pos}: window has {len(window)} rows, expected {SEQ_LEN}")
        gap_bug_detected = True
    elif window.shape[1] != 40:
        print(f"  {FAIL}  label_pos={label_pos}: window has {window.shape[1]} cols, expected 40")
        gap_bug_detected = True
    else:
        # full_row should be >> pos_in_asset (the index in the sparse labels for this asset)
        # Calculate how many labels for THIS asset preceded this one
        pos_in_asset = np.sum(asset_idx[:label_pos] == a_idx)

        density_ratio = full_row / (pos_in_asset + 1)
        print(f"  label_pos={label_pos:>6}  full_row={full_row:>7}  asset_idx={a_idx}  density_ratio={density_ratio:.1f}x  rows={len(window)}")

if not gap_bug_detected and len(sample_label_positions) > 0:
    # If density_ratio is ~10, it means full_row >> pos_in_asset — correct (full matrix is ~10x denser)
    # We check a sample that is far enough into the asset data
    last_label_pos = sample_label_positions[-1]
    a_idx = asset_idx[last_label_pos]
    pos_in_asset = np.sum(asset_idx[:last_label_pos] == a_idx)
    ratio = full_idx[last_label_pos] / (pos_in_asset + 1)

    if ratio > 3.0:
        print(f"  {PASS}  density_ratio > 3x confirms windows are from full bar matrix, not sparse labels")
    else:
        print(f"  {FAIL}  density_ratio ≈ 1x — windows are still drawn from sparse labeled file (gappy bug still present)")
        all_passed = False

# ---- TEST 6: label distribution ----
print("\nTEST 6 — Direction label distribution (expect roughly 25-40% each side)")
directions = labels['direction']
unique, counts = np.unique(directions, return_counts=True)
total = len(directions)
dist = {int(k): round(v/total*100, 1) for k,v in zip(unique, counts)}
print(f"  Total labels: {total:,}")
print(f"  Short (-1): {dist.get(-1, 0)}%")
print(f"  Flat  ( 0): {dist.get(0, 0)}%")
print(f"  Long  (+1): {dist.get(1, 0)}%")

flat_pct = dist.get(0, 100)
if flat_pct < 60:
    print(f"  {PASS}  Flat={flat_pct}% — distribution is learnable")
else:
    print(f"  {FAIL}  Flat={flat_pct}% — label collapse. Lower upper_mult in Labeler (currently should be 2.0)")
    all_passed = False

# ---- TEST 7: AlphaDataset end-to-end tensor shape ----
print("\nTEST 7 — AlphaDataset tensor shape check")
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from Alpha.run_pipeline import AlphaDataset
    ds = AlphaDataset(FULL_PATH, LABELS_PATH, seq_len=SEQ_LEN)
    sample = ds[0]
    x, d, q, m = sample
    print(f"  x shape:  {x.shape}   (expect torch.Size([{SEQ_LEN}, 40]))")
    print(f"  dir:      {d.item():.0f}")
    print(f"  quality:  {q.item():.4f}")
    print(f"  meta:     {m.item():.0f}")
    if x.shape == torch.Size([SEQ_LEN, 40]):
        print(f"  {PASS}  Tensor shape correct — LSTM-ready")
    else:
        print(f"  {FAIL}  Wrong shape. Got {x.shape}, expected [{SEQ_LEN}, 40]")
        all_passed = False
except Exception as e:
    print(f"  {FAIL}  AlphaDataset raised: {e}")
    all_passed = False

# ---- FINAL VERDICT ----
print("\n=========================================")
if all_passed:
    print("ALL TESTS PASSED — pipeline is clean. Safe to retrain.")
else:
    print("SOME TESTS FAILED — do not retrain until all pass.")
print("=========================================\n")
