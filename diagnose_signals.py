"""Diagnostic: inspect alpha & risk signal distributions to find why no trades execute."""

import os, sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
if not hasattr(np, "_core"):
    sys.modules["numpy._core"] = np.core

import torch
import joblib
from tqdm import tqdm
from stable_baselines3 import PPO
from Alpha.src.trading_env import TradingEnv
from RiskLayer.src.risk_model_sl import RiskModelSL

# ── load env ──────────────────────────────────────────────────────────────────
print("Loading environment (this takes a minute)...")
env = TradingEnv(data_dir="data", stage=1, is_training=False)
assets     = env.assets
num_assets = len(assets)

master_obs     = env.master_obs_matrix        # (N, num_assets * 40)
N, total_dims  = master_obs.shape
obs_flat       = master_obs.reshape(-1, 40)   # (N*num_assets, 40)

print(f"Dataset: {N} steps × {num_assets} assets = {len(obs_flat)} obs vectors")
print(f"obs_flat stats: min={obs_flat.min():.3f}  max={obs_flat.max():.3f}  mean={obs_flat.mean():.3f}")

# ── load alpha model ──────────────────────────────────────────────────────────
alpha_model = PPO.load("models/checkpoints/ppo_final_model.zip")
print(f"\nAlpha model policy: {alpha_model.policy}")
print(f"  action_space : {alpha_model.action_space}")
print(f"  obs_space    : {alpha_model.observation_space}")

# Run a tiny sample to check raw output shape
sample = obs_flat[:16]
sample_actions, _ = alpha_model.predict(sample, deterministic=True)
print(f"\nSample alpha predictions (first 16 obs):")
print(f"  shape  : {sample_actions.shape}")
print(f"  values : {sample_actions.flatten()}")

# ── full alpha batch inference (sample only to save time) ─────────────────────
SAMPLE_SIZE = 50_000
idx = np.random.choice(len(obs_flat), size=SAMPLE_SIZE, replace=False)
sample_obs = obs_flat[idx]

batch_size = 4096
preds = []
for i in range(0, len(sample_obs), batch_size):
    a, _ = alpha_model.predict(sample_obs[i:i+batch_size], deterministic=True)
    preds.append(a)

alpha_sample = np.concatenate(preds, axis=0).squeeze()
print(f"\n── Alpha signal distribution (sample of {SAMPLE_SIZE}) ──────────────────")
print(f"  min={alpha_sample.min():.4f}  max={alpha_sample.max():.4f}  mean={alpha_sample.mean():.4f}  std={alpha_sample.std():.4f}")
thresholds = [0.10, 0.20, 0.33, 0.50]
for t in thresholds:
    pct_long  = (alpha_sample >  t).mean() * 100
    pct_short = (alpha_sample < -t).mean() * 100
    pct_flat  = ((alpha_sample >= -t) & (alpha_sample <= t)).mean() * 100
    print(f"  threshold ±{t:.2f}: LONG={pct_long:.1f}%  SHORT={pct_short:.1f}%  FLAT={pct_flat:.1f}%")

# ── load risk model ───────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
risk_model = RiskModelSL(input_dim=40)
state_dict = torch.load("RiskLayer/models/risk_model_sl_final.pth", map_location=device)
risk_model.load_state_dict(state_dict)
risk_model.to(device)
risk_model.eval()

scaler = joblib.load("RiskLayer/models/sl_risk_scaler.pkl")
risk_obs = scaler.transform(sample_obs).astype(np.float32)

sl_all, tp_all, size_all = [], [], []
with torch.no_grad():
    for i in range(0, len(risk_obs), batch_size):
        t = torch.from_numpy(risk_obs[i:i+batch_size]).to(device)
        p = risk_model(t)
        sl_all.append(p['sl'].cpu().numpy())
        tp_all.append(p['tp'].cpu().numpy())
        size_all.append(p['size'].cpu().numpy())

sl_arr   = np.concatenate(sl_all).squeeze()
tp_arr   = np.concatenate(tp_all).squeeze()
size_arr = np.concatenate(size_all).squeeze()

print(f"\n── Risk model output distribution (sample of {SAMPLE_SIZE}) ─────────────")
print(f"  SL   : min={sl_arr.min():.4f}  max={sl_arr.max():.4f}  mean={sl_arr.mean():.4f}")
print(f"  TP   : min={tp_arr.min():.4f}  max={tp_arr.max():.4f}  mean={tp_arr.mean():.4f}")
print(f"  Size : min={size_arr.min():.4f}  max={size_arr.max():.4f}  mean={size_arr.mean():.4f}")

size_thresholds = [0.0, 0.10, 0.20, 0.30, 0.50]
for t in size_thresholds:
    pct = (size_arr > t).mean() * 100
    print(f"  Size > {t:.2f}: {pct:.1f}% of obs pass")

print("\n── Trade logic simulation ─────────────────────────────────────────────────")
# Simulate what backtest_combined.py does for the first 10000 steps
env.reset()
start_step = env.current_step
end_step   = min(start_step + 10_000, env.max_steps)

# Build a quick alpha + risk matrix for just these rows
step_idx     = np.arange(start_step, end_step)
step_obs_all = master_obs[start_step:end_step]  # (steps, num_assets*40)
step_obs_flat = step_obs_all.reshape(-1, 40)

a_preds = []
for i in range(0, len(step_obs_flat), batch_size):
    a, _ = alpha_model.predict(step_obs_flat[i:i+batch_size], deterministic=True)
    a_preds.append(a)
a_mat = np.concatenate(a_preds, axis=0).squeeze().reshape(len(step_idx), num_assets)

r_obs = scaler.transform(step_obs_flat).astype(np.float32)
sl_l, tp_l, sz_l = [], [], []
with torch.no_grad():
    for i in range(0, len(r_obs), batch_size):
        t2 = torch.from_numpy(r_obs[i:i+batch_size]).to(device)
        p2 = risk_model(t2)
        sl_l.append(p2['sl'].cpu().numpy()); tp_l.append(p2['tp'].cpu().numpy()); sz_l.append(p2['size'].cpu().numpy())

sl_m   = np.concatenate(sl_l).squeeze().reshape(len(step_idx), num_assets)
tp_m   = np.concatenate(tp_l).squeeze().reshape(len(step_idx), num_assets)
size_m = np.concatenate(sz_l).squeeze().reshape(len(step_idx), num_assets)

# Count filter hits
cnt_total = len(step_idx) * num_assets
cnt_dir_zero = 0
cnt_size_fail = 0
cnt_trade = 0

for r in range(len(step_idx)):
    for i in range(num_assets):
        alpha_val = a_mat[r, i]
        direction = 1 if alpha_val > 0.33 else (-1 if alpha_val < -0.33 else 0)
        if direction == 0:
            cnt_dir_zero += 1
            continue
        size_out = size_m[r, i]
        if size_out < 0.30:
            cnt_size_fail += 1
            continue
        cnt_trade += 1

print(f"  Over first {len(step_idx)} steps × {num_assets} assets = {cnt_total} decisions:")
print(f"    Direction == 0 (killed by ±0.33 threshold): {cnt_dir_zero}  ({cnt_dir_zero/cnt_total*100:.1f}%)")
print(f"    Size < 0.30 (killed by size filter)       : {cnt_size_fail}  ({cnt_size_fail/cnt_total*100:.1f}%)")
print(f"    Would open trade                           : {cnt_trade}  ({cnt_trade/cnt_total*100:.1f}%)")
