# Quick Start Guide - Multi-Asset Trading AI

## 🚀 Running the Complete Pipeline

### Option 1: Automated (Recommended)
```bash
python main.py
```

This runs everything automatically:
1. ✅ Data fetching (if needed)
2. ✅ Environment validation
3. ✅ Training with fixes applied
4. ✅ Report generation

### Option 2: Manual Steps
```bash
# Step 1: Fetch data (only if missing)
python ctradercervice.py

# Step 2: Test environment
python test_env.py

# Step 3: Train model
python train.py
```

---

## 📊 Monitoring Training

### Watch TensorBoard (in separate terminal)
```bash
tensorboard --logdir ./logs/
```
Then open: http://localhost:6006

### Key Metrics to Watch

**First 100k steps - Check if learning started:**
- ✅ `value_loss`: Should DROP from billions → **< 1000**
- ✅ `explained_variance`: Should RISE to **0.3-0.5**
- ✅ `clip_fraction`: Should be **0.05-0.15**

**If these don't improve in first 2 hours, training is broken!**

---

## 📁 Output Files

After completion (~24 hours):

```
./models/
  ├── recurrent_ppo_trading_50000_steps.zip    # Checkpoint at 50k
  ├── recurrent_ppo_trading_100000_steps.zip   # Checkpoint at 100k
  ├── ...
  ├── final_model.zip                          # Final trained model
  └── final_model_vecnormalize.pkl             # Reward normalization stats

./models/best/
  └── best_model.zip                           # Best performing checkpoint

./logs/
  ├── RecurrentPPO_1/                          # TensorBoard logs
  └── eval/                                    # Evaluation logs

pipeline_YYYYMMDD_HHMMSS.log                   # Full execution log
training_report_YYYYMMDD_HHMMSS.json           # Summary report
```

---

## 🔍 Critical Fixes Applied

### Fix 1: Reward Scaling (in trading_env.py)
**Before:** `final_reward = normalized_reward * 100`  
**After:** `final_reward = normalized_reward * 10`

**Why:** Prevents value function explosion (was seeing billions)

### Fix 2: VecNormalize (in main.py & train.py)
```python
vec_env = VecNormalize(vec_env, norm_reward=True, clip_reward=10.0)
```

**Why:** Stabilizes reward distribution across episodes

---

## ⚠️ Troubleshooting

### Training stalls (value_loss stays huge)
```bash
# Stop training (Ctrl+C)
# Verify reward range
python -c "from trading_env import TradingEnv; env = TradingEnv(); obs = env.reset()[0]; print(env.step(env.action_space.sample()))"
```

### Out of memory
Reduce parallel environments in `main.py`:
```python
n_envs = 4  # Change from 8 to 4
```

### GPU not detected
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📧 Notifications (Optional)

Add to `main.py` → `send_notification()`:

### Discord Webhook
```python
import requests
DISCORD_WEBHOOK = "your_webhook_url"
requests.post(DISCORD_WEBHOOK, json={"content": message})
```

### Email
```python
import smtplib
# Add your SMTP config
```

---

## 🎯 Expected Timeline

| Phase | Duration | Action |
|-------|----------|--------|
| Data Fetch | 30-60 min | Can skip if files exist |
| Env Test | 1 min | Quick validation |
| Training | ~24 hours | Go sleep! |
| Report | 1 min | Auto-generated |

**Total:** ~25 hours (mostly unattended)

---

## 📈 Success Criteria

After 2M steps, you should see:
- ✅ Portfolio value trending upward
- ✅ Sharpe ratio > 0.5
- ✅ Max drawdown < 30%
- ✅ Model uses cash position (not always 100% invested)

If not → Hyperparameter tuning needed (see RPD v3.0)

---

## 🆘 Emergency Stop

```bash
# Press Ctrl+C in terminal
# Latest checkpoint saved in ./models/
# Resume with: model = RecurrentPPO.load("./models/recurrent_ppo_trading_XXXXX_steps.zip")
```
