import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Seaborn not found. Using matplotlib styles.")
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import TradingEnv from Alpha
# We need to make sure feature_engine can be imported by TradingEnv
sys.path.append(str(project_root / "Alpha" / "src"))

try:
    from Alpha.src.trading_env import TradingEnv
except ImportError:
    print("Could not import TradingEnv via module path. Trying direct import...")
    sys.path.append(str(project_root / "Alpha" / "src"))
    from trading_env import TradingEnv

def visualize_model():
    # Paths
    model_path = project_root / "models/checkpoints/ppo_final_model.zip"
    vec_norm_path = project_root / "models/checkpoints/ppo_final_vecnormalize.pkl"
    output_dir = project_root / "visualizing" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from {model_path}...")
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    # 1. Policy & Value Networks
    print("\n" + "="*50)
    print("1️⃣ Policy & Value Networks (Architecture Check)")
    print("="*50)
    
    # Load model (initially without env to inspect structure)
    model = PPO.load(model_path, device='cpu')
    print("Policy Network Architecture:")
    print(model.policy)
    
    # Check depth
    print("\nAnalysis:")
    if hasattr(model.policy, 'mlp_extractor'):
        print(f"Policy Net (Actor): {model.policy.mlp_extractor.policy_net}")
        print(f"Value Net (Critic): {model.policy.mlp_extractor.value_net}")
    else:
        print("Could not access mlp_extractor (custom policy?)")

    # 2. TensorBoard (Guidance)
    print("\n" + "="*50)
    print("2️⃣ TensorBoard (Guidance)")
    print("="*50)
    print("To view training metrics, run the following command in your terminal:")
    print(f"tensorboard --logdir={project_root / 'logs'}")
    print("Look for: explained_variance (<0.2 is bad), value_loss (should decrease), approx_kl (spikes are bad).")

    # Setup Environment for Rollouts
    print("\nSetting up environment for data collection...")
    data_dir = project_root / "data"
    
    # Create env
    env = DummyVecEnv([lambda: TradingEnv(data_dir=str(data_dir), is_training=False)])
    
    # Load VecNormalize if exists
    if vec_norm_path.exists():
        print(f"Loading VecNormalize stats from {vec_norm_path}")
        env = VecNormalize.load(str(vec_norm_path), env)
        env.training = False # Do not update stats during visualization
        env.norm_reward = False # We want to see real rewards mostly, but for advantage calc SB3 uses internal rewards
    else:
        print("Warning: VecNormalize file not found. Using raw environment.")

    model.set_env(env)
    
    # 3. Advantage Distribution
    print("\n" + "="*50)
    print("3️⃣ Advantage Distribution")
    print("="*50)
    
    print("Collecting rollout (2048 steps)...")
    # We need to fill the rollout buffer. 
    # PPO uses model.rollout_buffer. 
    # We can use model.collect_rollouts to fill it.
    
    # Ensure rollout buffer is reset
    model.rollout_buffer.reset()
    
    # Ensure model has last observation
    if model._last_obs is None:
        print("Initializing model._last_obs...")
        model._last_obs = env.reset()
    
    # Collect rollouts
    # Note: collect_rollouts is an internal method, but standard in SB3 PPO execution
    if hasattr(model, 'collect_rollouts'):
        # We need a callback object
        callback = model._init_callback(None)
        callback.on_training_start(locals(), globals())
        
        model.collect_rollouts(env, callback=callback, rollout_buffer=model.rollout_buffer, n_rollout_steps=2048)
    else:
        print("Error: Could not access collect_rollouts method.")
        return

    # Extract advantages
    rollout = model.rollout_buffer
    if rollout.advantages is None:
        # Advantages are usually computed at the end of collect_rollouts in SB3 PPO?
        # Actually, they are computed in `learn` usually? 
        # Wait, collect_rollouts calls `compute_returns_and_advantage` at the end usually.
        # Let's check if they are populated.
        pass
        
    def to_numpy(data):
        if torch.is_tensor(data):
            return data.cpu().numpy()
        return data

    advantages = to_numpy(rollout.advantages).flatten()
    
    plt.figure(figsize=(10, 6))
    if HAS_SEABORN:
        sns.histplot(advantages, kde=True, bins=50)
    else:
        plt.hist(advantages, bins=50, alpha=0.7, density=True)
        plt.grid(True, alpha=0.3)
        
    plt.title("Advantage Distribution")
    plt.xlabel("Advantage")
    plt.ylabel("Frequency")
    plt.axvline(0, color='r', linestyle='--')
    plt.grid(True, alpha=0.3)
    save_path = output_dir / "advantage_distribution.png"
    plt.savefig(save_path)
    print(f"Saved advantage distribution to {save_path}")
    
    # Analyze advantages
    mean_adv = np.mean(advantages)
    std_adv = np.std(advantages)
    print(f"Advantage Mean: {mean_adv:.4f} (Should be near 0)")
    print(f"Advantage Std:  {std_adv:.4f}")
    if mean_adv > 0.1:
        print("⚠️ Warning: Mostly positive advantages -> Critic might be underestimating value.")
    elif mean_adv < -0.1:
        print("⚠️ Warning: Mostly negative advantages -> Critic might be overestimating value.")
        
    # 4. Action Probability Heatmaps
    print("\n" + "="*50)
    print("4️⃣ Action Probability & Analysis")
    print("="*50)
    
    # Get observations from buffer
    # Observations in buffer might be normalized if VecNormalize is used
    # Shape is (n_steps, n_envs, obs_shape)
    obs_raw = rollout.observations
    obs_flattened = obs_raw.reshape(-1, *obs_raw.shape[2:])
    
    # Convert to torch tensor
    obs_tensor, _ = model.policy.obs_to_tensor(obs_flattened)
    
    # Get action distribution
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
        # For Continuous actions (Box), usually mean and std
        if hasattr(dist, 'distribution'):
             # Torch distribution
             action_means = dist.distribution.mean.cpu().numpy().flatten()
             action_stds = dist.distribution.stddev.cpu().numpy().flatten()
        else:
             # Categorical or other
             action_means = dist.mode().cpu().numpy().flatten() # Fallback
             action_stds = np.zeros_like(action_means)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    if HAS_SEABORN:
        sns.histplot(action_means, bins=50, kde=True)
    else:
        plt.hist(action_means, bins=50, alpha=0.7, density=True)
    
    plt.title("Action Mean Distribution")
    plt.xlabel("Action Mean (Direction)")
    
    plt.subplot(1, 2, 2)
    # Plot Action vs "Market Volatility" if we can extract it.
    # Feature 131 is market_volatility in the env (index 131 in 140-dim, but obs is 40-dim).
    # In `_get_observation`:
    # 40 features = [25 asset features] + [15 global features]
    # Global features start at index 25.
    # "market_volatility" is 7th in global list -> index 25 + 6 = 31?
    # Global feats: equity, margin, drawdown, pos_count, risk, dispersion, vol...
    # 0: equity, 1: margin, 2: drawdown, 3: pos, 4: risk, 5: disp, 6: vol.
    # So index 31 (0-based relative to global start? No, 25+6=31).
    
    # We need to know if obs are normalized. If VecNormalize is on, they are.
    # It's hard to interpret normalized features directly.
    # But we can look at action vs time/step
    plt.plot(action_means[:200], label="Action Mean")
    plt.title("Action Mean (First 200 Steps)")
    plt.xlabel("Step")
    plt.ylabel("Action")
    plt.legend()
    
    plt.tight_layout()
    save_path = output_dir / "action_analysis.png"
    plt.savefig(save_path)
    print(f"Saved action analysis to {save_path}")
    
    if np.std(action_means) < 0.05:
         print("⚠️ Warning: Action collapse? Very low standard deviation in actions.")

    # 5. Critic Sanity Check
    print("\n" + "="*50)
    print("5️⃣ Critic Sanity Check (Values vs Returns)")
    print("="*50)
    
    values = to_numpy(rollout.values).flatten()
    returns = to_numpy(rollout.returns).flatten() # Returns are computed as advantages + values usually
    
    plt.figure(figsize=(8, 8))
    plt.scatter(returns, values, alpha=0.1)
    
    # Diagonal line
    min_val = min(np.min(returns), np.min(values))
    max_val = max(np.max(returns), np.max(values))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title("Critic Sanity: Values vs Returns")
    plt.xlabel("Actual Returns (Discounted)")
    plt.ylabel("Predicted Values")
    plt.grid(True)
    
    save_path = output_dir / "critic_sanity.png"
    plt.savefig(save_path)
    print(f"Saved critic sanity plot to {save_path}")
    
    # Correlation
    correlation = np.corrcoef(returns, values)[0, 1]
    print(f"Correlation (Returns vs Values): {correlation:.4f}")
    if correlation < 0.5:
         print("⚠️ Warning: Low correlation. Critic might be hallucinating.")
    
    # Bonus: Latent Space PCA
    print("\n" + "="*50)
    print("🧠 Bonus: Latent Space Analysis")
    print("="*50)
    
    try:
        from sklearn.decomposition import PCA
        
        # Extract latent features
        with torch.no_grad():
            # For MlpPolicy, features are extracted then passed to mlp_extractor
            # We want the output of mlp_extractor.policy_net or shared net?
            # Usually extract_features gives the features from the feature extractor (CNN/MLP)
            # Then passed to mlp_extractor.
            
            features = model.policy.extract_features(obs_tensor)
            latent_pi, latent_vf = model.policy.mlp_extractor(features)
            
            # Use latent_pi (policy latent space)
            latent_data = latent_pi.cpu().numpy()
            
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(latent_data)
        
        plt.figure(figsize=(10, 8))
        # Color by value/reward or action
        sc = plt.scatter(reduced[:, 0], reduced[:, 1], c=action_means, cmap='viridis', alpha=0.5)
        plt.colorbar(sc, label='Mean Action')
        plt.title("Latent Space (PCA) - Colored by Action")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        
        save_path = output_dir / "latent_space_pca.png"
        plt.savefig(save_path)
        print(f"Saved latent space PCA to {save_path}")
        print("If clusters form, the model sees distinct regimes. If soup, it might be noise.")
        
    except ImportError:
        print("sklearn not installed. Skipping PCA.")
    except Exception as e:
        print(f"Error in PCA analysis: {e}")

    print("\n" + "="*50)
    print(f"✅ Visualization complete. Check {output_dir}")

if __name__ == "__main__":
    visualize_model()
