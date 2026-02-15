import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from risk_model_sl import RiskModelSL

# Configuration
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "risk_model_sl_best.pth")
OUTPUT_CHART = os.path.join(os.path.dirname(__file__), "risk_feature_importance.png")

# Define Feature Names
FEATURE_NAMES = [
    # 1. Asset Specific (13)
    "close", "return_1", "return_12", "atr_14", "atr_ratio", 
    "bb_position", "ema_9", "ema_21", "price_vs_ema9", "ema9_vs_ema21",
    "rsi_14", "macd_hist", "volume_ratio",
    
    # 2. Pro Features (11)
    "htf_ema_alignment", "htf_rsi_divergence", "swing_structure_proximity",
    "vwap_deviation", "delta_pressure", "volume_shock", 
    "volatility_squeeze", "wick_rejection_strength", "breakout_velocity",
    "rsi_slope_divergence", "macd_momentum_quality",
    
    # 3. Cross-Asset (5)
    "corr_basket", "rel_strength", "corr_xauusd", "corr_eurusd", "rank",
    
    # 4. Global & Macro (11)
    "market_volatility", "avg_atr_ratio", "asset_dispersion", 
    "adx_14", "return_48", "sma_200_dist", "stoch_rsi",
    "hour_sin", "hour_cos", "day_sin", "day_cos"
]

def analyze_importance():
    print("Analyzing Risk Model Feature Importance...")
    
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Extract weights and determine input dimension from state_dict
    state_dict = torch.load(MODEL_PATH, map_location=device)
    
    # Check for DataParallel wrapping (starts with 'module.')
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Dynamically detect input dimension from weights
    weight_key = 'input_proj.0.weight'
    if weight_key in state_dict:
        detected_dim = state_dict[weight_key].shape[1]
    else:
        print(f"Error: Could not find {weight_key} in state_dict")
        return

    print(f"Detected Model Input Dimension: {detected_dim}")
    
    # Initialize model with detected dimension
    model = RiskModelSL(input_dim=detected_dim)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. Extract Weights from First Layer
    # The first layer projection weights represent the sensitivity of the hidden units to each input
    # Heuristic: sum of absolute weights for each input feature
    weights = model.input_proj[0].weight.data.abs().cpu().numpy() # [hidden_dim, input_dim]
    importance = np.sum(weights, axis=0) # [input_dim]
    
    # Normalize to 100%
    importance = 100 * (importance / np.sum(importance))
    
    # 4. Create DataFrame
    # Handle dimension mismatch gracefully
    if len(FEATURE_NAMES) != len(importance):
        print(f"Warning: Feature names count ({len(FEATURE_NAMES)}) does not match importance vector length ({len(importance)}).")
        if len(importance) == 60:
             print("It seems you are analyzing an OLD 60-feature model with the NEW 40-feature name list.")
             # Pad or truncate for display purposes if possible, or just fail
             display_names = FEATURE_NAMES + [f"Unknown_{i}" for i in range(len(importance) - len(FEATURE_NAMES))]
        else:
             display_names = [f"Feature_{i}" for i in range(len(importance))]
    else:
        display_names = FEATURE_NAMES

    df_imp = pd.DataFrame({
        'Feature': display_names[:len(importance)],
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    # 4. Visualization
    plt.figure(figsize=(12, 16))
    sns.set_style("whitegrid")
    
    # Use a premium color palette
    colors = sns.color_palette("viridis_r", len(df_imp))
    
    ax = sns.barplot(x='Importance', y='Feature', data=df_imp, palette=colors)
    
    # Aesthetics
    plt.title('Risk Model Feature Importance Analysis', fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Importance Score (%)', fontsize=14, fontweight='semibold')
    plt.ylabel('Features', fontsize=14, fontweight='semibold')
    
    # Add percentage labels
    for p in ax.patches:
        width = p.get_width()
        plt.text(width + 0.1, p.get_y() + p.get_height()/2. + 0.1, 
                 '{:1.2f}%'.format(width), 
                 ha="left", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_CHART, dpi=300)
    print(f"Importance chart saved to: {OUTPUT_CHART}")

    # 5. Print Top 20 Features
    print("\n--- Top 20 Most Important Features ---")
    print(df_imp.head(20).to_string(index=False))
    
    # 6. Print Grouped Importance
    print("\n--- Importance by Category ---")
    categories = {
        "Asset Specific": FEATURE_NAMES[:13],
        "Pro Features": FEATURE_NAMES[13:24],
        "Cross-Asset": FEATURE_NAMES[24:29],
        "Global/Macro": FEATURE_NAMES[29:]
    }
    
    for cat_name, feat_list in categories.items():
        cat_imp = df_imp[df_imp['Feature'].isin(feat_list)]['Importance'].sum()
        print(f"{cat_name:15}: {cat_imp:.2f}%")

if __name__ == "__main__":
    analyze_importance()
