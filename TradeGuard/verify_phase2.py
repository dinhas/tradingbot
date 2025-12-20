import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

def verify_phase2():
    print("--- Phase 2 Manual Verification: Feature Engineering ---")
    
    try:
        from generate_dataset import FeatureEngine
    except ImportError as e:
        print(f"Error: Could not import FeatureEngine. {e}")
        return

    fe = FeatureEngine()
    print("1. FeatureEngine initialized successfully.")

    # Mock inputs for Alpha Confidence
    portfolio_state = {
        'equity': 10000,
        'peak_equity': 10500,
        'total_exposure': 4000,
        'open_positions_count': 2,
        'recent_trades': [{'pnl': 100}, {'pnl': -50}, {'pnl': 200}],
        'asset_action_raw': 0.75,
        'asset_recent_actions': [0.5, 0.6, 0.7, 0.8, 0.75],
        'asset_signal_persistence': 3,
        'asset_signal_reversal': 0
    }
    
    alpha_features = fe.calculate_alpha_confidence(None, portfolio_state)
    print(f"2. Alpha Confidence Features (1-10): {alpha_features}")
    if len(alpha_features) == 10:
        print("   - SUCCESS: 10 features calculated.")
    else:
        print(f"   - FAILURE: Expected 10 features, got {len(alpha_features)}")

    # Mock inputs for Risk Output
    risk_params = {
        'sl_mult': 1.5,
        'tp_mult': 2.5,
        'risk_factor': 0.8
    }
    risk_features = fe.calculate_risk_output(risk_params)
    print(f"3. Risk Output Features (sl, tp, risk): {risk_features}")
    if len(risk_features) == 3:
        print("   - SUCCESS: 3 features calculated.")
    else:
        print(f"   - FAILURE: Expected 3 features, got {len(risk_features)}")

    # Mock inputs for News Proxies
    # Create a small dataframe
    data = {
        'open': [1.1000] * 50 + [1.1000],
        'high': [1.1010] * 50 + [1.1050],
        'low': [1.0990] * 50 + [1.0950],
        'close': [1.1005] * 50 + [1.1000],
        'volume': [1000.0] * 50 + [5000.0]
    }
    df = pd.DataFrame(data)
    news_features = fe.calculate_news_proxies(df)
    print(f"4. News Proxy Features (11-20): {[f'{x:.4f}' if isinstance(x, float) else x for x in news_features]}")
    if len(news_features) == 10:
        print("   - SUCCESS: 10 features calculated.")
    else:
        print(f"   - FAILURE: Expected 10 features, got {len(news_features)}")

    print("\nManual Verification Complete.")

if __name__ == "__main__":
    verify_phase2()
