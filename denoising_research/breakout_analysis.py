import pandas as pd
import numpy as np
import logging
from denoising_research.data_utils import load_research_data
from denoising_research.pipelines import apply_kalman
from denoising_research.regimes import classify_regimes_sophisticated

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_breakout_preservation():
    logger.info("Analyzing Signal Preservation in Breakout Regimes...")
    raw_df = load_research_data(nrows=50000)

    Q, R = 1e-4, 1e-4
    kalman_df = apply_kalman(raw_df, Q=Q, R=R)

    regimes = classify_regimes_sophisticated(raw_df)
    breakout_mask = regimes == 3

    if breakout_mask.sum() == 0:
        logger.warning("No breakout regimes detected in this sample.")
        return

    raw_rets = raw_df['close'].pct_change()
    kalman_rets = kalman_df['close'].pct_change()

    # Measure lag and directional accuracy preservation in breakouts
    raw_dir = np.sign(raw_rets[breakout_mask])
    kalman_dir = np.sign(kalman_rets[breakout_mask])

    dir_match = (raw_dir == kalman_dir).mean()

    # Check if Kalman dampens the big moves too much
    raw_magnitude = raw_rets[breakout_mask].abs().mean()
    kalman_magnitude = kalman_rets[breakout_mask].abs().mean()
    magnitude_ratio = kalman_magnitude / (raw_magnitude + 1e-8)

    print(f"\nBreakout Signal Preservation:")
    print(f"Directional Match with Raw: {dir_match:.4f}")
    print(f"Magnitude Preservation Ratio: {magnitude_ratio:.4f}")

    if magnitude_ratio < 0.3:
        print("WARNING: Kalman filter may be destroying breakout signal (excessive dampening).")
    else:
        print("SUCCESS: Breakout signal energy is adequately preserved.")

if __name__ == "__main__":
    analyze_breakout_preservation()
