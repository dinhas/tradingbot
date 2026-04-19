import pandas as pd
import numpy as np
from denoising_research.pipelines import apply_kalman

def test_kalman_causality():
    data = pd.DataFrame({'close': np.random.randn(100)})
    result_full = apply_kalman(data)

    # Check if changing future values affects past results
    data_short = data.iloc[:50].copy()
    result_short = apply_kalman(data_short)

    np.testing.assert_array_almost_equal(result_full['close'].iloc[:50], result_short['close'])
    print("Causality test passed!")

if __name__ == "__main__":
    test_kalman_causality()
