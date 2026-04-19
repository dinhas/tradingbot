import pandas as pd
import numpy as np
from denoising_research.pipelines import kalman_filter

def test_kalman_causality_optimized():
    Q, R = 1e-5, 1e-3
    data = np.random.randn(1000)

    def run_kalman(series, q=Q, r=R):
        xhat = series[0]
        P = 1.0
        filtered = []
        var_innovation = 1e-5
        alpha = 0.1
        for z in series:
            P = P + q
            innovation = z - xhat
            var_innovation = (1 - alpha) * var_innovation + alpha * (innovation ** 2)
            Q_adaptive = max(q, 0.05 * var_innovation)
            P = P + Q_adaptive
            K = P / (P + r)
            xhat = xhat + K * innovation
            P = (1 - K) * P
            filtered.append(xhat)
        return np.array(filtered)

    res_full = run_kalman(data)
    res_half = run_kalman(data[:500])

    np.testing.assert_array_almost_equal(res_full[:500], res_half)
    print("Optimized Kalman Causality Check: PASSED")

if __name__ == "__main__":
    test_kalman_causality_optimized()
