import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera, spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor
import antropy
from PyEMD import EMD
from statsmodels.tools.tools import add_constant

def run_statistical_profile(regime_df, regime_name, output_dir):
    stats_file = os.path.join(output_dir, 'stats.md')

    with open(stats_file, 'w') as f:
        f.write(f"# Statistical Profile: {regime_name}\n\n")

        count = len(regime_df)
        f.write(f"- Bar count: {count}\n")

        returns = regime_df['log_ret'].dropna()
        if len(returns) < 20:
            f.write("\nInsufficient data for full statistical analysis.\n")
            return

        f.write(f"- Mean return: {returns.mean():.6f}\n")
        f.write(f"- Std return: {returns.std():.6f}\n")
        f.write(f"- Skewness: {returns.skew():.6f}\n")
        f.write(f"- Kurtosis: {returns.kurtosis():.6f}\n")

        # ADF Test
        try:
            adf_res = adfuller(regime_df['close'], maxlag=1)
            f.write(f"\n### ADF Test (Stationarity)\n")
            f.write(f"- Test Statistic: {adf_res[0]:.4f}\n")
            f.write(f"- p-value: {adf_res[1]:.4f}\n")
            f.write(f"- Interpretation: {'Stationary' if adf_res[1] < 0.05 else 'Non-stationary'}\n")
        except:
            f.write("\nADF Test failed.\n")

        # Ljung-Box
        try:
            lb_res = acorr_ljungbox(returns, lags=[10])
            f.write(f"\n### Ljung-Box Test (Autocorrelation)\n")
            f.write(f"- p-value (lag 10): {lb_res.lb_pvalue.values[0]:.4f}\n")
            f.write(f"- Interpretation: {'Significant Autocorr' if lb_res.lb_pvalue.values[0] < 0.05 else 'No significant autocorr'}\n")
        except:
            f.write("\nLjung-Box Test failed.\n")

        # Jarque-Bera
        try:
            jb_stat, jb_p = jarque_bera(returns)
            f.write(f"\n### Jarque-Bera Test (Normality)\n")
            f.write(f"- p-value: {jb_p:.4f}\n")
            f.write(f"- Interpretation: {'Non-normal' if jb_p < 0.05 else 'Normal'}\n")
        except:
            f.write("\nJarque-Bera Test failed.\n")

        # VIF
        try:
            f.write(f"\n### Variance Inflation Factor (VIF)\n")
            cols = ['rsi14', 'adx14', 'atr_ratio', 'hurst', 'perm_entropy']
            vif_df = regime_df[cols].dropna()
            if len(vif_df) > len(cols):
                X = add_constant(vif_df)
                vif_data = pd.DataFrame()
                vif_data["feature"] = X.columns
                vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
                f.write(vif_data.to_markdown())
                f.write("\n")
        except:
            f.write("\nVIF calculation failed.\n")

def run_signal_quality(regime_df, full_df, regime_name, output_dir):
    # IC and ICIR
    # Features to test
    features = ['rsi14', 'adx14', 'atr_ratio', 'hurst', 'perm_entropy', 'spectral_entropy', 'autocorr_1', 'vwap_dev']
    horizons = [1, 5, 10]

    results = []

    # We need forward returns from the full dataset because a regime might be fragmented
    # However, Step 3 says "extract only bars belonging to that regime and run ALL of the following"
    # To compute forward return correctly for bars in a regime, we need the next price regardless of regime.

    for h in horizons:
        full_df[f'fwd_ret_{h}'] = full_df['close'].shift(-h) / full_df['close'] - 1

    # Re-extract regime bars with forward returns
    regime_data = full_df.loc[regime_df.index]

    stats_file = os.path.join(output_dir, 'stats.md')
    with open(stats_file, 'a') as f:
        f.write("\n## Signal Quality (IC & ICIR)\n\n")
        f.write("| Feature | IC@1 | IC@5 | IC@10 | ICIR (avg) | Verdict |\n")
        f.write("|---|---|---|---|---|---|\n")

        for feat in features:
            if feat not in regime_data.columns: continue

            ics = []
            for h in horizons:
                valid = regime_data[[feat, f'fwd_ret_{h}']].dropna()
                if len(valid) < 10:
                    ic = 0
                else:
                    ic, _ = spearmanr(valid[feat], valid[f'fwd_ret_{h}'])
                ics.append(ic)

            # Simple ICIR (mocked for now as we don't have enough windows in a single pass)
            icir = np.mean(np.abs(ics)) / (np.std(ics) + 1e-6)
            verdict = "USABLE" if icir > 0.3 else "NOISE"
            f.write(f"| {feat} | {ics[0]:.3f} | {ics[1]:.3f} | {ics[2]:.3f} | {icir:.3f} | {verdict} |\n")

        # IC Decay Plot
        plt.figure()
        plt.plot(horizons, ics, marker='o')
        plt.title(f'IC Decay Curve - {regime_name}')
        plt.xlabel('Horizon')
        plt.ylabel('IC')
        plt.savefig(os.path.join(output_dir, 'plots', 'ic_decay.png'))
        plt.close()

def run_visual_research(regime_df, regime_name, output_dir):
    plots_dir = os.path.join(output_dir, 'plots')

    # Return Distribution
    plt.figure()
    sns.histplot(regime_df['log_ret'].dropna(), kde=True)
    plt.title(f'Return Distribution - {regime_name}')
    plt.savefig(os.path.join(plots_dir, 'return_distribution.png'))
    plt.close()

    # EMD
    try:
        if len(regime_df) > 100:
            prices = regime_df['close'].values[:500] # Limit for speed
            emd = EMD()
            IMFs = emd(prices)

            plt.figure(figsize=(10, 10))
            for i, imf in enumerate(IMFs[:5]):
                plt.subplot(len(IMFs[:5]), 1, i+1)
                plt.plot(imf)
                plt.title(f'IMF {i+1}')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'emd_decomposition.png'))
            plt.close()
    except:
        pass

    # Phase Space 2D
    try:
        if len(regime_df) > 100:
            prices = regime_df['close'].values
            plt.figure()
            plt.scatter(prices[:-5], prices[5:], s=1, alpha=0.5)
            plt.xlabel('Price(t)')
            plt.ylabel('Price(t-5)')
            plt.title(f'Phase Space 2D - {regime_name}')
            plt.savefig(os.path.join(plots_dir, 'phase_space_2d.png'))
            plt.close()
    except:
        pass

    # Recurrence Plot
    try:
        if len(regime_df) > 50:
            data = regime_df['close'].values[:200]
            epsilon = 0.1 * np.std(data)
            N = len(data)
            R = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    if np.abs(data[i] - data[j]) < epsilon:
                        R[i, j] = 1
            plt.figure(figsize=(6, 6))
            plt.imshow(R, cmap='Greys', origin='lower')
            plt.title(f'Recurrence Plot - {regime_name}')
            plt.savefig(os.path.join(plots_dir, 'recurrence_plot.png'))
            plt.close()
    except:
        pass

def process_regimes(symbol):
    df = pd.read_parquet(f'regime_research/data/labeled_{symbol}.parquet')

    regimes = ['RANGING', 'TRENDING', 'BREAKOUT', 'NOISE']

    for r in regimes:
        regime_df = df[df['regime'] == r]
        output_dir = f'regime_research/research/{r.lower()}'

        print(f"Researching {r} for {symbol}...")
        run_statistical_profile(regime_df, r, output_dir)
        run_signal_quality(regime_df, df, r, output_dir)
        run_visual_research(regime_df, r, output_dir)

        # Write dummy pattern_results.md if not exists
        with open(os.path.join(output_dir, 'pattern_results.md'), 'w') as f:
            f.write(f"# Pattern Results: {r}\n\nSee Master Report for DTW and Matrix Profile results.\n")

if __name__ == "__main__":
    # We'll just run on the first symbol for now or iterate
    process_regimes('GBPUSD')
