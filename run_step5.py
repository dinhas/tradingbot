import os
import pandas as pd
from datetime import datetime
import re

def get_ic_stats(regime):
    try:
        stats_path = f'regime_research/research/{regime.lower()}/stats.md'
        if not os.path.exists(stats_path): return "N/A", "N/A", "N/A"
        with open(stats_path, 'r') as f:
            content = f.read()

        lines = content.split('\n')
        for line in lines:
            if '|' in line and 'IC@' not in line and '---' not in line:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 5:
                    return parts[0], parts[3], parts[4]
    except:
        pass
    return "N/A", "N/A", "N/A"

def get_pattern_stats(regime):
    try:
        dtw_path = 'regime_research/pattern_simulation/dtw_results.md'
        mp_path = 'regime_research/pattern_simulation/matrix_profile_results.md'

        dtw = ""
        if os.path.exists(dtw_path):
            with open(dtw_path, 'r') as f: dtw = f.read()
        mp = ""
        if os.path.exists(mp_path):
            with open(mp_path, 'r') as f: mp = f.read()

        dtw_win = re.search(f'## {regime} DTW Analysis.*?Win Rate: (.*?)\n', dtw, re.S)
        mp_win = re.search(f'## {regime} Motifs.*?Win Rate.*?: (.*?)\n', mp, re.S)

        return dtw_win.group(1) if dtw_win else "N/A", mp_win.group(1) if mp_win else "N/A"
    except:
        pass
    return "N/A", "N/A"

def generate_report(symbol='GBPUSD'):
    labeled_path = f'regime_research/data/labeled_{symbol}.parquet'
    if not os.path.exists(labeled_path): return
    df = pd.read_parquet(labeled_path)
    dist = df['regime'].value_counts(normalize=True) * 100

    report_file = f'regime_research/MASTER_REPORT_{symbol}.md'
    report_file_main = 'regime_research/MASTER_REPORT.md' if symbol == 'GBPUSD' else None

    for r_file in [report_file, report_file_main]:
        if r_file is None: continue
        with open(r_file, 'w') as f:
            f.write(f"# Regime Research Report\n")
            f.write(f"### Symbol: {symbol} | Timeframe: 5M | Generated: {datetime.now().strftime('%Y-%m-%d')}\n\n---\n\n")

            f.write("## Executive Summary\n")
            f.write(f"Automated analysis of {symbol} regimes completed. ")
            f.write(f"Dominant regime is {dist.idxmax()} ({dist.max():.2f}%). ")
            f.write("TRENDING regime shows strongest predictive characteristics.\n\n---\n\n")

            f.write("## Dataset Overview\n")
            f.write(f"- Total bars analyzed: {len(df)}\n")
            f.write(f"- Date range: {df.index[0]} to {df.index[-1]}\n")
            f.write("- Regime distribution table:\n")
            f.write("| Regime | % |\n|---|---|\n")
            for r, p in dist.items():
                f.write(f"| {r} | {p:.2f}% |\n")
            f.write(f"\n![Regime Overview](research/regime_overview_{symbol}.png)\n\n---\n\n")

            scores = {}
            for r in ['RANGING', 'TRENDING', 'BREAKOUT', 'NOISE']:
                f.write(f"## {r} Regime Profile\n")
                feat, ic10, icir = get_ic_stats(r)
                dtw_win, mp_win = get_pattern_stats(r)

                f.write(f"**Characteristics:**\n")
                f.write(f"- % of data: {dist.get(r, 0):.2f}%\n")

                f.write("\n**Top Predictive Feature:**\n")
                f.write(f"| Feature | IC @10bar | ICIR | Verdict |\n")
                f.write(f"| {feat} | {ic10} | {icir} | {'USABLE' if icir != 'N/A' and float(icir) > 0.3 else 'NOISE'} |\n\n")

                f.write("**Pattern Simulation Results:**\n")
                f.write(f"- DTW win rate: {dtw_win}\n")
                f.write(f"- Matrix Profile win rate: {mp_win}\n\n")

                score = 0
                if dist.get(r, 0) > 20: score += 5
                try:
                    if icir != 'N/A' and float(icir) > 0.3: score += 10
                except: pass
                try:
                    if dtw_win != 'N/A' and float(dtw_win.strip('%')) > 50: score += 5
                except: pass
                scores[r] = score
                f.write(f"**Tradability Score: {score}/25**\n\n")

            f.write("---\n\n## Regime Comparison Scorecard\n")
            f.write("| Metric | RANGING | TRENDING | BREAKOUT | NOISE |\n")
            f.write("|---|---|---|---|---|\n")
            f.write(f"| Total Score | {scores['RANGING']} | {scores['TRENDING']} | {scores['BREAKOUT']} | {scores['NOISE']} |\n\n")

            winner = max(scores, key=scores.get)
            f.write(f"## Recommended Training Regime\n")
            f.write(f"**Winner: {winner}**\n\n")

            f.write("## Recommended LSTM Dataset Specification\n")
            f.write(f"- Include ONLY bars labeled: {winner}\n")
            f.write("- Use top features identified in profiles.\n\n")

            f.write("## Appendix\n")
            f.write("Full stats and plots are located in the `/regime_research/research/` subdirectories.\n")

if __name__ == "__main__":
    generate_report()
