import pandas as pd
import numpy as np

# Load the data
file_path = r'e:\tradingbot\backtest\results\trades_alpha_lstm_20260415_174651.csv'
df = pd.read_csv(file_path)

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour

# Define sessions (UTC)
# Asian: 00:00 - 09:00
# London: 08:00 - 17:00
# New York: 13:00 - 22:00

def get_session(hour):
    sessions = []
    if 0 <= hour < 9:
        sessions.append('Asian')
    if 8 <= hour < 17:
        sessions.append('London')
    if 13 <= hour < 22:
        sessions.append('New York')
    return sessions

df['sessions'] = df['hour'].apply(get_session)

# Flag if it's in any session
df['in_session'] = df['sessions'].apply(lambda x: len(x) > 0)

# Calculate win rates
def calc_win_rate(data):
    if len(data) == 0:
        return 0, 0
    wins = len(data[data['net_pnl'] > 0])
    total = len(data)
    win_rate = (wins / total) * 100
    return win_rate, total

# Comparison 1: Session vs Any Time
win_rate_all, total_all = calc_win_rate(df)
win_rate_session, total_session = calc_win_rate(df[df['in_session']])

print("--- Comparison: Session vs Any Time ---")
print(f"Any Time:   Win Rate: {win_rate_all:.2f}%, Total Trades: {total_all}")
print(f"In Sessions: Win Rate: {win_rate_session:.2f}%, Total Trades: {total_session}")
print("\n")

# Comparison 2: Individual Sessions
segments = ['Asian', 'London', 'New York']
session_stats = []

for session in segments:
    # A trade can be in multiple sessions (overlap), we'll count it if it's in that session
    session_df = df[df['sessions'].apply(lambda x: session in x)]
    wr, total = calc_win_rate(session_df)
    session_stats.append({'Session': session, 'Win Rate (%)': wr, 'Total Trades': total})

print("--- Win Rates by Session ---")
print(pd.DataFrame(session_stats).to_string(index=False))
print("\n")

# Best Pairs per Session
pair_stats = []

for session in segments:
    session_df = df[df['sessions'].apply(lambda x: session in x)]
    pairs = session_df['asset'].unique()
    for pair in pairs:
        pair_df = session_df[session_df['asset'] == pair]
        wr, total = calc_win_rate(pair_df)
        avg_pnl = pair_df['net_pnl'].mean()
        pair_stats.append({
            'Session': session,
            'Pair': pair,
            'Win Rate (%)': wr,
            'Total Trades': total,
            'Avg PNL': avg_pnl
        })

pair_stats_df = pd.DataFrame(pair_stats)
print("--- Best Pairs to Trade inside Specific Sessions ---")
for session in segments:
    print(f"\n[{session} Session]")
    top_pairs = pair_stats_df[pair_stats_df['Session'] == session].sort_values(by='Win Rate (%)', ascending=False)
    print(top_pairs[['Pair', 'Win Rate (%)', 'Total Trades', 'Avg PNL']].to_string(index=False))
