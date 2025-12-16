import pandas as pd
import numpy as np

# Read monitor.csv
df = pd.read_csv('monitor.csv', skiprows=1, names=['r', 'l', 't'])

print(f"Total episodes: {len(df)}")
print(f"\nReward Statistics:")
print(f"  Mean: {df['r'].mean():.2f}")
print(f"  Std: {df['r'].std():.2f}")
print(f"  Min: {df['r'].min():.2f}")
print(f"  Max: {df['r'].max():.2f}")

# Calculate approximate step number for each episode
# Each episode has length 'l', so cumulative steps would be cumulative sum of 'l'
df['cumulative_steps'] = df['l'].cumsum()

# Find episodes around 1M steps
idx_1m = df[df['cumulative_steps'] <= 1_000_000].index[-1] if len(df[df['cumulative_steps'] <= 1_000_000]) > 0 else 0
print(f"\nEpisode at ~1M steps: {idx_1m}")
print(f"  Cumulative steps: {df.loc[idx_1m, 'cumulative_steps']:.0f}")
print(f"  Reward: {df.loc[idx_1m, 'r']:.2f}")

# Look at rewards before and after 1M steps
window = 50
before_1m = df.iloc[max(0, idx_1m-window):idx_1m]
after_1m = df.iloc[idx_1m:min(len(df), idx_1m+window)]

print(f"\nBefore 1M steps (last {len(before_1m)} episodes):")
print(f"  Mean reward: {before_1m['r'].mean():.2f}")
print(f"  Std reward: {before_1m['r'].std():.2f}")
print(f"  Min/Max: {before_1m['r'].min():.2f} / {before_1m['r'].max():.2f}")

print(f"\nAfter 1M steps (next {len(after_1m)} episodes):")
print(f"  Mean reward: {after_1m['r'].mean():.2f}")
print(f"  Std reward: {after_1m['r'].std():.2f}")
print(f"  Min/Max: {after_1m['r'].min():.2f} / {after_1m['r'].max():.2f}")

# Find where the reward transition happens
# Look for sudden jumps in reward
reward_diff = df['r'].diff()
large_jumps = reward_diff.abs() > 50
jump_indices = df[large_jumps].index

print(f"\nLarge reward jumps (>50): {len(jump_indices)} found")
if len(jump_indices) > 0:
    print(f"First large jump at episode {jump_indices[0]}")
    print(f"  Cumulative steps: {df.loc[jump_indices[0], 'cumulative_steps']:.0f}")
    print(f"  Reward before: {df.loc[jump_indices[0]-1, 'r']:.2f}")
    print(f"  Reward after: {df.loc[jump_indices[0], 'r']:.2f}")
    print(f"  Change: {reward_diff.loc[jump_indices[0]]:.2f}")
