"""Script to fix reward system in trading_env.py"""

import re

def apply_fixes(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Change 1: Update constants (lines 24-27)
    content = content.replace(
        "DEPLOYMENT_BONUS_SCALE = 0.5      # Max bonus for full deployment",
        "DEPLOYMENT_REWARD_SCALE = 1.0     # Symmetric: -1.0 to +1.0 (target 50% deployed)"
    )
    content = content.replace(
        "HOLDING_BONUS_MAX = 1.0           # Max bonus for holding positions\n",
        ""
    )
    content = content.replace(
        "WINRATE_BONUS_SCALE = 2.0         # Win rate bonus scaling",
        "WINRATE_REWARD_SCALE = 2.0        # Win rate reward scaling"
    )
    
    # Change 2: Replace WIN RATE section
    content = content.replace(
        """        # --- WIN RATE BONUS ---
        if len(self.trade_history) >= 5:
            recent_trades = self.trade_history[-50:]  # Last 50 trades
            wins = sum(1 for t in recent_trades if t['net_profit'] > 0)
            win_rate = wins / len(recent_trades)
            winrate_bonus = (win_rate - 0.5) * WINRATE_BONUS_SCALE
        else:
            winrate_bonus = 0.0
            win_rate = 0.0""",
        """        # --- WIN RATE REWARD (Symmetric) ---
        if len(self.trade_history) >= 5:
            recent_trades = self.trade_history[-50:]  # Last 50 trades
            wins = sum(1 for t in recent_trades if t['net_profit'] > 0)
            win_rate = wins / len(recent_trades)
            winrate_reward = (win_rate - 0.5) * WINRATE_REWARD_SCALE
        else:
            winrate_reward = 0.0
            win_rate = 0.0"""
    )
    
    # Change 3: Replace HOLDING and DEPLOYMENT section
    old_holding_deployment = """        # --- HOLDING BONUS ---
        active_positions = sum(1 for a in self.assets if self.holdings[a] > 0)
        if active_positions > 0:
            avg_holding = sum(self.position_ages.values()) / active_positions
            # Linear bonus up to max at 16 steps (4 hours at 15min bars)
            holding_bonus = min(HOLDING_BONUS_MAX, avg_holding / 16.0)
        else:
            avg_holding = 0
            holding_bonus = 0.0
        
        # --- DEPLOYMENT BONUS ---
        cash_weight = weights[6]
        deployed_pct = 1.0 - cash_weight
        deployment_bonus = deployed_pct * DEPLOYMENT_BONUS_SCALE"""
    
    new_deployment = """        # --- DEPLOYMENT REWARD (Symmetric) ---
        # Target: 50% deployed. Range: -1.0 (0% deployed) to +1.0 (100% deployed)
        # deployed_pct < 0.5: negative (underutilized capital)
        # deployed_pct = 0.5: zero (neutral)
        # deployed_pct > 0.5: positive (active trading)
        cash_weight = weights[6]
        deployed_pct = 1.0 - cash_weight
        deployment_reward = (deployed_pct - 0.5) * 2.0 * DEPLOYMENT_REWARD_SCALE
        deployment_reward = float(np.clip(deployment_reward, -1.0, 1.0))"""
    
    content = content.replace(old_holding_deployment, new_deployment)
    
    # Change 4: Update final reward composition
    old_reward = """        # --- FINAL REWARD COMPOSITION ---
        final_reward = (
            return_reward +           # Primary: portfolio change
            trade_profit_reward +     # Closed trade P&L
            rr_quality +              # Risk-reward quality
            winrate_bonus +           # Trade win rate
            holding_bonus +           # Patience bonus
            deployment_bonus -        # Encourage capital use
            turnover_penalty -        # Trading cost
            cash_hoarding_penalty -   # Anti-hoarding
            staleness_penalty         # Anti-stagnation
        )"""
    
    new_reward = """        # --- FINAL REWARD COMPOSITION ---
        final_reward = (
            return_reward +           # Primary: portfolio change
            trade_profit_reward +     # Closed trade P&L
            rr_quality +              # Risk-reward quality (symmetric)
            winrate_reward +          # Trade win rate (symmetric)
            deployment_reward -       # Capital utilization (symmetric)
            turnover_penalty -        # Trading cost
            cash_hoarding_penalty -   # Anti-hoarding
            staleness_penalty         # Anti-stagnation
        )"""
    
    content = content.replace(old_reward, new_reward)
    
    # Change 5: Update reward components dict
    old_dict = """        reward_components = {
            'return_reward': return_reward,
            'trade_profit': trade_profit_reward,
            'rr_quality': rr_quality,
            'winrate_bonus': winrate_bonus,
            'holding_bonus': holding_bonus,
            'deployment_bonus': deployment_bonus,
            'turnover_penalty': -turnover_penalty,
            'cash_hoard_penalty': -cash_hoarding_penalty,
            'stale_penalty': -staleness_penalty,
            'total': final_reward
        }"""
    
    new_dict = """        reward_components = {
            'return_reward': return_reward,
            'trade_profit': trade_profit_reward,
            'rr_quality': rr_quality,
            'winrate_reward': winrate_reward,
            'deployment_reward': deployment_reward,
            'turnover_penalty': -turnover_penalty,
            'cash_hoard_penalty': -cash_hoarding_penalty,
            'stale_penalty': -staleness_penalty,
            'total': final_reward
        }"""
    
    content = content.replace(old_dict, new_dict)
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"[OK] Fixed {filepath}")

if __name__ == "__main__":
    apply_fixes("e:/tradingbot/trading_env.py")
    apply_fixes("e:/tradingbot/backtesting/trading_env.py")
    print("\n[SUCCESS] All fixes applied!")
