import os

def apply_new_reward_system(filepath):
    print(f"Processing {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find start and end indices
    start_idx = -1
    end_idx = -1
    
    for i, line in enumerate(lines):
        if "# 5. CALCULATE REWARD" in line:
            # Go back one line to capture the separator
            if i > 0 and "========" in lines[i-1]:
                start_idx = i - 1
            else:
                start_idx = i
        
        if "# 6. Check Termination" in line:
            end_idx = i
            break
    
    if start_idx == -1 or end_idx == -1:
        print(f"❌ Could not find reward block in {filepath}")
        return
    
    print(f"Found reward block from line {start_idx+1} to {end_idx}")
    
    # New code block
    new_code = """        # ================================================================
        # 5. CALCULATE REWARD - FIXED SYSTEM (User Provided)
        # ================================================================
        
        # --- Robust portfolio return ---
        # (PV - PrevPV) / max(PrevPV, 1e-8)
        portfolio_return = (self.portfolio_value - prev_portfolio_value) / max(prev_portfolio_value, 1e-8)
        
        # Use global RETURN_REWARD_SCALE (1000.0)
        bounded_return = np.tanh(portfolio_return * RETURN_REWARD_SCALE / 10.0)
        return_reward = float(bounded_return * 5.0)   # final range approx [-5, +5]
        
        # ----------------------
        # Risk / quality shaping
        # ----------------------
        # rr_ratio = TP/SL
        if self.sl_multiplier > 0:
            rr_ratio = self.tp_multiplier / self.sl_multiplier
        else:
            rr_ratio = 1.0
        
        # scale down big impact
        RR_QUALITY_SCALE_LOCAL = 0.2
        rr_quality = float(np.clip((rr_ratio - 2.0) * RR_QUALITY_SCALE_LOCAL, -1.0, 1.0))
        
        # ------------------------
        # Win rate reward (small)
        # ------------------------
        WINRATE_REWARD_SCALE_LOCAL = 0.5
        total_trades = len(self.trade_history)
        if total_trades > 10:
            wins = sum(1 for t in self.trade_history if t['net_profit'] > 0)
            winrate = wins / max(total_trades, 1)
            winrate_reward = (winrate - 0.5) * WINRATE_REWARD_SCALE_LOCAL
        else:
            winrate_reward = 0.0
            
        # -------------------------
        # Deployment reward (small)
        # -------------------------
        DEPLOYMENT_REWARD_SCALE_LOCAL = 0.5
        cash_weight = weights[6]
        deployment_ratio = 1.0 - cash_weight
        deployment_reward = (deployment_ratio - 0.5) * DEPLOYMENT_REWARD_SCALE_LOCAL  # centered at 0
        
        # -----------------------
        # Turnover penalty (big)
        # -----------------------
        WARMUP_STEPS_LOCAL = 50
        TURNOVER_PENALTY_SCALE_LOCAL = 6.0
        
        turnover = np.sum(np.abs(weights - self.previous_weights)) / 2.0
        self.previous_weights = weights.copy()
        
        if self.current_step < WARMUP_STEPS_LOCAL:
            turnover_penalty = 0.0
        else:
            turnover_penalty = (turnover ** 1.5) * TURNOVER_PENALTY_SCALE_LOCAL
            
        # -----------------------
        # Cash hoarding penalty
        # -----------------------
        cash_ratio = cash_weight
        cash_hoarding_penalty = max(0.0, cash_ratio - 0.8) * 1.0
        
        # -----------------------
        # Stagnation penalty
        # -----------------------
        staleness_penalty = 0.0
        if abs(portfolio_return) < 0.0001:
            staleness_penalty = 0.05
            
        # ==================================================
        # 🔥 FINAL REWARD (NO DOUBLE COUNTING ANYMORE)
        # ==================================================
        final_reward = (
            return_reward +
            rr_quality +
            winrate_reward +
            deployment_reward -
            turnover_penalty -
            cash_hoarding_penalty -
            staleness_penalty
        )
        
        # Store for debugging
        reward_components = {
            "return_reward": return_reward,
            "rr_quality": rr_quality,
            "winrate_reward": winrate_reward,
            "deployment_reward": deployment_reward,
            "turnover_penalty": turnover_penalty,
            "cash_penalty": cash_hoarding_penalty,
            "staleness": staleness_penalty,
            "final_reward": final_reward
        }
        self.reward_components_history.append(reward_components)
        
        # Debug print every 100 steps
        if self.current_step % 100 == 0:
            rc = reward_components
            print(
                f"[REWARD-DEBUG step={self.current_step}] PV={self.portfolio_value:.2f} "
                f"R={rc['final_reward']:.3f} "
                f"ret={rc['return_reward']:.3f} rrQ={rc['rr_quality']:.3f} "
                f"win={rc['winrate_reward']:.3f} dep={rc['deployment_reward']:.3f} "
                f"turn=-{rc['turnover_penalty']:.3f} cash=-{rc['cash_penalty']:.3f}"
            )
        
"""
    
    # Construct new content
    new_lines = lines[:start_idx] + [new_code] + lines[end_idx:]
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"✅ Successfully updated {filepath}")

if __name__ == "__main__":
    apply_new_reward_system("e:/tradingbot/trading_env.py")
    apply_new_reward_system("e:/tradingbot/backtesting/trading_env.py")
