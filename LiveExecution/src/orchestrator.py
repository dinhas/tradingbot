import logging
import numpy as np

class Orchestrator:
    """
    Coordinates data fetching, feature engineering, and sequential inference.
    """
    def __init__(self, client, feature_manager, model_loader):
        self.logger = logging.getLogger("LiveExecution")
        self.client = client
        self.fm = feature_manager
        self.ml = model_loader
        
        # Internal state
        self.portfolio_state = {} # To be updated from account summary
        self.active_positions = {} # To be updated from account summary

    def update_account_state(self, account_res):
        """Updates internal portfolio state from cTrader response."""
        # This will be refined as we see the actual ProtoOATraderRes structure
        # For now, we assume it provides equity, balance etc.
        self.portfolio_state['equity'] = account_res.trader.balance / 100.0 # Just a guess on divisor
        # ... update other metrics

    def is_asset_locked(self, symbol_id):
        """Checks if a position is already open for the given symbol."""
        locked = symbol_id in self.active_positions
        if locked:
            self.logger.info(f"Asset {symbol_id} is locked (position already exists).")
        return locked

    def run_inference_chain(self, symbol_id):
        """
        Executes the full inference pipeline for a given symbol.
        1. Get Alpha/Risk Observation
        2. Alpha Direction
        3. Risk Sizing & SL/TP
        4. TradeGuard Allow/Block
        """
        try:
            # Prepare portfolio state for Alpha/Risk
            # (In a real scenario, we'd ensure this is fresh)
            
            # 1. Get Alpha/Risk Observation
            alpha_obs = self.fm.get_alpha_risk_observation(self.portfolio_state)
            
            # 2. Alpha Prediction
            alpha_action = self.ml.get_alpha_action(alpha_obs)
            self.logger.info(f"Alpha predicted action for symbol {symbol_id}: {alpha_action}")
            
            if alpha_action == 0: # Hold/No Action
                return {'action': 0, 'allowed': False, 'reason': 'Alpha Hold'}
            
            # 3. Risk Prediction
            # Risk model often shares the same observation as Alpha in this project's curriculum
            risk_action = self.ml.get_risk_action(alpha_obs)
            self.logger.info(f"Risk predicted for symbol {symbol_id}: {risk_action}")
            
            # action format usually: [size_pct, sl_pct, tp_pct]
            size_pct = risk_action[0]
            sl_pct = risk_action[1]
            tp_pct = risk_action[2]
            
            # 4. TradeGuard Prediction
            # Prepare TradeGuard Observation (needs trade info)
            # We need to map symbolId back to name for fm
            symbol_name = self._get_symbol_name(symbol_id)
            
            # Calculate absolute SL/TP based on current price
            current_price = self.fm.history[symbol_name].iloc[-1]['close']
            
            # Simplified SL/TP calculation for TradeGuard obs
            side = 1 if alpha_action == 1 else -1 # 1: Long, 2: Short (Adjust based on model convention)
            # Usually Alpha action 1=Long, 2=Short
            
            sl_price = current_price * (1 - (sl_pct * 0.05 * side)) # Dummy scale factor
            tp_price = current_price * (1 + (tp_pct * 0.1 * side))
            
            trade_infos = {
                symbol_name: {
                    'entry': current_price,
                    'sl': sl_price,
                    'tp': tp_price
                }
            }
            
            # Add action_raw to portfolio_state for TradeGuard
            self.portfolio_state[symbol_name] = self.portfolio_state.get(symbol_name, {})
            self.portfolio_state[symbol_name]['action_raw'] = alpha_action
            
            tg_obs = self.fm.get_tradeguard_observation(trade_infos, self.portfolio_state)
            tg_action = self.ml.get_tradeguard_action(tg_obs)
            
            allowed = (tg_action == 1)
            self.logger.info(f"TradeGuard decision for symbol {symbol_id}: {'ALLOW' if allowed else 'BLOCK'}")
            
            return {
                'symbol_id': symbol_id,
                'action': alpha_action,
                'size': size_pct,
                'sl': sl_price,
                'tp': tp_price,
                'allowed': allowed
            }
            
        except Exception as e:
            self.logger.error(f"Error in inference chain for symbol {symbol_id}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _get_symbol_name(self, symbol_id):
        """Reverse mapping from symbolId to name."""
        inv_map = {v: k for k, v in self.client.symbol_ids.items()}
        return inv_map.get(symbol_id, "Unknown")
