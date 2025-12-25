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
        self.portfolio_state = {asset: {} for asset in self.fm.assets} 
        self.active_positions = {} 

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
        1. Get Alpha Observation
        2. Alpha Direction
        3. Get Risk Observation
        4. Risk Sizing & SL/TP
        5. TradeGuard Allow/Block
        """
        try:
            # 1. Get Alpha Observation (140)
            alpha_obs = self.fm.get_alpha_observation(self.portfolio_state)
            
            # 2. Alpha Prediction (Returns actions for ALL assets)
            all_alpha_actions = self.ml.get_alpha_action(alpha_obs)
            
            # Map symbol_id to index in the 5-asset action array
            asset_name = self._get_symbol_name(symbol_id)
            asset_index = self.fm.assets.index(asset_name)
            
            alpha_action = all_alpha_actions[asset_index]
            self.logger.info(f"Alpha predicted action for {asset_name} (index {asset_index}): {alpha_action}")
            
            if abs(alpha_action) < 0.1: # Threshold for 'Hold'
                return {'action': 0, 'allowed': False, 'reason': 'Alpha Hold'}
            
            # 3. Get Risk Observation (165)
            risk_obs = self.fm.get_risk_observation(asset_name, alpha_obs, self.portfolio_state)
            
            # 4. Risk Prediction
            all_risk_actions = self.ml.get_risk_action(risk_obs)
            
            # If Risk model returns a flat array of 15 elements (3 per asset), or nested
            # Let's assume 3 per asset, flattened: [size0, sl0, tp0, size1, sl1, tp1, ...]
            # Note: Risk Management model usually predicts for the whole episode sequence, 
            # but in live it's one trade at a time.
            
            # Checking Risk Layer Action Space: shape=(3,)
            # Wait, if it was trained on sequential episodic, predict might return (3,)
            # Let's assume it returns (3,) for the current asset being managed.
            
            if all_risk_actions.shape[0] == 15: # Multi-asset Risk
                risk_start = asset_index * 3
                asset_risk_action = all_risk_actions[risk_start : risk_start + 3]
            else:
                asset_risk_action = all_risk_actions
            
            size_pct = asset_risk_action[0]
            sl_pct = asset_risk_action[1]
            tp_pct = asset_risk_action[2]
            
            # 5. TradeGuard Prediction
            # Prepare TradeGuard Observation (needs trade info)
            current_price = self.fm.history[asset_name].iloc[-1]['close']
            
            side = 1 if alpha_action > 0 else -1
            
            sl_price = current_price * (1 - (sl_pct * 0.05 * side)) 
            tp_price = current_price * (1 + (tp_pct * 0.1 * side))
            
            trade_infos = {
                asset_name: {
                    'entry': current_price,
                    'sl': sl_price,
                    'tp': tp_price
                }
            }
            
            # Add discrete action to portfolio_state for TradeGuard
            discrete_action = 1 if side == 1 else 2
            self.portfolio_state[asset_name] = self.portfolio_state.get(asset_name, {})
            self.portfolio_state[asset_name]['action_raw'] = discrete_action
            
            tg_obs = self.fm.get_tradeguard_observation(trade_infos, self.portfolio_state)
            tg_action = self.ml.get_tradeguard_action(tg_obs)
            
            allowed = (tg_action == 1)
            self.logger.info(f"TradeGuard decision for {asset_name}: {'ALLOW' if allowed else 'BLOCK'}")
            
            return {
                'symbol_id': symbol_id,
                'asset': asset_name,
                'action': discrete_action,
                'raw_action': alpha_action,
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
