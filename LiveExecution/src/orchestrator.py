import logging
import numpy as np
import time
from twisted.internet import reactor
from twisted.internet.defer import inlineCallbacks, gatherResults
from twisted.internet.task import deferLater
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOATradeSide
from .sizing import PositionSizer

class Orchestrator:
    """
    Coordinates data fetching, feature engineering, and sequential inference.
    """
    def __init__(self, client, feature_manager, model_loader, notifier):
        self.logger = logging.getLogger("LiveExecution")
        self.client = client
        self.fm = feature_manager
        self.ml = model_loader
        self.notifier = notifier
        self.sizer = PositionSizer()
        
        # Internal state
        self.portfolio_state = {asset: {} for asset in self.fm.assets} 
        self.active_positions = {} 
        self.pending_risk_actions = {} # Maps positionId to risk_actions
        self.last_global_sync = 0 # Timestamp of last account/position sync
        self.sync_lock = False # Primitive lock to prevent concurrent syncs
        
        # Price precision (digits) for each asset
        self.symbol_digits = {
            'EURUSD': 5,
            'GBPUSD': 5,
            'USDCHF': 5,
            'USDJPY': 3,
            'XAUUSD': 2 # Gold usually has 2 or 3, most commonly 2 for demo/live
        }
    def update_account_state(self, account_res):
        """Updates internal portfolio state from cTrader response."""
        self.portfolio_state['balance'] = account_res.trader.balance / 100.0
        self.portfolio_state['equity'] = self.portfolio_state.get('equity', self.portfolio_state['balance'])
        self.portfolio_state['initial_equity'] = self.portfolio_state.get('initial_equity', self.portfolio_state['equity'])
        self.portfolio_state['peak_equity'] = max(self.portfolio_state.get('peak_equity', 0), self.portfolio_state['equity'])
        
        # Calculate Global TradeGuard features
        equity = self.portfolio_state['equity']
        peak = self.portfolio_state['peak_equity']
        self.portfolio_state['total_drawdown'] = 1.0 - (equity / peak) if peak > 0 else 0
        
        # Total exposure placeholder - would need to sum position values in real time
        # For now we'll use number of positions as proxy or just 0 if not easily calculated
        # Actually we have self.active_positions, but not their sizes here.
        self.portfolio_state['total_exposure'] = len(self.active_positions) * 0.25 # Assumption 25% each

    def on_order_execution(self, event):
        """Handles order execution events from cTrader."""
        try:
            self.logger.info(f"=== ORDER EXECUTION EVENT ===")
            self.logger.info(f"Execution Type: {event.executionType}")
            
            # Position Data
            if event.position:
                pos = event.position
                pos_id = pos.positionId
                symbol_id = pos.tradeData.symbolId
                asset_name = self._get_symbol_name(symbol_id)
                
                # Check for closures or updates
                # ExecutionType: 3 (TRADE) for closures, 2 (ORDER_ACCEPTED) for openings?
                # Actually in cTrader OpenAPI:
                # 2 = ORDER_ACCEPTED (The order is accepted by the server)
                # 3 = ORDER_FILLED (The order is fully or partially filled)
                # Here we want to track when a position is GONE or CLOSED.
                
                from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOAExecutionType
                
                if event.executionType == ProtoOAExecutionType.ORDER_FILLED:
                    # If positionStatus is CLOSED/DELETED, it's a closure
                    from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOAPositionStatus
                    if pos.positionStatus in [ProtoOAPositionStatus.POSITION_STATUS_CLOSED, 6]: # 6 is often used for DELETED/GONE
                         self.logger.info(f"Position {pos_id} CLOSED for {asset_name}")
                         
                         if symbol_id in self.active_positions:
                             del self.active_positions[symbol_id]

                    elif pos.positionStatus == ProtoOAPositionStatus.POSITION_STATUS_OPEN:
                        # Position just opened or updated
                        self.active_positions[symbol_id] = pos_id
                        self.logger.info(f"Position {pos_id} is now OPEN for {asset_name}")

            # If it's a trade closure, let's log the detail
            if hasattr(event, 'order') and event.order:
                 order = event.order
                 if order.closingOrder:
                      self.logger.info(f"Order {order.orderId} is a CLOSING order.")
                
        except Exception as e:
            self.logger.error(f"Error handling execution event: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def on_order_error(self, event):
        """Handles order error events from cTrader."""
        self.logger.error(f"=== ORDER ERROR EVENT ===")
        self.logger.error(f"Error Code: {event.errorCode}")
        self.logger.error(f"Description: {event.description if hasattr(event, 'description') else 'N/A'}")
        self.logger.error(f"Order ID: {event.orderId if hasattr(event, 'orderId') else 'N/A'}")
        
        # Try to get more details
        if hasattr(event, 'ctidTraderAccountId'):
            self.logger.error(f"Account ID: {event.ctidTraderAccountId}")
        
        # Send notification
        self.notifier.send_error(f"Order rejected: {event.errorCode} - {getattr(event, 'description', 'No description')}")

    def is_asset_locked(self, symbol_id):
        """Checks if a position is already open for the given symbol."""
        locked = symbol_id in self.active_positions
        if locked:
            self.logger.info(f"Asset {symbol_id} is locked (position already exists).")
        return locked
    
    @inlineCallbacks
    def sync_active_positions(self, force=False):
        """Fetches open positions from API and updates internal state."""
        if not force and (time.time() - self.last_global_sync < 10):
            return

        # Simple lock to prevent multiple concurrent sync requests
        if self.sync_lock:
            # Wait a bit if a sync is already in progress
            yield deferLater(reactor, 0.5, lambda: None)
            if time.time() - self.last_global_sync < 10:
                return

        self.sync_lock = True
        try:
            res = yield self.client.fetch_open_positions()
            # Reset active positions
            new_active = {}
            for pos in res.position:
                if hasattr(pos, 'tradeData') and hasattr(pos.tradeData, 'symbolId'):
                    new_active[pos.tradeData.symbolId] = pos.positionId
            
            self.active_positions = new_active
            self.logger.info(f"Synced {len(self.active_positions)} active positions from API.")
            
            # Update portfolio state for features
            self.portfolio_state['num_open_positions'] = len(self.active_positions)
            
            # This doesn't update last_global_sync yet, 
            # we'll do it after account summary too if called from bootstrap
        except Exception as e:
            self.logger.error(f"Failed to sync active positions: {e}")
        finally:
            self.sync_lock = False

    @inlineCallbacks
    def bootstrap(self):
        """Pre-fetches historical data for all assets to be ready for inference."""
        self.logger.info("Bootstrapping history for all assets...")
        try:
            # 1. Sync Positions and Account State
            yield self.sync_active_positions(force=True)
            acc_res = yield self.client.fetch_account_summary()
            self.update_account_state(acc_res)
            self.last_global_sync = time.time()
            
            # 2. Fetch History
            tasks = []
            for asset_name in self.fm.assets:
                symbol_id = self.client.symbol_ids.get(asset_name)
                if symbol_id:
                    tasks.append(self.client.fetch_ohlcv(symbol_id, count=300))
            
            results = yield gatherResults(tasks, consumeErrors=True)
            
            symbols_to_subscribe = []
            for i, res in enumerate(results):
                asset_name = self.fm.assets[i]
                symbol_id = self.client.symbol_ids.get(asset_name)
                if not isinstance(res, Exception):
                    self.fm.update_data(symbol_id, res)
                    
                    # Initialize last_bar_timestamps from history to prevent immediate trigger on startup
                    if res.trendbar:
                        latest_ts = max(bar.utcTimestampInMinutes for bar in res.trendbar)
                        self.client.last_bar_timestamps[symbol_id] = latest_ts
                        
                    self.logger.info(f"Loaded {len(self.fm.history[asset_name])} bars for {asset_name}")
                    symbols_to_subscribe.append(symbol_id)
                else:
                    self.logger.error(f"Failed to bootstrap {asset_name}: {res}")
            
            if symbols_to_subscribe:
                self.logger.info(f"Subscribing to spots for: {symbols_to_subscribe}")
                yield self.client.subscribe(symbols_to_subscribe)
            
            if self.fm.is_ready():
                self.logger.info("System is READY for live execution.")
                self.notifier.send_message("System is READY for live execution.")
            else:
                self.logger.warning("System bootstrap complete but FeatureManager not ready yet.")
                
        except Exception as e:
            self.logger.error(f"Error during bootstrap: {e}")
            self.notifier.send_error(f"Bootstrap failed: {e}")

    @inlineCallbacks
    def on_m5_candle_close(self, symbol_id):
        """
        Main Event Handler: Triggered when an M5 candle closes.
        """
        start_time = time.time()
        asset_name = self._get_symbol_name(symbol_id)
        self.logger.info(f"--- M5 Close Detected: {asset_name} ({symbol_id}) ---")
        
        # 0. Check Readiness
        if not self.fm.is_ready():
             self.logger.info("FeatureManager not ready (insufficient history). Skipping.")
             return

        # 1. Sync Positions and Check System/Asset Limits
        yield self.sync_active_positions()
        
        # System-level limit: Max 2 open positions
        if len(self.active_positions) >= 2:
            self.logger.info(f"System reached max open positions limit (2). Currently open: {list(self.active_positions.keys())}")
            return

        if self.is_asset_locked(symbol_id):
            self.logger.info(f"Skipping {asset_name} due to active lock (symbol already has position).")
            return

        try:
            # 2. Fetch Data (Parallel)
            # Use cached account data if available and fresh (< 10s)
            now = time.time()
            if now - self.last_global_sync < 10:
                self.logger.debug(f"Using cached account data for {asset_name}")
                ohlcv_res = yield self.client.fetch_ohlcv(symbol_id)
            else:
                # Fetch both if cache expired
                d_ohlcv = self.client.fetch_ohlcv(symbol_id)
                d_account = self.client.fetch_account_summary()
                
                results = yield gatherResults([d_ohlcv, d_account], consumeErrors=True)
                
                # Check for errors in results
                for res in results:
                    if isinstance(res, Exception): # Or Failure
                         self.logger.error(f"Data fetch failed: {res}")
                         self.notifier.send_error(f"Data fetch failed for {asset_name}: {res}")
                         return

                ohlcv_res = results[0]
                account_res = results[1]
                self.update_account_state(account_res)
                self.last_global_sync = time.time()
            
            # 3. Update State
            self.fm.update_data(symbol_id, ohlcv_res)
            # account_state already updated if we fetched it, or it's from cache
            
            # 4. Run Inference
            inference_start = time.time()
            decision = self.run_inference_chain(symbol_id)
            inference_time = time.time() - inference_start
            
            if not decision or decision.get('action') == 0:
                self.logger.info(f"Inference complete in {inference_time:.3f}s. No action taken.")
                return 
            
            # 5. Execute & Notify
            if decision['allowed']:
                yield self.execute_decision(decision, symbol_id)
            else:
                self.notifier.send_block_event({
                    'symbol': asset_name,
                    'reason': decision.get('reason', 'TradeGuard Block')
                })
            
            total_time = time.time() - start_time
            self.logger.info(f"M5 Cycle for {asset_name} completed in {total_time:.3f}s (Inference: {inference_time:.3f}s)")
                
        except Exception as e:
            self.logger.error(f"Orchestration error for {asset_name}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.notifier.send_error(f"Orchestration error for {asset_name}: {e}")

    @inlineCallbacks
    def execute_decision(self, decision, symbol_id):
        """Places the order and notifies."""
        try:
            side = ProtoOATradeSide.BUY if decision['action'] == 1 else ProtoOATradeSide.SELL
            
            # Convert Lots to cTrader volume (Units * 100)
            # Lots are already calculated in run_inference_chain to match backtest
            # Round volume to nearest 100,000 (0.01 lots) - cTrader step requirement
            # 1 Lot = 100,000 units = 10,000,000 protocol volume
            # 0.01 Lot = 1,000 units = 100,000 protocol volume
            raw_volume = decision['lots'] * 100000 * 100
            volume = int(round(raw_volume / 100000) * 100000)
            
            # Ensure minimum volume (100,000 = 0.01 lots)
            if volume < 100000: 
                volume = 100000
            
            self.logger.info(f"Executing {decision['asset']}: {side} Lots: {decision['lots']:.2f} Vol: {volume}")
            
            # Use relative SL/TP as required for Market Orders in ProtoOANewOrderReq
            execution_res = yield self.client.execute_market_order(
                symbol_id, 
                volume, 
                side, 
                relative_sl=decision.get('relative_sl'), 
                relative_tp=decision.get('relative_tp')
            )
            
            self.logger.info(f"Order Execution Response for {decision['asset']}: {execution_res}")
            
            self.notifier.send_trade_event({
                'symbol': decision['asset'],
                'action': 'BUY' if side == ProtoOATradeSide.BUY else 'SELL',
                'size': f"{decision['lots']:.2f} lots"
            })
            
            # Record risk actions for this position to use when it closes
            if hasattr(execution_res, 'position') and execution_res.position:
                pos_id = execution_res.position.positionId
                self.pending_risk_actions[pos_id] = decision.get('risk_actions', np.zeros(2))
                self.active_positions[symbol_id] = pos_id
                self.logger.info(f"Recorded pending risk actions for {decision['asset']} position {pos_id}")
            else:
                # Optimistically lock asset if we don't have ID yet
                self.active_positions[symbol_id] = "PENDING"
            
        except Exception as e:
            self.logger.error(f"Execution failed: {e}")
            self.notifier.send_error(f"Execution failed for {decision['asset']}: {e}")

    def run_inference_chain(self, symbol_id):
        """
        Executes the full inference pipeline for a given symbol.
        Matching Backtest Logic Exactly.
        """
        try:
            asset_name = self._get_symbol_name(symbol_id)
            
            # 1. Get Alpha Observation (40)
            alpha_obs = self.fm.get_alpha_observation(asset_name, self.portfolio_state)
            
            # 2. Alpha Prediction (Single-Pair)
            alpha_action = self.ml.get_alpha_action(alpha_obs)
            
            # Handle array output
            if isinstance(alpha_action, np.ndarray):
                alpha_val = float(alpha_action.flatten()[0])
            else:
                alpha_val = float(alpha_action)
                
            # Parse Alpha Direction (> 0.33 Buy, < -0.33 Sell)
            direction = 1 if alpha_val > 0.33 else (-1 if alpha_val < -0.33 else 0)
            
            self.logger.info(f"Alpha predicted direction for {asset_name}: {direction} (raw: {alpha_val:.4f})")
            
            if direction == 0:
                return {'action': 0, 'allowed': False, 'reason': 'Alpha Hold'}
            
            # 3. Get Risk Observation (45)
            # Alpha Obs + Account Stats
            risk_obs = self.fm.get_risk_observation(alpha_obs, self.portfolio_state)
            
            # 4. Risk Prediction
            risk_action = self.ml.get_risk_action(risk_obs)
            
            # Parse Risk Action (2 Outputs)
            # sl_mult: 0.2-2.0, tp_mult: 0.5-4.0
            sl_mult = np.clip((risk_action[0] + 1) / 2 * 1.8 + 0.2, 0.2, 2.0)
            tp_mult = np.clip((risk_action[1] + 1) / 2 * 3.5 + 0.5, 0.5, 4.0)
            
            # 5. Calculate Sizing & SL/TP Prices
            digits = self.symbol_digits.get(asset_name, 5)
            current_price = round(self.fm.history[asset_name].iloc[-1]['close'], digits)
            atr = self.fm.get_atr(asset_name)
            if atr <= 0: atr = current_price * 0.0001
            
            sl_dist = sl_mult * atr
            tp_dist = tp_mult * atr
            
            sl_price = round(current_price - (direction * sl_dist), digits)
            tp_price = round(current_price + (direction * tp_dist), digits)
            
            # Round distances to symbol's precision
            pip_unit = 10 ** -digits
            sl_dist = round(sl_dist / pip_unit) * pip_unit
            tp_dist = round(tp_dist / pip_unit) * pip_unit
            
            # Calculate Relative values for API (Price Distance * 100,000)
            relative_sl = int(round(sl_dist * 100000))
            relative_tp = int(round(tp_dist * 100000))
            
            # Lot Calculation using PositionSizer
            lots = self.sizer.calculate_risk_lots(
                self.portfolio_state,
                asset_name,
                current_price,
                sl_price,
                atr
            )
            
            self.logger.info(f"Risk Output: SL_Mult={sl_mult:.2f}, TP_Mult={tp_mult:.2f}, Lots={lots}")
            
            if lots < 0.01:
                self.logger.info(f"Lot size calculated as {lots}. Too small. Blocking.")
                return {'action': 0, 'allowed': False, 'reason': f'Risk Sizing Zero ({lots})'}
            
            # 6. TradeGuard Prediction
            trade_info = {
                'entry': current_price,
                'sl': sl_price,
                'tp': tp_price,
                'risk_val': 0.5 # Default Risk Model conviction
            }
            
            # Prepare TradeGuard context (Single-Pair)
            self.portfolio_state[asset_name] = self.portfolio_state.get(asset_name, {})
            self.portfolio_state[asset_name]['action_raw'] = direction
            
            # Get signal persistence if possible
            # (In live this would need a history tracker, but we'll use dummy for now)
            self.portfolio_state[asset_name]['signal_persistence'] = 0
            self.portfolio_state[asset_name]['signal_reversal'] = 0
            
            tg_obs = self.fm.get_tradeguard_observation(
                asset_name, 
                trade_info, 
                self.portfolio_state,
                norm_stats=getattr(self.ml, 'tg_norm_stats', None)
            )
            tg_action = self.ml.get_tradeguard_action(tg_obs)
            
            allowed = (tg_action == 1)
            self.logger.info(f"TradeGuard decision for {asset_name}: {'ALLOW' if allowed else 'BLOCK'}")
            
            return {
                'symbol_id': symbol_id,
                'asset': asset_name,
                'action': 1 if direction == 1 else 2, # 1=Buy, 2=Sell
                'lots': float(lots),
                'sl': float(sl_price),
                'tp': float(tp_price),
                'relative_sl': relative_sl,
                'relative_tp': relative_tp,
                'risk_actions': risk_action,
                'allowed': allowed
            }
            
        except Exception as e:
            self.logger.error(f"Error in inference chain for symbol {symbol_id}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def stop(self):
        """Stops the orchestrator and client."""
        self.logger.info("Stopping Orchestrator...")
        self.notifier.send_message("🛑 **System Stopping...**")
        self.client.stop()

    def _get_symbol_name(self, symbol_id):
        """Reverse mapping from symbolId to name."""
        inv_map = {v: k for k, v in self.client.symbol_ids.items()}
        return inv_map.get(symbol_id, "Unknown")
