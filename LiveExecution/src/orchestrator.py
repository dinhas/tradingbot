import logging
import numpy as np
from twisted.internet.defer import inlineCallbacks, gatherResults
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOATradeSide

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
        
        # Internal state
        self.portfolio_state = {asset: {} for asset in self.fm.assets} 
        self.active_positions = {} 
        self.pending_risk_actions = {} # Maps positionId to risk_actions
        
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
        # ProtoOATrader provides balance, and usually we can derive equity
        self.portfolio_state['balance'] = account_res.trader.balance / 100.0
        # For simplicity, if we don't have real-time equity from ProtoOATrader, use balance
        # Usually ProtoOATrader doesn't have live Equity, you need ProtoOAAssetReq or similar for margins
        # But we'll use balance as a proxy or assume equity=balance for sizing if not provided.
        self.portfolio_state['equity'] = self.portfolio_state.get('equity', self.portfolio_state['balance'])
        self.portfolio_state['initial_equity'] = self.portfolio_state.get('initial_equity', self.portfolio_state['equity'])
        self.portfolio_state['peak_equity'] = max(self.portfolio_state.get('peak_equity', 0), self.portfolio_state['equity'])

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
                         
                         # Calculate PnL % for Risk Model
                         # Note: In a real bot, we'd pull the actual realized PnL from the broker
                         # For now, we'll derive it if possible or use balance change
                         # But since we have the price, we can calculate it:
                         if pos_id in self.pending_risk_actions:
                             entry_price = pos.price
                             # Realized PnL is usually in pos.swap + pos.commission + pnl
                             # However, Risk Model features expect PnL % relative to entry
                             # This is how it was trained in the backtest.
                             
                             # We'll use a simplified PnL % for now or wait for account balance sync
                             # But better to record something close to what the model expects.
                             # Let's assume we can calculate it relative to entry price.
                             # (ClosePrice - EntryPrice) / EntryPrice * side
                             # But we might not have the closure price directly in this event if it just says 'closed'
                             
                             # Actually, let's keep it simple: fetch account summary after closure 
                             # and use balance delta? No, the risk model wants PnL per trade.
                             pass
                         
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
    def sync_active_positions(self):
        """Fetches open positions from API and updates internal state."""
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
            # You could add more detail here (PnL, etc) if needed by features
            
        except Exception as e:
            self.logger.error(f"Failed to sync active positions: {e}")

    @inlineCallbacks
    def bootstrap(self):
        """Pre-fetches historical data for all assets to be ready for inference."""
        self.logger.info("Bootstrapping history for all assets...")
        try:
            # 1. Sync Positions and Account State
            yield self.sync_active_positions()
            acc_res = yield self.client.fetch_account_summary()
            self.update_account_state(acc_res)
            
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
        import time
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
            # Fetch OHLCV and Account Summary
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
            
            # 3. Update State
            self.fm.update_data(symbol_id, ohlcv_res)
            self.update_account_state(account_res)
            
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
                self.pending_risk_actions[pos_id] = decision.get('risk_actions', np.zeros(3))
                self.active_positions[symbol_id] = pos_id
                self.logger.info(f"Recorded pending risk actions for {asset_name} position {pos_id}")
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
            # 1. Get Alpha Observation (140)
            alpha_obs = self.fm.get_alpha_observation(self.portfolio_state)
            
            # 2. Alpha Prediction
            all_alpha_actions = self.ml.get_alpha_action(alpha_obs)
            asset_index = self.fm.assets.index(asset_name)
            alpha_val = all_alpha_actions[asset_index]
            
            # Parse Alpha Direction (Match Backtest: > 0.33 Buy, < -0.33 Sell)
            direction = 1 if alpha_val > 0.33 else (-1 if alpha_val < -0.33 else 0)
            
            self.logger.info(f"Alpha predicted direction for {asset_name}: {direction} (raw: {alpha_val:.4f})")
            
            if direction == 0:
                return {'action': 0, 'allowed': False, 'reason': 'Alpha Hold'}
            
            # 3. Get Risk Observation (165)
            risk_obs = self.fm.get_risk_observation(asset_name, alpha_obs, self.portfolio_state)
            
            # 4. Risk Prediction
            risk_action = self.ml.get_risk_action(risk_obs)
            
            # Parse Risk Action (Match Backtest)
            # sl_mult: 0.2-2.0, tp_mult: 0.5-4.0, risk_raw: 0.0-1.0
            sl_mult = np.clip((risk_action[0] + 1) / 2 * 1.8 + 0.2, 0.2, 2.0)
            tp_mult = np.clip((risk_action[1] + 1) / 2 * 3.5 + 0.5, 0.5, 4.0)
            risk_raw = np.clip((risk_action[2] + 1) / 2, 0.0, 1.0)
            
            # Blocking Logic (Threshold: 0.20)
            if risk_raw < 0.20:
                self.logger.info(f"Risk Block for {asset_name}: raw risk {risk_raw:.4f} < 0.20")
                return {'action': 0, 'allowed': False, 'reason': f'Risk Block ({risk_raw:.2f})'}

            # Rescale Risk (Match Backtest)
            risk_raw_scaled = (risk_raw - 0.20) / 0.80
            
            # 5. Calculate Sizing & SL/TP Prices
            digits = self.symbol_digits.get(asset_name, 5)
            current_price = round(self.fm.history[asset_name].iloc[-1]['close'], digits)
            atr = self.fm.get_atr(asset_name)
            if atr <= 0: atr = current_price * 0.0001
            
            sl_dist = sl_mult * atr
            tp_dist = tp_mult * atr
            
            sl_price = round(current_price - (direction * sl_dist), digits)
            tp_price = round(current_price + (direction * tp_dist), digits)
            
            # Round distances to symbol's precision to avoid 'invalid precision' errors
            # This ensures the SL/TP price levels are mathematically valid for the broker
            pip_unit = 10 ** -digits
            sl_dist = round(sl_dist / pip_unit) * pip_unit
            tp_dist = round(tp_dist / pip_unit) * pip_unit
            
            # Calculate Relative values for API (Price Distance * 100,000)
            relative_sl = int(round(sl_dist * 100000))
            relative_tp = int(round(tp_dist * 100000))
            
            # Lot Calculation (Match Backtest calculate_position_size)
            equity = self.portfolio_state.get('equity', 10.0)
            balance = self.portfolio_state.get('balance', 0)
            
            # USER OVERRIDE: 0.01 lots if balance < $30
            if balance < 30:
                lots = 0.01
                self.logger.info(f"Balance ${balance:.2f} < $30. Using hardcoded lot size: 0.01")
            else:
                MAX_RISK_PER_TRADE = 0.80
                drawdown = 1.0 - (equity / max(self.portfolio_state.get('peak_equity', equity), 1e-9))
                risk_cap_mult = max(0.2, 1.0 - (drawdown * 2.0))
                
                actual_risk_pct = risk_raw_scaled * MAX_RISK_PER_TRADE * risk_cap_mult
                risk_amount_cash = equity * actual_risk_pct
                
                contract_size = 100 if asset_name == 'XAUUSD' else 100000
                # Simplified for USD quote pairs (EURUSD, GBPUSD, XAUUSD)
                lots = risk_amount_cash / (sl_dist * contract_size)
            
            lots = np.clip(lots, 0.01, 100.0)
            
            # 6. TradeGuard Prediction
            t_info = {
                'entry': current_price,
                'sl': sl_price,
                'tp': tp_price
            }
            
            # Prepare TradeGuard context
            self.portfolio_state[asset_name] = self.portfolio_state.get(asset_name, {})
            self.portfolio_state[asset_name]['action_raw'] = direction
            
            tg_obs = self.fm.get_tradeguard_observation(asset_name, t_info, self.portfolio_state)
            tg_action = self.ml.get_tradeguard_action(tg_obs)
            
            allowed = (tg_action == 1)
            self.logger.info(f"TradeGuard decision for {asset_name}: {'ALLOW' if allowed else 'BLOCK'}")
            
            return {
                'symbol_id': symbol_id,
                'asset': asset_name,
                'action': 1 if direction == 1 else 2, # 1=Buy, 2=Sell in protocol? Actually logic uses side.
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
            return None
            
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
