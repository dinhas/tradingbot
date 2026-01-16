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
        
        # Batching State
        self.pending_batch = {} # Maps symbol_id to candle_data
        self.batch_timer = None
        self.is_processing_batch = False
    def update_account_state(self, account_res):
        """Updates internal portfolio state from cTrader response."""
        # ProtoOATrader provides balance
        self.portfolio_state['balance'] = account_res.trader.balance / 100.0
        
        # Determine Equity (Proxy via Balance if live equity unavailable)
        #Ideally we would fetch margin/unrealized PnL, but using balance as base for now
        self.portfolio_state['equity'] = self.portfolio_state.get('equity', self.portfolio_state['balance'])
        
        self.portfolio_state['initial_equity'] = self.portfolio_state.get('initial_equity', self.portfolio_state['equity'])
        self.portfolio_state['peak_equity'] = max(self.portfolio_state.get('peak_equity', 0), self.portfolio_state['equity'])
        
        # Calculate Drawdown
        peak = self.portfolio_state['peak_equity']
        if peak > 0:
            self.portfolio_state['total_drawdown'] = 1.0 - (self.portfolio_state['equity'] / peak)
        else:
             self.portfolio_state['total_drawdown'] = 0.0

    @inlineCallbacks
    def sync_active_positions(self):
        """Fetches open positions from API and updates internal state."""
        try:
            res = yield self.client.fetch_open_positions()
            # Reset active positions
            new_active = {}
            total_value_usd = 0.0
            
            for pos in res.position:
                if hasattr(pos, 'tradeData') and hasattr(pos.tradeData, 'symbolId'):
                    sym_id = pos.tradeData.symbolId
                    new_active[sym_id] = pos.positionId
                    
                    # Calculate Exposure logic
                    asset_name = self._get_symbol_name(sym_id)
                    units = pos.volume / 100.0 # cTrader volume is Unit*100
                    
                    # Get Price for valuation
                    price = pos.price 
                    if asset_name in self.fm.history and not self.fm.history[asset_name].empty:
                        price = self.fm.history[asset_name].iloc[-1]['close']
                        
                    # Exposure in USD
                    if asset_name in ['USDJPY', 'USDCHF']:
                         total_value_usd += units
                    else:
                         total_value_usd += units * price
            
            self.active_positions = new_active
            self.logger.info(f"Synced {len(self.active_positions)} active positions. Exposure Value: ${total_value_usd:.2f}")
            
            # Update portfolio state
            self.portfolio_state['num_open_positions'] = len(self.active_positions)
            
            equity = self.portfolio_state.get('equity', 1.0)
            if equity <= 0: equity = 1.0
            self.portfolio_state['total_exposure'] = total_value_usd / equity
            
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

    def on_m5_candle_close(self, symbol_id, candle_data=None):
        """
        Main Event Handler: Collects symbols into a batch over a 200ms window.
        This solves the 'symbol-by-symbol' disadvantage by allowing ranking.
        """
        from twisted.internet import reactor
        
        asset_name = self._get_symbol_name(symbol_id)
        self.logger.info(f"M5 Close Detected for {asset_name}. Adding to batch.")
        
        # Add to pending batch
        self.pending_batch[symbol_id] = candle_data
        
        # Schedule batch execution if not already scheduled
        if not self.batch_timer or not self.batch_timer.active():
            self.logger.info("Starting batch window (200ms)...")
            self.batch_timer = reactor.callLater(0.2, self.execute_batch)

    @inlineCallbacks
    def execute_batch(self):
        """
        Executes inference for all collected symbols in the batch.
        Ranks them by Alpha confidence and picks the best 2.
        """
        import time
        start_time = time.time()
        
        if self.is_processing_batch:
            self.logger.warning("Batch processing already in progress. Skipping.")
            return
            
        self.is_processing_batch = True
        symbols_to_process = list(self.pending_batch.keys())
        batch_data = self.pending_batch.copy()
        self.pending_batch = {} # Clear for next cycle
        
        self.logger.info(f"--- Processing Batch of {len(symbols_to_process)} symbols: {[self._get_symbol_name(s) for s in symbols_to_process]} ---")
        
        try:
            # 1. Update Feature Manager for all symbols in batch
            for sid, data in batch_data.items():
                if data and 'bar' in data:
                    self.fm.update_single_bar(sid, data['bar'])
            
            # 2. Run Inference for all symbols
            decisions = []
            for sid in symbols_to_process:
                asset_name = self._get_symbol_name(sid)
                
                # Check Lock (Local State)
                if self.is_asset_locked(sid):
                    self.logger.info(f"Asset {asset_name} is locked. Skipping.")
                    continue
                
                # Run Inference
                decision = self.run_inference_chain(sid)
                if decision and decision.get('action') != 0:
                    # Add decision with confidence score (absolute alpha_val)
                    decisions.append(decision)
            
            # 3. Handle Positions & Ranking
            if not decisions:
                self.logger.info("No trade signals in batch.")
                self.is_processing_batch = False
                return

            # Rank by alpha_val absolute magnitude (Confidence)
            # High magnitude = High priority
            decisions.sort(key=lambda x: abs(x.get('alpha_val', 0)), reverse=True)
            
            self.logger.info(f"Batch signals ranked: {[(d['asset'], round(d['alpha_val'], 4)) for d in decisions]}")
            
            # 4. Execute Top Decisions (Up to System Limit)
            max_to_open = 2 - len(self.active_positions)
            if max_to_open <= 0:
                self.logger.info(f"System reached limit (2 positions). Ignoring {len(decisions)} new signals.")
            else:
                to_execute = decisions[:max_to_open]
                self.logger.info(f"Executing top {len(to_execute)} signals from batch.")
                
                execution_tasks = []
                for d in to_execute:
                    execution_tasks.append(self.execute_decision(d, d['symbol_id']))
                
                yield gatherResults(execution_tasks, consumeErrors=True)
            
            total_time = time.time() - start_time
            self.logger.info(f"Batch Cycle completed in {total_time:.3f}s. Processed {len(symbols_to_process)} symbols.")
            
        except Exception as e:
            self.logger.error(f"Error in batch execution: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        finally:
            self.is_processing_batch = False

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
            
            # 2. Alpha Prediction (Returns value in [-1, 1])
            alpha_pred = self.ml.get_alpha_action(alpha_obs)
            # Alpha model now predicts for ONE asset, returns (1,) or scalar
            alpha_val = float(alpha_pred[0] if isinstance(alpha_pred, (np.ndarray, list)) else alpha_pred)
            
            # Parse Alpha Direction (Match Backtest: > 0.33 Buy, < -0.33 Sell)
            direction = 1 if alpha_val > 0.33 else (-1 if alpha_val < -0.33 else 0)
            
            self.logger.info(f"Alpha predicted direction for {asset_name}: {direction} (raw: {alpha_val:.4f})")
            
            if direction == 0:
                return {'action': 0, 'allowed': False, 'reason': 'Alpha Hold', 'alpha_val': alpha_val}
            
            # 3. Get Risk Observation (65)
            risk_obs = self.fm.get_risk_observation(asset_name, alpha_obs, self.portfolio_state)
            
            # 4. Risk Prediction (Returns [SL_Mult, TP_Mult, Risk_Factor])
            risk_action = self.ml.get_risk_action(risk_obs)
            
            # Parse Risk Action (Match RiskManagementEnv scaling)
            # sl_mult: 0.75-2.5, tp_mult: 0.5-4.0, risk_raw: 0.0-1.0
            sl_mult = np.clip((risk_action[0] + 1) / 2 * 1.75 + 0.75, 0.75, 2.5)
            tp_mult = np.clip((risk_action[1] + 1) / 2 * 3.5 + 0.5, 0.5, 4.0)
            risk_raw = np.clip((risk_action[2] + 1) / 2, 0.0, 1.0)
            
            # BLOCKING LOGIC (Match RiskManagementEnv: action < 0.10 is a BLOCK)
            BLOCK_THRESHOLD = 0.10
            if risk_raw < BLOCK_THRESHOLD:
                self.logger.info(f"Risk model BLOCKED trade for {asset_name} (risk_factor: {risk_raw:.4f})")
                return {'action': 0, 'allowed': False, 'reason': 'Risk Blocked', 'alpha_val': alpha_val}
            
            # Rescale Risk Factor for lot calculation
            risk_raw_scaled = (risk_raw - BLOCK_THRESHOLD) / (1.0 - BLOCK_THRESHOLD)
            
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
            pip_unit = 10 ** -digits
            sl_dist = round(sl_dist / pip_unit) * pip_unit
            tp_dist = round(tp_dist / pip_unit) * pip_unit
            
            # Calculate Relative values for API (Price Distance * 100,000)
            relative_sl = int(round(sl_dist * 100000))
            relative_tp = int(round(tp_dist * 100000))
            
            # Lot Calculation (Match RiskManagementEnv exactly)
            equity = self.portfolio_state.get('equity', 10.0)
            balance = self.portfolio_state.get('balance', 0)
            
            # USER OVERRIDE: 0.01 lots if balance < $30
            if balance < 30:
                lots = 0.01
                self.logger.info(f"Balance ${balance:.2f} < $30. Using hardcoded lot size: 0.01")
            else:
                # MATCH RiskManagementEnv: MAX_RISK_PER_TRADE = 0.40
                MAX_RISK_PER_TRADE = 0.40
                drawdown = 1.0 - (equity / max(self.portfolio_state.get('peak_equity', equity), 1e-9))
                risk_cap_mult = max(0.2, 1.0 - (drawdown * 2.0))
                
                actual_risk_pct = risk_raw_scaled * MAX_RISK_PER_TRADE * risk_cap_mult
                risk_amount_cash = equity * actual_risk_pct
                
                contract_size = 100 if asset_name == 'XAUUSD' else 100000
                # Simplified for USD quote pairs (EURUSD, GBPUSD, XAUUSD)
                lots = risk_amount_cash / (sl_dist * contract_size)
            
            lots = np.clip(lots, 0.01, 100.0)
            
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
                'alpha_val': alpha_val
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

    def on_order_execution(self, event):
        """Processes execution events (fills, closes) and updates local state."""
        from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOAExecutionType
        
        self.logger.info(f"Execution Event: {event.executionType}")
        
        # 1. Update Balance and Account State
        if hasattr(event, 'trader') and event.trader.balance:
            self.portfolio_state['balance'] = event.trader.balance / 100.0
            self.portfolio_state['equity'] = self.portfolio_state.get('equity', self.portfolio_state['balance'])
            self.logger.info(f"Account state updated via event. Balance: ${self.portfolio_state['balance']:.2f}")

        # 2. Update Position State
        if event.executionType in [ProtoOAExecutionType.ORDER_FILLED, ProtoOAExecutionType.ORDER_PARTIAL_FILL]:
            if hasattr(event, 'position'):
                pos = event.position
                symbol_id = pos.tradeData.symbolId
                self.active_positions[symbol_id] = pos.positionId
                self.logger.info(f"Position TRACKED: {pos.positionId} for symbol {symbol_id}")
        
        elif event.executionType == ProtoOAExecutionType.ORDER_CLOSED:
             if hasattr(event, 'position'):
                 pos_id = event.position.positionId
                 # Remove from active map
                 for sym_id, active_pos_id in list(self.active_positions.items()):
                     if active_pos_id == pos_id:
                         del self.active_positions[sym_id]
                         self.logger.info(f"Position CLOSED & UNTRACKED: {pos_id} for symbol {sym_id}")
                         break

    def on_order_error(self, event):
        """Handles order errors."""
        self.logger.error(f"Order Error: {event.errorCode} - {event.description}")
        self.notifier.send_error(f"Order Error: {event.description}")

    def is_asset_locked(self, symbol_id):
        """Checks if there's already an active position for this symbol."""
        return symbol_id in self.active_positions

    def _get_symbol_name(self, symbol_id):
        """Reverse mapping from symbolId to name."""
        inv_map = {v: k for k, v in self.client.symbol_ids.items()}
        return inv_map.get(symbol_id, "Unknown")
