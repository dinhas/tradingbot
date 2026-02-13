import logging
import time
import numpy as np
from twisted.internet import reactor
from twisted.internet.defer import inlineCallbacks, gatherResults
from twisted.python.failure import Failure
from twisted.internet.task import LoopingCall, deferLater
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOATradeSide
from LiveExecution.src.database import DatabaseManager

class Orchestrator:
    """
    Coordinates data fetching, feature engineering, and sequential inference.
    """
    def __init__(self, client, feature_manager, model_loader, notifier, config=None):
        self.logger = logging.getLogger("LiveExecution")
        self.client = client
        self.fm = feature_manager
        self.ml = model_loader
        self.notifier = notifier
        self.config = config

        # Database
        db_path = config.get("DB_PATH", "LiveExecution/data/live_trading.db") if config else "LiveExecution/data/live_trading.db"
        self.db = DatabaseManager(db_path)
        
        # Internal state
        self.portfolio_state = {asset: {} for asset in self.fm.assets} 
        self.active_positions = {} 
        self.entry_prices = {} # Maps positionId to entry_price
        self.pnl_milestones = {} # Maps positionId to last notified % milestone
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
                    from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOAPositionStatus

                    # 1. Handle Closures
                    if pos.positionStatus in [ProtoOAPositionStatus.POSITION_STATUS_CLOSED, 6]: # 6=DELETED
                        self.logger.info(f"Position {pos_id} CLOSED for {asset_name}")

                        # Log to DB (Simplified PnL from event if available, otherwise just mark closed)
                        # cTrader event has moneyBalance (realized PnL) for closing orders
                        realized_pnl = 0
                        if hasattr(event, 'order') and event.order and hasattr(event.order, 'moneyBalance'):
                             realized_pnl = event.order.moneyBalance / 100.0

                        self.db.log_trade_closure(pos_id, pos.price, realized_pnl, realized_pnl, "CLOSED")

                        if symbol_id in self.active_positions:
                            del self.active_positions[symbol_id]

                    # 2. Handle Openings
                    elif pos.positionStatus == ProtoOAPositionStatus.POSITION_STATUS_OPEN:
                        # Position just opened or updated
                        self.active_positions[symbol_id] = pos_id
                        self.entry_prices[pos_id] = pos.price
                        self.logger.info(f"Position {pos_id} is now OPEN for {asset_name}")

                        # Log to DB
                        contract_size = 100 if asset_name == 'XAUUSD' else 100000
                        lots = pos.tradeData.volume / (contract_size * 100)

                        self.db.log_trade_opening(
                            pos_id,
                            asset_name,
                            'BUY' if pos.tradeData.tradeSide == 1 else 'SELL',
                            lots,
                            pos.price
                        )

            # If it's a trade closure, let's log the detail
            if hasattr(event, 'order') and event.order:
                 order = event.order
                 if order.closingOrder:
                      self.logger.info(f"Order {order.orderId} is a CLOSING order. PnL: {getattr(order, 'moneyBalance', 0)/100.0}")
                
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
            new_entries = {}
            for pos in res.position:
                if hasattr(pos, 'tradeData') and hasattr(pos.tradeData, 'symbolId'):
                    new_active[pos.tradeData.symbolId] = pos.positionId
                    new_entries[pos.positionId] = pos.price
            
            self.active_positions = new_active
            self.entry_prices = new_entries
            self.logger.info(f"Synced {len(self.active_positions)} active positions from API.")
            
            # Update portfolio state for features
            self.portfolio_state['num_open_positions'] = len(self.active_positions)
            # You could add more detail here (PnL, etc) if needed by features
            
        except Exception as e:
            self.logger.error(f"Failed to sync active positions: {e}")

    @inlineCallbacks
    def _poll_account_state(self):
        """Polls account summary and logs it to database."""
        try:
            acc_res = yield self.client.fetch_account_summary()
            self.update_account_state(acc_res)

            balance = self.portfolio_state.get('balance', 0)
            equity = self.portfolio_state.get('equity', balance)
            peak = self.portfolio_state.get('peak_equity', equity)
            drawdown = 1.0 - (equity / peak) if peak > 0 else 0

            self.db.log_account_state(balance, equity, drawdown, 0.0)
            self.logger.debug(f"Polled account state: Balance=${balance}, Equity=${equity}")

            # Check PnL Milestones
            self._check_pnl_milestones()

        except Exception as e:
            self.logger.error(f"Account polling failed: {e}")

    def _check_pnl_milestones(self):
        """Checks if any open position has crossed a 1% PnL milestone."""
        for symbol_id, pos_id in self.active_positions.items():
            if pos_id not in self.entry_prices or pos_id == "PENDING": continue

            asset = self._get_symbol_name(symbol_id)
            if asset not in self.fm.history or self.fm.history[asset].empty: continue

            entry_price = self.entry_prices[pos_id]
            current_price = self.fm.history[asset].iloc[-1]['close']

            # Determine direction from DB
            active_trades = self.db.get_active_trades()
            trade = next((t for t in active_trades if t['pos_id'] == pos_id), None)
            if not trade: continue

            direction = 1 if trade['action'] == 'BUY' else -1
            pnl_pct = (current_price - entry_price) / entry_price * direction * 100.0

            last_m = self.pnl_milestones.get(pos_id, 0)
            current_m = int(pnl_pct)

            if abs(current_m - last_m) >= 1:
                self.pnl_milestones[pos_id] = current_m
                emoji = "📈" if current_m > last_m else "📉"
                self.notifier.send_message(f"{emoji} **PnL Milestone:** `{asset}` is now `{current_m:+.0f}%` ({pnl_pct:+.2f}%)")

    def _send_pulse_check(self):
        """Sends a recurring health check message."""
        balance = self.portfolio_state.get('balance', 0)
        num_pos = len(self.active_positions)
        msg = (
            "💚 **System Pulse Check**\n"
            "Status: `All systems nominal`\n"
            f"Balance: `${balance:,.2f}`\n"
            f"Active Positions: `{num_pos}`"
        )
        self.notifier.send_message(msg)

    def _check_daily_summary(self):
        """Checks if it's time to send the daily performance summary."""
        from datetime import datetime
        now = datetime.utcnow()
        if now.hour == 0 and not getattr(self, '_daily_summary_sent', False):
            self._send_daily_summary()
            self._daily_summary_sent = True
        elif now.hour != 0:
            self._daily_summary_sent = False

    def _send_daily_summary(self):
        """Sends a summary of the last 24 hours of trading."""
        recent = self.db.get_recent_trades(limit=100)
        # Filter for trades closed in the last 24 hours
        from datetime import datetime, timedelta
        yesterday = (datetime.utcnow() - timedelta(days=1)).isoformat()

        daily_trades = [t for t in recent if t['exit_time'] and t['exit_time'] > yesterday]

        total_pnl = sum(t['net_pnl'] for t in daily_trades)
        wins = sum(1 for t in daily_trades if t['net_pnl'] > 0)
        total = len(daily_trades)
        win_rate = (wins / total) if total > 0 else 0

        msg = (
            "📅 **Daily Performance Summary**\n"
            f"Total Trades: `{total}`\n"
            f"Win Rate: `{win_rate:.1%}`\n"
            f"Daily PnL: `${total_pnl:+.2f}`\n"
            f"Current Balance: `${self.portfolio_state.get('balance', 0):,.2f}`"
        )
        self.notifier.send_message(msg)

    @inlineCallbacks
    def bootstrap(self):
        """Pre-fetches historical data for all assets to be ready for inference."""
        self.logger.info("Bootstrapping history for all assets...")
        try:
            # 1. Sync Positions and Account State
            yield self.sync_active_positions()
            acc_res = yield self.client.fetch_account_summary()
            self.update_account_state(acc_res)
            
            # Start 60s Account Poller
            self.poller = LoopingCall(self._poll_account_state)
            self.poller.start(60.0, now=False)

            # Start 2h Pulse Check
            self.pulse_timer = LoopingCall(self._send_pulse_check)
            self.pulse_timer.start(7200.0, now=False)

            # Start 1h Daily Summary Checker
            self.daily_timer = LoopingCall(self._check_daily_summary)
            self.daily_timer.start(3600.0, now=False)

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
        # Stagger requests slightly to avoid bursts
        jitter = (symbol_id % 5) * 0.5 # Up to 2s stagger
        yield deferLater(reactor, jitter, lambda: None)
        
        start_time = time.time()
        asset_name = self._get_symbol_name(symbol_id)
        self.logger.info(f"--- M5 Close Detected: {asset_name} ({symbol_id}) ---")
        
        # 0. Check Readiness
        if not self.fm.is_ready():
             self.logger.info("FeatureManager not ready (insufficient history). Skipping.")
             return

        # 1. Sync Positions and Check System/Asset Limits
        # Only sync once every 30 seconds across all symbol triggers to avoid redundancy
        now_ts = time.time()
        if not hasattr(self, '_last_sync_ts') or now_ts - self._last_sync_ts > 30:
            self._last_sync_ts = now_ts
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
            
            # Only fetch account summary if not recently updated
            if not hasattr(self, '_last_account_refresh') or now_ts - self._last_account_refresh > 30:
                self._last_account_refresh = now_ts
                d_account = self.client.fetch_account_summary()
                results = yield gatherResults([d_ohlcv, d_account], consumeErrors=True)
            else:
                results = yield gatherResults([d_ohlcv], consumeErrors=True)
                # Use current portfolio state as account res if we didn't fetch it
                results.append(None) 
            
            # Check for errors in results
            for res in results:
                if res is not None and isinstance(res, (Exception, Failure)):
                     self.logger.error(f"Data fetch failed: {res}")
                     self.notifier.send_error(f"Data fetch failed for {asset_name}: {res}")
                     return

            ohlcv_res = results[0]
            account_res = results[1]
            
            # 3. Update State
            self.fm.update_data(symbol_id, ohlcv_res)
            if account_res:
                self.update_account_state(account_res)
                self._last_account_refresh = time.time()
            
            # 4. Run Inference
            inference_start = time.time()
            decision = self.run_inference_chain(symbol_id)
            inference_time = time.time() - inference_start
            
            if not decision or decision.get('action') == 0:
                self.logger.info(f"Inference complete in {inference_time:.3f}s. No action taken.")
                return 
            
            # 5. Execute & Notify
            yield self.execute_decision(decision, symbol_id)
            
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
            asset_name = decision['asset']
            
            # Convert Lots to cTrader volume (Units * 100)
            # Forex: 1 Lot = 100,000 units
            # Gold: 1 Lot = 100 units
            contract_size = 100 if asset_name == 'XAUUSD' else 100000

            raw_volume = decision['lots'] * contract_size * 100

            # Round volume to nearest step (cTrader step requirement)
            # For Forex: 0.01 lots = 1,000 units = 100,000 protocol volume
            # For Gold: 0.01 lots = 1 unit = 100 protocol volume
            step = 100 if asset_name == 'XAUUSD' else 100000
            volume = int(round(raw_volume / step) * step)
            
            # Ensure minimum volume
            if volume < step:
                volume = step
            
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
            # 1. Get Alpha Observation (40)
            alpha_obs = self.fm.get_alpha_observation(asset_name, self.portfolio_state)
            
            # 2. Alpha Prediction
            alpha_actions = self.ml.get_alpha_action(alpha_obs)
            alpha_val = alpha_actions[0]
            
            # Parse Alpha Direction (Match Backtest: > 0.33 Buy, < -0.33 Sell)
            direction = 1 if alpha_val > 0.33 else (-1 if alpha_val < -0.33 else 0)
            
            self.logger.info(f"Alpha predicted direction for {asset_name}: {direction} (raw: {alpha_val:.4f})")
            
            if direction == 0:
                return {'action': 0, 'allowed': False, 'reason': 'Alpha Hold'}
            
            # 3. Get Risk Observation (60)
            risk_obs = self.fm.get_risk_observation(asset_name, alpha_obs)
            
            # 4. Risk Prediction (SL Model)
            risk_action = self.ml.get_risk_action(risk_obs)
            
            sl_mult = risk_action['sl_mult']
            tp_mult = risk_action['tp_mult']
            size_out = risk_action['size']
            
            # Blocking Logic (Threshold: 0.10 confidence)
            if size_out < 0.10:
                self.logger.info(f"Risk Block for {asset_name}: confidence {size_out:.4f} < 0.10")
                return {'action': 0, 'allowed': False, 'reason': f'Risk Block ({size_out:.2f})'}

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
            
            # Calculate Relative values for API (Price Distance in Points)
            multiplier = 10 ** digits
            relative_sl = int(round(sl_dist * multiplier))
            relative_tp = int(round(tp_dist * multiplier))
            
            # Lot Calculation (Match Backtest calculate_position_size)
            equity = self.portfolio_state.get('equity', 10.0)
            MAX_LEVERAGE = 100.0
            
            # Final_Size = Equity * size from model (0.0 to 1.0)
            position_size = equity * size_out
            position_value_usd = position_size * MAX_LEVERAGE

            # Calculate lot_value_usd
            contract_size = 100 if asset_name == 'XAUUSD' else 100000
            is_usd_quote = asset_name in ['EURUSD', 'GBPUSD', 'XAUUSD']
            if is_usd_quote:
                lot_value_usd = contract_size * current_price
            else:
                lot_value_usd = contract_size
                
            lots = position_value_usd / (lot_value_usd + 1e-9)
            lots = np.clip(lots, 0.01, 100.0)
            
            return {
                'symbol_id': symbol_id,
                'asset': asset_name,
                'action': 1 if direction == 1 else 2,
                'lots': float(lots),
                'sl': float(sl_price),
                'tp': float(tp_price),
                'relative_sl': relative_sl,
                'relative_tp': relative_tp,
                'risk_actions': risk_action
            }
            
        except Exception as e:
            self.logger.error(f"Error in inference chain for symbol {symbol_id}: {e}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error in inference chain for symbol {symbol_id}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    @inlineCallbacks
    def close_position_by_id(self, pos_id, symbol_id):
        """Manually closes a position."""
        self.logger.warning(f"Manual close requested for position {pos_id}")
        try:
            # We need the volume to close.
            res = yield self.client.fetch_open_positions()
            target_pos = None
            for pos in res.position:
                if pos.positionId == pos_id:
                    target_pos = pos
                    break

            if target_pos:
                yield self.client.close_position(pos_id, target_pos.tradeData.volume)
                self.notifier.send_message(f"🗑️ **Manual Close:** Position `{pos_id}` closed.")
            else:
                self.logger.error(f"Could not find position {pos_id} to close.")
        except Exception as e:
            self.logger.error(f"Failed to close position {pos_id}: {e}")

    @inlineCallbacks
    def kill_switch(self):
        """Closes ALL positions and stops the system."""
        self.logger.critical("!!! KILL SWITCH ACTIVATED !!!")
        self.notifier.send_message("🚨 **KILL SWITCH ACTIVATED!** Closing all positions and stopping...")

        try:
            res = yield self.client.fetch_open_positions()
            for pos in res.position:
                yield self.client.close_position(pos.positionId, pos.tradeData.volume)

            self.logger.info("All positions closed. Stopping system in 2s.")
            reactor.callLater(2, reactor.stop)
        except Exception as e:
            self.logger.error(f"Kill switch failed to close some positions: {e}")
            reactor.callLater(2, reactor.stop)

    def stop(self):
        """Stops the orchestrator and client."""
        self.logger.info("Stopping Orchestrator...")
        self.notifier.send_message("🛑 **System Stopping...**")
        if hasattr(self, 'poller') and self.poller.running:
            self.poller.stop()
        self.client.stop()

    def _get_symbol_name(self, symbol_id):
        """Reverse mapping from symbolId to name."""
        inv_map = {v: k for k, v in self.client.symbol_ids.items()}
        return inv_map.get(symbol_id, "Unknown")
