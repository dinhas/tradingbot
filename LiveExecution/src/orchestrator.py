import logging
import time
import numpy as np
from twisted.internet import reactor
from twisted.internet.defer import inlineCallbacks, gatherResults
from twisted.python.failure import Failure
from twisted.internet.task import LoopingCall, deferLater
from twisted.internet import threads
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOATradeSide
from LiveExecution.src.database import DatabaseManager
from LiveExecution.src.logger import generate_correlation_id

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
        self.dashboard = None
        self.start_time = time.time()
        self.last_inference_time = 0

    def set_dashboard(self, dashboard):
        """Links the dashboard server for real-time updates."""
        self.dashboard = dashboard

        # Database
        db_path = self.config.get("DB_PATH", "LiveExecution/data/live_trading.db") if self.config else "LiveExecution/data/live_trading.db"
        self.db = DatabaseManager(db_path)
        
        # Internal state
        self.portfolio_state = {asset: {} for asset in self.fm.assets} 
        self.active_positions = {} 
        self.entry_prices = {} # Maps positionId to entry_price
        self.pnl_milestones = {} # Maps positionId to last notified % milestone
        self.pending_risk_actions = {} # Maps positionId to risk_actions
        
        # Price precision (digits) for each asset (Match native cTrader precision)
        self.symbol_digits = {
            'EURUSD': 5,
            'GBPUSD': 5,
            'USDCHF': 5,
            'USDJPY': 3,
            'XAUUSD': 2 # Gold is 2 digits on most cTrader brokers
        }

    def update_account_state(self, account_res):
        """Updates internal portfolio state from cTrader response."""
        self.portfolio_state['balance'] = account_res.trader.balance / 100.0
        self.portfolio_state['equity'] = self.portfolio_state.get('equity', self.portfolio_state['balance'])
        self.portfolio_state['initial_equity'] = self.portfolio_state.get('initial_equity', self.portfolio_state['equity'])
        self.portfolio_state['peak_equity'] = max(self.portfolio_state.get('peak_equity', 0), self.portfolio_state['equity'])

    def on_order_execution(self, event):
        """Handles order execution events from cTrader."""
        try:
            self.logger.info(f"=== ORDER EXECUTION EVENT ===")
            
            if event.position:
                pos = event.position
                pos_id = pos.positionId
                symbol_id = pos.tradeData.symbolId
                asset_name = self._get_symbol_name(symbol_id)
                
                from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOAExecutionType
                
                if event.executionType == ProtoOAExecutionType.ORDER_FILLED:
                    from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOAPositionStatus

                    if pos.positionStatus in [ProtoOAPositionStatus.POSITION_STATUS_CLOSED, 6]: 
                        self.logger.info(f"Position {pos_id} CLOSED for {asset_name}")

                        realized_pnl = 0
                        reason = "SIGNAL" 
                        if hasattr(event, 'order') and event.order:
                             if hasattr(event.order, 'moneyBalance'):
                                 realized_pnl = event.order.moneyBalance / 100.0
                             
                             from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOAOrderType
                             if event.order.orderType == ProtoOAOrderType.STOP:
                                 reason = "SL"
                             elif event.order.orderType == ProtoOAOrderType.LIMIT:
                                 reason = "TP"
                             elif event.order.orderType == ProtoOAOrderType.MARKET:
                                 reason = "SIGNAL/MANUAL"

                        self.db.log_trade_closure(pos_id, pos.price, realized_pnl, realized_pnl, reason)

                        self.notifier.send_trade_closed({
                            'symbol': asset_name,
                            'pnl': realized_pnl,
                            'reason': reason
                        })

                        if self.dashboard:
                            import asyncio
                            asyncio.run_coroutine_threadsafe(
                                self.dashboard.broadcast_update("trade_closed", {"symbol": asset_name, "pnl": realized_pnl}),
                                self.dashboard.app.loop if hasattr(self.dashboard.app, 'loop') else asyncio.get_event_loop()
                            )

                        if symbol_id in self.active_positions:
                            del self.active_positions[symbol_id]

                    elif pos.positionStatus == ProtoOAPositionStatus.POSITION_STATUS_OPEN:
                        self.active_positions[symbol_id] = pos_id
                        self.entry_prices[pos_id] = pos.price
                        self.logger.info(f"Position {pos_id} is now OPEN for {asset_name}")

                        contract_size = 100 if asset_name == 'XAUUSD' else 100000
                        lots = pos.tradeData.volume / (contract_size * 100)

                        self.db.log_trade_opening(pos_id, asset_name, 'BUY' if pos.tradeData.tradeSide == 1 else 'SELL', lots, pos.price)

                        if self.dashboard:
                            import asyncio
                            asyncio.run_coroutine_threadsafe(
                                self.dashboard.broadcast_update("trade_opened", {"symbol": asset_name}),
                                self.dashboard.app.loop if hasattr(self.dashboard.app, 'loop') else asyncio.get_event_loop()
                            )

            if hasattr(event, 'order') and event.order:
                 order = event.order
                 if order.closingOrder:
                      self.logger.info(f"Order {order.orderId} is a CLOSING order. PnL: {getattr(order, 'moneyBalance', 0)/100.0}")
                
        except Exception as e:
            self.logger.error(f"Error handling execution event: {e}")

    def on_order_error(self, event):
        """Handles order error events from cTrader."""
        self.logger.error(f"=== ORDER ERROR EVENT ===")
        self.notifier.send_error(f"Order rejected: {event.errorCode} - {getattr(event, 'description', 'No description')}")

    def is_asset_locked(self, symbol_id):
        return symbol_id in self.active_positions
    
    @inlineCallbacks
    def sync_active_positions(self):
        try:
            res = yield self.client.fetch_open_positions()
            new_active = {}
            new_entries = {}
            for pos in res.position:
                if hasattr(pos, 'tradeData') and hasattr(pos.tradeData, 'symbolId'):
                    new_active[pos.tradeData.symbolId] = pos.positionId
                    new_entries[pos.positionId] = pos.price
            
            self.active_positions = new_active
            self.entry_prices = new_entries
            self.portfolio_state['num_open_positions'] = len(self.active_positions)
        except Exception as e:
            self.logger.error(f"Failed to sync active positions: {e}")

    @inlineCallbacks
    def _poll_account_state(self):
        try:
            acc_res = yield self.client.fetch_account_summary()
            self.update_account_state(acc_res)

            balance = self.portfolio_state.get('balance', 0)
            equity = self.portfolio_state.get('equity', balance)
            peak = self.portfolio_state.get('peak_equity', equity)
            drawdown = 1.0 - (equity / peak) if peak > 0 else 0

            self.db.log_account_state(balance, equity, drawdown, 0.0)

            if self.dashboard:
                import asyncio
                asyncio.run_coroutine_threadsafe(
                    self.dashboard.broadcast_update("equity_update", {"equity": equity, "balance": balance}),
                    self.dashboard.app.loop if hasattr(self.dashboard.app, 'loop') else asyncio.get_event_loop()
                )

            self._check_drawdown_alert(drawdown)
            self._check_pnl_milestones()
        except Exception as e:
            self.logger.error(f"Account polling failed: {e}")

    def _check_drawdown_alert(self, current_drawdown):
        threshold = self.config.get('TELEGRAM_DRAWDOWN_ALERT', 5.0) / 100.0
        if current_drawdown >= threshold:
            last_alert = getattr(self, '_last_drawdown_alert_time', 0)
            if time.time() - last_alert > 3600:
                self.notifier.send_message(f"⚠️ **DRAWDOWN ALERT**: Current drawdown is `{current_drawdown:.2%}` (Threshold: `{threshold:.2%}`)")
                self._last_drawdown_alert_time = time.time()

    def _check_pnl_milestones(self):
        for symbol_id, pos_id in self.active_positions.items():
            if pos_id not in self.entry_prices or pos_id == "PENDING": continue
            asset = self._get_symbol_name(symbol_id)
            if asset not in self.fm.history or self.fm.history[asset].empty: continue

            entry_price = self.entry_prices[pos_id]
            current_price = self.fm.history[asset].iloc[-1]['close']

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
        from datetime import datetime
        now = datetime.utcnow()
        if now.hour == 0 and not getattr(self, '_daily_summary_sent', False):
            self._send_daily_summary()
            self._daily_summary_sent = True
        elif now.hour != 0:
            self._daily_summary_sent = False

    def _send_daily_summary(self):
        recent = self.db.get_recent_trades(limit=100)
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
        self.logger.info("Bootstrapping history for all assets...")
        try:
            yield self.sync_active_positions()
            acc_res = yield self.client.fetch_account_summary()
            self.update_account_state(acc_res)
            
            self.poller = LoopingCall(self._poll_account_state)
            self.poller.start(60.0, now=False)

            self.pulse_timer = LoopingCall(self._send_pulse_check)
            self.pulse_timer.start(7200.0, now=False)

            self.daily_timer = LoopingCall(self._check_daily_summary)
            self.daily_timer.start(3600.0, now=False)

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
                    if res.trendbar:
                        latest_ts = max(bar.utcTimestampInMinutes for bar in res.trendbar)
                        self.client.last_bar_timestamps[symbol_id] = latest_ts
                    symbols_to_subscribe.append(symbol_id)
            
            if symbols_to_subscribe:
                yield self.client.subscribe(symbols_to_subscribe)
            
            if self.fm.is_ready():
                self.notifier.send_message("System is READY for live execution.")
                
        except Exception as e:
            self.logger.error(f"Error during bootstrap: {e}")

    @inlineCallbacks
    def on_m5_candle_close(self, symbol_id, trendbar):
        start_time = time.time()
        asset_name = self._get_symbol_name(symbol_id)
        self.logger.info(f"--- M5 Close Detected: {asset_name} ({symbol_id}) ---")
        
        if not self.fm.is_ready():
            self.logger.info(f"Feature manager not ready for {asset_name}. Skipping.")
            return

        self.fm.update_from_trendbar(asset_name, trendbar)

        if len(self.active_positions) >= 5 and symbol_id not in self.active_positions:
            self.logger.info(f"Max active positions (5) reached. Skipping {asset_name}.")
            return
        if self.is_asset_locked(symbol_id):
            self.logger.info(f"Asset {asset_name} is locked (already has an active position). Skipping.")
            return

        try:
            decision = yield threads.deferToThread(self.run_inference_chain, symbol_id)
            self.last_inference_time = time.time()
            
            if decision and decision.get('action') != 0:
                self.logger.info(f"Executing trade for {asset_name}: {decision}")
                yield self.execute_decision(decision, symbol_id)
            elif decision and decision.get('action') == 0:
                self.logger.debug(f"No trade action for {asset_name} based on inference results.")
            
            reactor.callLater(240.0, self.sync_active_positions)
        except Exception as e:
            self.logger.error(f"Orchestration error for {asset_name}: {e}")

    @inlineCallbacks
    def execute_decision(self, decision, symbol_id):
        try:
            side = ProtoOATradeSide.BUY if decision['action'] == 1 else ProtoOATradeSide.SELL
            asset_name = decision['asset']
            self.logger.info(f"Placing {asset_name} {side} order. Lots: {decision['lots']:.2f}")
            contract_size = 100 if asset_name == 'XAUUSD' else 100000
            raw_volume = decision['lots'] * contract_size * 100
            step = 100 if asset_name == 'XAUUSD' else 100000
            volume = int(round(raw_volume / step) * step)
            
            execution_res = yield self.client.execute_market_order(symbol_id, volume, side, relative_sl=decision.get('relative_sl'), relative_tp=decision.get('relative_tp'))
            
            digits = self.symbol_digits.get(asset_name, 5)
            current_price = self.fm.history[asset_name].iloc[-1]['close']

            self.notifier.send_trade_event({
                'symbol': asset_name,
                'action': 'BUY' if side == ProtoOATradeSide.BUY else 'SELL',
                'size': f"{decision['lots']:.2f} lots",
                'entry_price': round(current_price, digits),
                'sl': decision.get('sl'),
                'tp': decision.get('tp')
            })
            
            if hasattr(execution_res, 'position') and execution_res.position:
                self.active_positions[symbol_id] = execution_res.position.positionId
        except Exception as e:
            self.logger.error(f"Execution failed: {e}")

    def run_inference_chain(self, symbol_id):
        try:
            asset_name = self._get_symbol_name(symbol_id)
            alpha_obs = self.fm.get_alpha_observation(asset_name, self.portfolio_state)
            alpha_out = self.ml.get_alpha_action(alpha_obs)
            direction = int(alpha_out['direction'][0])
            quality = float(alpha_out['quality'][0])
            meta = float(alpha_out['meta'][0])
            
            self.logger.debug(f"Inference results for {asset_name}: Direction={direction}, Quality={quality:.3f}, Meta={meta:.3f}")
            
            if direction == 0:
                self.logger.info(f"Model predicted neutral direction for {asset_name}. Skipping.")
                return {'action': 0}
            
            meta_thresh = self.config.get('META_THRESHOLD', 0.7071)
            qual_thresh = self.config.get('QUAL_THRESHOLD', 0.70)
            
            if meta < meta_thresh:
                self._notify_threshold_breach(asset_name, "Meta", meta, meta_thresh)
                return {'action': 0}
            if quality < qual_thresh:
                self._notify_threshold_breach(asset_name, "Quality", quality, qual_thresh)
                return {'action': 0}
            
            risk_obs = self.fm.get_risk_observation(asset_name, alpha_obs)
            risk_action = self.ml.get_risk_action(risk_obs)
            size_out = risk_action['size']
            
            self.logger.debug(f"Risk results for {asset_name}: Size={size_out:.3f}, SL_Mult={risk_action['sl_mult']:.2f}, TP_Mult={risk_action['tp_mult']:.2f}")
            
            risk_thresh = self.config.get('RISK_THRESHOLD', 0.10)
            if size_out < risk_thresh:
                self._notify_threshold_breach(asset_name, "Risk", size_out, risk_thresh)
                return {'action': 0}

            digits = self.symbol_digits.get(asset_name, 5)
            real_price = self.fm.history[asset_name].iloc[-1]['close']
            atr_scaled = self.fm.get_atr(asset_name)
            if atr_scaled <= 0: atr_scaled = real_price * 0.0001
            
            sl_dist = risk_action['sl_mult'] * atr_scaled
            tp_dist = risk_action['tp_mult'] * atr_scaled
            step = 10**(5 - digits)
            
            relative_sl = max(int(round(sl_dist * 100000 / step) * step), step)
            relative_tp = max(int(round(tp_dist * 100000 / step) * step), step)

            sl_price = round(real_price - (direction * relative_sl / 100000.0), digits)
            tp_price = round(real_price + (direction * relative_tp / 100000.0), digits)
            
            equity = self.portfolio_state.get('equity', 10.0)
            position_size = equity * size_out
            position_value_usd = position_size * 100.0
            contract_size = 100 if asset_name == 'XAUUSD' else 100000
            lot_value_usd = contract_size * real_price if asset_name in ['EURUSD', 'GBPUSD', 'XAUUSD'] else contract_size
            lots = np.clip(position_value_usd / (lot_value_usd + 1e-9), 0.01, 100.0)
            
            return {'asset': asset_name, 'action': 1 if direction == 1 else 2, 'lots': float(lots), 'sl': float(sl_price), 'tp': float(tp_price), 'relative_sl': relative_sl, 'relative_tp': relative_tp}
        except Exception as e:
            self.logger.error(f"Inference error for {asset_name}: {e}")
            return None

    def _notify_threshold_breach(self, asset, name, value, threshold):
        key = f"thresh_{asset}_{name}"
        if not hasattr(self, '_last_thresh_alerts'): self._last_thresh_alerts = {}
        last = self._last_thresh_alerts.get(key, 0)
        
        # Always log at DEBUG level
        self.logger.debug(f"THRESHOLD CHECK: {asset} {name} {value:.3f} (Threshold: {threshold:.3f})")
        
        if time.time() - last > 300:
             self._last_thresh_alerts[key] = time.time()
             self.logger.warning(f"THRESHOLD BREACH: {asset} {name} {value:.3f} < {threshold:.3f}")

    def _get_symbol_name(self, symbol_id):
        inv_map = {v: k for k, v in self.client.symbol_ids.items()}
        return inv_map.get(symbol_id, "Unknown")
