import logging
import time
import numpy as np
from twisted.internet import reactor
from twisted.internet.defer import inlineCallbacks, gatherResults
from twisted.python.failure import Failure
from twisted.internet.task import LoopingCall, deferLater
from twisted.internet import threads
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOATradeSide
from LiveExecution.src.ctrader_client import CTraderClient, CTraderAmendError
from LiveExecution.src.database import DatabaseManager
from LiveExecution.src.logger import generate_correlation_id

class Orchestrator:
    """
    Coordinates data fetching, feature engineering, and sequential inference.

    Pipeline: Data → Filter (RF) → Alpha (LSTM/V7) → Execute
    SL/TP: Fixed ATR multipliers (default 2x / 4x)
    """
    def __init__(self, client, feature_manager, model_loader, config=None):
        self.logger = logging.getLogger("LiveExecution")
        self.client = client
        self.fm = feature_manager
        self.fm.client = client
        self.ml = model_loader
        self.config = config
        self.dashboard = None
        self.start_time = time.time()
        self.last_inference_time = 0

        # Database initialization
        db_path = self.config.get("DB_PATH", "LiveExecution/data/live_trading.db") if self.config else "LiveExecution/data/live_trading.db"
        self.db = DatabaseManager(db_path)

        # Internal state
        self.portfolio_state = {asset: {} for asset in self.fm.assets} if hasattr(self.fm, 'assets') else {}
        self.active_positions = {}
        self.entry_prices = {}
        self.pending_decisions = {}
        self._positions_pending_verify = set()  # pos_ids being verified by execute_decision

        # Price precision (digits) for each asset
        self.symbol_digits = {
            'EURUSD': 5,
            'GBPUSD': 5,
            'USDCHF': 5,
            'USDJPY': 3,
        }

    def set_dashboard(self, dashboard):
        """Links the dashboard server for real-time updates."""
        self.dashboard = dashboard

    @inlineCallbacks
    def bootstrap(self):
        """Called after cTrader authentication. Loads history, subscribes, syncs positions."""
        self.logger.info("--- STARTING ORCHESTRATOR BOOTSTRAP ---")

        from twisted.internet import threads

        # Stage 1: COT & Macro Data Fetch & Cache Warm-up (Off-reactor thread)
        self.logger.info("[Bootstrap Stage 1/5] Starting COT and Macro data cache warm-up...")
        try:
            yield threads.deferToThread(self.fm.initialize_market_data)
            self.logger.info("[Bootstrap Stage 1/5] COT and Macro data successfully loaded and validated.")
        except Exception as e:
            self.logger.exception(f"FATAL: [Bootstrap Stage 1/5] COT & Macro warm-up failed: {e}")
            self.logger.critical("Orchestrator cannot proceed to live trading without valid market indicators. Stopping client.")
            self.client.stop()
            raise e

        # Stage 2: Historical Candle Backfill per Asset
        self.logger.info("[Bootstrap Stage 2/5] Starting historical M5 candle backfill...")
        symbol_ids = list(self.client.symbol_ids.values())
        try:
            for asset_name, symbol_id in self.client.symbol_ids.items():
                self.logger.info(f"  Fetching historical candles for {asset_name}...")
                res = yield self.client.fetch_ohlcv(symbol_id, count=300)
                if hasattr(res, 'trendbar') and res.trendbar:
                    self.fm.update_data(symbol_id, res)
                    self.logger.info(
                        f"  Loaded {len(res.trendbar)} historical bars for {asset_name}"
                    )
                else:
                    raise ValueError(f"No historical candles returned for {asset_name}")

            self.logger.info(
                f"[Bootstrap Stage 2/5] Historical backfill complete. Buffer sizes: { {a: len(self.fm.history[a]) for a in self.fm.assets} }"
            )
        except Exception as e:
            self.logger.exception(f"FATAL: [Bootstrap Stage 2/5] Historical backfill failed: {e}")
            self.logger.critical("Orchestrator cannot trade without continuous historical bar history. Stopping client.")
            self.client.stop()
            raise e

        # Stage 3: Subscribe to live spots + M5 trendbars
        self.logger.info("[Bootstrap Stage 3/5] Subscribing to live spot events and M5 trendbars...")
        try:
            yield self.client.subscribe(symbol_ids)
            self.logger.info("[Bootstrap Stage 3/5] Subscriptions established successfully.")
        except Exception as e:
            self.logger.exception(f"FATAL: [Bootstrap Stage 3/5] Live subscription failed: {e}")
            self.logger.critical("Orchestrator cannot receive real-time ticks. Stopping client.")
            self.client.stop()
            raise e

        # Stage 4: Sync account and open positions
        self.logger.info("[Bootstrap Stage 4/5] Synchronizing account balance and open positions...")
        try:
            acct = yield self.client.fetch_account_summary()
            self.update_account_state(acct)
            yield self.sync_active_positions()
            self.logger.info(
                f"[Bootstrap Stage 4/5] Account synchronized. Balance={self.portfolio_state.get('balance', '?')}, "
                f"Open positions={len(self.active_positions)}"
            )
        except Exception as e:
            self.logger.exception(f"FATAL: [Bootstrap Stage 4/5] Account sync failed: {e}")
            self.logger.critical("Orchestrator could not determine risk boundaries. Stopping client.")
            self.client.stop()
            raise e

        # Stage 5: Feature Buffer Warm-up and Validation
        self.logger.info("[Bootstrap Stage 5/5] Performing feature buffer warm-up & validation...")
        try:
            for asset in self.fm.assets:
                # Attempt sequence calculation to verify no exceptions
                seq = self.fm.get_alpha_sequence(asset, self.ml.alpha_sequence_length)
                if seq is None:
                    raise ValueError(f"Failed to generate alpha sequence for {asset}. Insufficient bar history.")
                self.logger.info(f"  Warm-up validation passed for {asset}: sequence shape={seq.shape}")

            self.logger.info("[Bootstrap Stage 5/5] Feature buffer verification passed.")
        except Exception as e:
            self.logger.exception(f"FATAL: [Bootstrap Stage 5/5] Warm-up verification failed: {e}")
            self.logger.critical("Feature engine failed to build a valid starting observation matrix. Stopping client.")
            self.client.stop()
            raise e

        # Start daily macro/COT refresh
        self.fm.start_daily_refresh()

        self.logger.info("system ready, entering live loop")
        self.logger.info("--- ORCHESTRATOR BOOTSTRAP COMPLETE ---")

    def on_m5_candle_close(self, symbol_id, bar):
        """Called by ctrader_client when a new M5 candle closes."""
        try:
            asset = self.fm._get_asset_name_from_id(symbol_id)
            if not asset:
                return

            self.fm.update_from_trendbar(asset, bar)
            self.logger.debug(
                f"Candle pushed for {asset}. Buffer={len(self.fm.history[asset])}"
            )

            if self.fm.is_ready():
                decision = self.run_inference_chain(symbol_id)
                if decision and decision.get('action', 0) != 0:
                    self.logger.info(f"Triggering execution for {asset} on the main reactor thread...")
                    d = self.execute_decision(decision, symbol_id)
                    d.addErrback(lambda f: self.logger.error(f"Execution failed with error: {f.getErrorMessage()}\nTraceback: {f.getTraceback()}"))
        except Exception as e:
            self.logger.error(f"on_m5_candle_close error for {symbol_id}: {e}")


    def update_account_state(self, account_res):
        """Updates internal portfolio state from cTrader response."""
        self.portfolio_state['balance'] = account_res.trader.balance / 100.0
        self.portfolio_state['equity'] = self.portfolio_state.get('equity', self.portfolio_state['balance'])
        self.portfolio_state['initial_equity'] = self.portfolio_state.get('initial_equity', self.portfolio_state['equity'])
        self.portfolio_state['peak_equity'] = max(self.portfolio_state.get('peak_equity', 0), self.portfolio_state['equity'])

    @inlineCallbacks
    def on_order_execution(self, event):
        """Handles order execution events from cTrader."""
        try:
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
                        self.logger.info(f"Position {pos_id} OPEN for {asset_name}")

                        contract_size = 100000
                        lots = pos.tradeData.volume / (contract_size * 100)

                        pending = self.pending_decisions.get(asset_name, {})
                        sl_val = pending.get('sl')
                        tp_val = pending.get('tp')
                        rel_sl = pending.get('relative_sl')
                        rel_tp = pending.get('relative_tp')
                        conf_val = pending.get('confidence')

                        self.db.log_trade_opening(pos_id, asset_name, 'BUY' if pos.tradeData.tradeSide == 1 else 'SELL', lots, pos.price, sl=sl_val, tp=tp_val, relative_sl=rel_sl, relative_tp=rel_tp, confidence=conf_val)

                        has_sl = bool(hasattr(pos, 'stopLoss') and pos.stopLoss and pos.stopLoss > 0)
                        has_tp = bool(hasattr(pos, 'takeProfit') and pos.takeProfit and pos.takeProfit > 0)
                        if not (has_sl and has_tp) and pos_id not in self._positions_pending_verify:
                            self.logger.warning(f"Position {pos_id} ({asset_name}) opened without SL/TP. Attempting recovery...")
                            yield self._attach_missing_sltp_for_pos(pos)

                        if self.dashboard:
                            import asyncio
                            asyncio.run_coroutine_threadsafe(
                                self.dashboard.broadcast_update("trade_opened", {"symbol": asset_name}),
                                self.dashboard.app.loop if hasattr(self.dashboard.app, 'loop') else asyncio.get_event_loop()
                            )

            if hasattr(event, 'order') and event.order:
                order = event.order
                if order.closingOrder:
                    self.logger.debug(f"Order {order.orderId} is a CLOSING order. PnL: {getattr(order, 'moneyBalance', 0)/100.0}")

        except Exception as e:
            self.logger.error(f"Error handling execution event: {e}")

    def on_order_error(self, event):
        """Handles order error events from cTrader."""
        self.logger.error(f"Order rejected: {event.errorCode} - {getattr(event, 'description', 'No description')}")

    def is_asset_locked(self, symbol_id):
        return symbol_id in self.active_positions

    @inlineCallbacks
    def sync_active_positions(self):
        try:
            res = yield self.client.fetch_open_positions()
            new_active = {}
            new_entries = {}
            for pos in getattr(res, 'position', []):
                if hasattr(pos, 'tradeData') and hasattr(pos.tradeData, 'symbolId'):
                    sym_id = pos.tradeData.symbolId
                    pos_id = pos.positionId
                    new_active[sym_id] = pos_id
                    new_entries[pos_id] = pos.price

                    has_sl = bool(hasattr(pos, 'stopLoss') and pos.stopLoss and pos.stopLoss > 0)
                    has_tp = bool(hasattr(pos, 'takeProfit') and pos.takeProfit and pos.takeProfit > 0)

                    if not (has_sl and has_tp):
                        asset_name = self._get_symbol_name(sym_id)
                        self.logger.warning(f"Active position {pos_id} ({asset_name}) is missing SL or TP! Attaching...")
                        yield self._attach_missing_sltp_for_pos(pos)

            self.active_positions = new_active
            self.entry_prices = new_entries
            self.portfolio_state['num_open_positions'] = len(self.active_positions)
        except Exception as e:
            self.logger.error(f"Failed to sync active positions: {e}")

    @inlineCallbacks
    def _attach_missing_sltp_for_pos(self, pos):
        """Retrieves or calculates entry-relative SL/TP and attaches to position."""
        try:
            pos_id = pos.positionId
            symbol_id = pos.tradeData.symbolId
            asset_name = self._get_symbol_name(symbol_id)
            entry_price = pos.price
            trade_side = pos.tradeData.tradeSide
            direction = 1 if trade_side == 1 else -1

            db_trade = self.db.get_trade_by_pos_id(pos_id)
            sl_to_set = db_trade.get('sl') if db_trade else None
            tp_to_set = db_trade.get('tp') if db_trade else None

            digits = self.symbol_digits.get(asset_name, 5)

            if sl_to_set is None or tp_to_set is None:
                rel_sl = db_trade.get('relative_sl') if db_trade else None
                rel_tp = db_trade.get('relative_tp') if db_trade else None

                if rel_sl and sl_to_set is None:
                    sl_to_set = entry_price - (direction * rel_sl / 100000.0)
                if rel_tp and tp_to_set is None:
                    tp_to_set = entry_price + (direction * rel_tp / 100000.0)

            # Fallback calculation relative to ENTRY PRICE (not current price)
            if sl_to_set is None or tp_to_set is None:
                atr_scaled = self.fm.get_atr(asset_name) if hasattr(self, 'fm') and self.fm else 0
                if atr_scaled <= 0:
                    atr_scaled = entry_price * 0.0001
                sl_multiplier = getattr(self.ml, 'sl_multiplier', 2.0)
                tp_multiplier = getattr(self.ml, 'tp_multiplier', 4.0)

                sl_dist = sl_multiplier * atr_scaled
                tp_dist = tp_multiplier * atr_scaled
                step = 10 ** (5 - digits)

                relative_sl = max(int(round(sl_dist * 100000 / step) * step), step)
                relative_tp = max(int(round(tp_dist * 100000 / step) * step), step)

                if sl_to_set is None:
                    sl_to_set = round(entry_price - (direction * relative_sl / 100000.0), digits)
                if tp_to_set is None:
                    tp_to_set = round(entry_price + (direction * relative_tp / 100000.0), digits)

            sl_to_set = round(float(sl_to_set), digits)
            tp_to_set = round(float(tp_to_set), digits)

            if direction == 1:
                if sl_to_set >= entry_price or tp_to_set <= entry_price:
                    self.logger.error(f"Invalid SL/TP for LONG {pos_id} ({asset_name}): SL={sl_to_set}, entry={entry_price}, TP={tp_to_set}. Closing.")
                    vol = pos.tradeData.volume if hasattr(pos, 'tradeData') else 0
                    if vol > 0:
                        yield self.client.close_position(pos_id, vol)
                    return
            else:
                if sl_to_set <= entry_price or tp_to_set >= entry_price:
                    self.logger.error(f"Invalid SL/TP for SHORT {pos_id} ({asset_name}): SL={sl_to_set}, entry={entry_price}, TP={tp_to_set}. Closing.")
                    vol = pos.tradeData.volume if hasattr(pos, 'tradeData') else 0
                    if vol > 0:
                        yield self.client.close_position(pos_id, vol)
                    return

            # Update DB with attached levels
            contract_size = 100000
            lots = pos.tradeData.volume / (contract_size * 100)
            self.db.log_trade_opening(pos_id, asset_name, 'BUY' if trade_side == 1 else 'SELL', lots, entry_price, sl=sl_to_set, tp=tp_to_set)

            # Send amend request to cTrader
            try:
                yield self.client.amend_position_sltp(pos_id, stop_loss=sl_to_set, take_profit=tp_to_set)
                self.logger.info(f"Successfully attached SL={sl_to_set}, TP={tp_to_set} to position {pos_id} ({asset_name})")
            except CTraderAmendError as amend_err:
                self.logger.error(
                    f"Amend REJECTED for {pos_id} ({asset_name}): "
                    f"code={amend_err.error_code}, desc={amend_err.description}. Closing position..."
                )
                try:
                    vol = pos.tradeData.volume if hasattr(pos, 'tradeData') else 0
                    if vol > 0:
                        yield self.client.close_position(pos_id, vol)
                        self.logger.info(f"Position {pos_id} ({asset_name}) CLOSED because SL/TP amend was rejected.")
                except Exception as close_err:
                    self.logger.critical(f"FATAL: Could not close unprotected position {pos_id} ({asset_name}): {close_err}")
            except Exception as amend_err:
                self.logger.error(f"Failed to attach missing SL/TP to position {pos_id} ({asset_name}): {amend_err}. Closing position to protect capital...")
                try:
                    vol = pos.tradeData.volume if hasattr(pos, 'tradeData') else 0
                    if vol > 0:
                        yield self.client.close_position(pos_id, vol)
                        self.logger.info(f"Position {pos_id} ({asset_name}) CLOSED because SL/TP could not be attached.")
                except Exception as close_err:
                    self.logger.critical(f"FATAL: Could not close unprotected position {pos_id} ({asset_name}): {close_err}")

        except Exception as e:
            self.logger.error(f"Failed in _attach_missing_sltp_for_pos for pos {getattr(pos, 'positionId', 'unknown')}: {e}")

    @inlineCallbacks
    def execute_decision(self, decision, symbol_id):
        try:
            side = ProtoOATradeSide.BUY if decision['action'] == 1 else ProtoOATradeSide.SELL
            asset_name = decision['asset']
            self.pending_decisions[asset_name] = decision
            _pos_id = None  # track for cleanup

            self.logger.info(f"Placing {asset_name} {side} order. Lots: {decision['lots']:.2f}")
            contract_size = 100000
            raw_volume = decision['lots'] * contract_size * 100
            step = 100000
            volume = int(round(raw_volume / step) * step)

            execution_res = yield self.client.execute_market_order(symbol_id, volume, side, relative_sl=decision.get('relative_sl'), relative_tp=decision.get('relative_tp'))

            if hasattr(execution_res, 'position') and execution_res.position:
                pos = execution_res.position
                _pos_id = pos.positionId
                self.active_positions[symbol_id] = _pos_id
                self.entry_prices[_pos_id] = pos.price
                self._positions_pending_verify.add(_pos_id)

                # Ensure DB has SL/TP recorded
                lots = pos.tradeData.volume / (contract_size * 100) if hasattr(pos, 'tradeData') else decision['lots']
                self.db.log_trade_opening(_pos_id, asset_name, 'BUY' if side == ProtoOATradeSide.BUY else 'SELL', lots, pos.price, sl=decision['sl'], tp=decision['tp'], relative_sl=decision.get('relative_sl'), relative_tp=decision.get('relative_tp'), confidence=decision.get('confidence'))

                # Reconcile to get real SL/TP (new order response may have empty fields)
                has_sl = False
                has_tp = False
                try:
                    reconcile_res = yield self.client.fetch_open_positions()
                    for p in getattr(reconcile_res, 'position', []):
                        if p.positionId == _pos_id:
                            has_sl = bool(hasattr(p, 'stopLoss') and p.stopLoss and p.stopLoss > 0)
                            has_tp = bool(hasattr(p, 'takeProfit') and p.takeProfit and p.takeProfit > 0)
                            break
                except Exception:
                    # Fallback: check the execution response directly
                    has_sl = bool(hasattr(pos, 'stopLoss') and pos.stopLoss and pos.stopLoss > 0)
                    has_tp = bool(hasattr(pos, 'takeProfit') and pos.takeProfit and pos.takeProfit > 0)

                if not (has_sl and has_tp):
                    self.logger.warning(f"Position {_pos_id} ({asset_name}) has no SL/TP. Attaching...")
                    try:
                        yield self.client.amend_position_sltp(_pos_id, stop_loss=decision['sl'], take_profit=decision['tp'])
                    except CTraderAmendError as amend_err:
                        self.logger.error(
                            f"Amend REJECTED for {_pos_id} ({asset_name}): "
                            f"code={amend_err.error_code}, desc={amend_err.description}. Closing..."
                        )
                        try:
                            yield self.client.close_position(_pos_id, volume)
                            self.logger.info(f"Position {_pos_id} ({asset_name}) CLOSED because SL/TP amend was rejected.")
                        except Exception as close_err:
                            self.logger.critical(f"FATAL: Could not close unprotected position {_pos_id} ({asset_name}): {close_err}")
                    except Exception as amend_err:
                        self.logger.error(f"Failed to attach SL/TP to new position {_pos_id} ({asset_name}): {amend_err}. Closing position immediately...")
                        try:
                            yield self.client.close_position(_pos_id, volume)
                            self.logger.info(f"Position {_pos_id} ({asset_name}) CLOSED because SL/TP attachment failed.")
                        except Exception as close_err:
                            self.logger.critical(f"FATAL: Could not close unprotected position {_pos_id} ({asset_name}): {close_err}")

                self._positions_pending_verify.discard(_pos_id)

        except Exception as e:
            self.logger.error(f"Execution failed: {e}")
        finally:
            self.pending_decisions.pop(decision.get('asset'), None)
            if _pos_id is not None:
                self._positions_pending_verify.discard(_pos_id)



    def run_inference_chain(self, symbol_id):
        """Run the full inference pipeline: Filter → Alpha → Execute.

        Pipeline:
          1. Build RF filter features from OHLCV history
          2. Run RF filter — skip if confidence < 0.565
          3. Run Alpha model — skip if confidence < 0.60
          4. Execute only if filter and alpha agree on direction
          5. SL = 2x ATR, TP = 4x ATR (fixed defaults)
        """
        try:
            asset_name = self._get_symbol_name(symbol_id)

            # --- Step 2: Filter (RF ensemble) ---
            filter_conf = 1.0
            filter_passed = True
            if self.ml.filter_ensemble is not None:
                filter_features = self.fm.get_filter_features()
                if filter_features is None or len(filter_features) == 0:
                    self.logger.info(f"[CANDLE CLOSE] Asset: {asset_name} | No filter features available. Skipping.")
                    return {'action': 0}

                # Use the last bar's features for the current decision
                last_bar = filter_features[-1:]
                filter_out = self.ml.get_filter_signal(last_bar)
                try:
                    filter_conf = float(filter_out['confidence'])
                except Exception:
                    filter_conf = 1.0
                filter_passed = filter_out['should_trade']

            # --- Step 3: Alpha model (LSTM/V7) ---
            alpha_seq = self.fm.get_alpha_sequence(asset_name, self.ml.alpha_sequence_length)
            if alpha_seq is None:
                self.logger.info(
                    f"[CANDLE CLOSE] Asset: {asset_name} | "
                    f"Filter Ensemble: {'PASSED' if filter_passed else 'REJECTED'} (conf: {filter_conf:.3f}) | "
                    f"Alpha Model: Insufficient sequence history. Skipping."
                )
                return {'action': 0}

            alpha_out = self.ml.get_alpha_signal(alpha_seq, threshold=self.ml.alpha_threshold)
            try:
                trade_direction = int(alpha_out['direction'][0])
            except Exception:
                trade_direction = 0
            try:
                confidence = float(alpha_out['confidence'][0])
            except Exception:
                confidence = 0.0
            try:
                buy_p = float(alpha_out['buy_prob'][0])
            except Exception:
                buy_p = 0.0
            try:
                filter_thresh = float(self.ml.filter_threshold)
            except Exception:
                filter_thresh = 0.72
            try:
                alpha_thresh = float(self.ml.alpha_threshold)
            except Exception:
                alpha_thresh = 0.60

            alpha_direction = "BUY" if trade_direction == 1 else "SELL"
            alpha_passed = confidence >= alpha_thresh

            try:
                close_price = float(self.fm.history[asset_name].iloc[-1]['close']) if not self.fm.history[asset_name].empty else 0.0
            except Exception:
                close_price = 0.0

            # Log candle close event with predictions
            self.logger.info(
                f"[CANDLE CLOSE] Asset: {asset_name} | Close Price: {close_price:.5f} | "
                f"Filter Ensemble: {'PASSED' if filter_passed else 'REJECTED'} (conf: {filter_conf:.3f}, thresh: {filter_thresh:.3f}) | "
                f"Alpha Model: Direction={alpha_direction} (buy_prob: {buy_p:.3f}, conf: {confidence:.3f}, thresh: {alpha_thresh:.3f}, {'PASSED' if alpha_passed else 'REJECTED'})"
            )

            # Check gating decisions
            if not filter_passed:
                return {'action': 0}

            if not alpha_passed:
                return {'action': 0}

            # --- Step 4: SL/TP — fixed ATR multipliers ---
            digits = self.symbol_digits.get(asset_name, 5)
            real_price = self.fm.history[asset_name].iloc[-1]['close']
            atr_scaled = self.fm.get_atr(asset_name)
            if atr_scaled <= 0:
                atr_scaled = real_price * 0.0001

            sl_dist = self.ml.sl_multiplier * atr_scaled
            tp_dist = self.ml.tp_multiplier * atr_scaled
            step = 10 ** (5 - digits)

            relative_sl = max(int(round(sl_dist * 100000 / step) * step), step)
            relative_tp = max(int(round(tp_dist * 100000 / step) * step), step)

            sl_price = round(real_price - (trade_direction * relative_sl / 100000.0), digits)
            tp_price = round(real_price + (trade_direction * relative_tp / 100000.0), digits)

            # --- Step 5: Position sizing — fixed fraction of equity ---
            equity = self.portfolio_state.get('equity', 10.0)
            size_fraction = 0.10  # 10% of equity per trade
            position_size = equity * size_fraction
            position_value_usd = position_size * 100.0
            contract_size = 100000
            lot_value_usd = contract_size * real_price if asset_name in ['EURUSD', 'GBPUSD'] else contract_size
            lots = np.clip(position_value_usd / (lot_value_usd + 1e-9), 0.01, 100.0)

            return {
                'asset': asset_name,
                'action': 1 if trade_direction == 1 else 2,
                'lots': float(lots),
                'sl': float(sl_price),
                'tp': float(tp_price),
                'relative_sl': relative_sl,
                'relative_tp': relative_tp,
                'confidence': float(confidence),
            }
        except Exception as e:
            self.logger.error(f"Inference error for {asset_name}: {e}")
            return None

    def _get_symbol_name(self, symbol_id):
        # 1. Lookup in broker_symbol_map if populated
        if hasattr(self.client, 'broker_symbol_map') and symbol_id in self.client.broker_symbol_map:
            raw_name = self.client.broker_symbol_map[symbol_id]
            for asset in ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']:
                if asset in raw_name:
                    return asset
            return raw_name

        # 2. Lookup in hardcoded/default symbol_ids map
        inv_map = {v: k for k, v in self.client.symbol_ids.items()}
        raw_name = inv_map.get(symbol_id, "Unknown")
        for asset in ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']:
            if asset in raw_name:
                return asset
        return raw_name
