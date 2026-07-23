from twisted.internet import reactor, defer
from twisted.internet.defer import inlineCallbacks
from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
from ctrader_open_api.messages.OpenApiMessages_pb2 import *
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import *
import logging

class CTraderClient:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("LiveExecution")
        
        # Callbacks
        self.on_candle_closed = None # To be set by orchestrator
        self.on_authenticated = None
        self.on_order_execution = None
        self.on_order_error = None
        
        # State tracking for deduplication
        self.last_bar_timestamps = {}
        
        self.app_id = config["CT_APP_ID"]
        self.app_secret = config["CT_APP_SECRET"]
        self.account_id = config["CT_ACCOUNT_ID"]
        self.access_token = config["CT_ACCESS_TOKEN"]
        
        # Asset Universe (as per PRD)
        self.symbol_ids = {
            'EURUSD': 1,
            'GBPUSD': 2,
            'XAUUSD': 41,
            'USDCHF': 6,
            'USDJPY': 4
        }
        
        # Reconnection parameters
        self.max_retries = 5
        self.retry_count = 0
        self.base_delay = 5.0 # Seconds
        
        # Heartbeat parameters
        self.heartbeat_interval = 25.0 # Seconds
        self.heartbeat_timer = None
        
        # Determine host
        self.host = EndPoints.PROTOBUF_LIVE_HOST if config["CT_HOST_TYPE"] == "live" else EndPoints.PROTOBUF_DEMO_HOST
        self.port = EndPoints.PROTOBUF_PORT
        
        # Initialize client
        self.client = Client(self.host, self.port, TcpProtocol)
        
    def start(self):
        """Starts the Twisted client service and sets callbacks."""
        self.client.setConnectedCallback(self._on_connected)
        self.client.setDisconnectedCallback(self._on_disconnected)
        self.client.setMessageReceivedCallback(self._on_message)
        self.client.startService()
        self.logger.info(f"Connecting to cTrader ({self.host}:{self.port})...")
    
    @inlineCallbacks
    def _on_connected(self, client):
        self.logger.info("Connected to cTrader. Authenticating...")
        self.retry_count = 0 # Reset retry count on successful connection
        
        try:
            # 1. Application Auth
            auth_req = ProtoOAApplicationAuthReq()
            auth_req.clientId = self.app_id
            auth_req.clientSecret = self.app_secret
            yield self.client.send(auth_req)
            self.logger.info("App Auth Success.")

            # 2. Account Auth
            acc_auth_req = ProtoOAAccountAuthReq()
            acc_auth_req.ctidTraderAccountId = self.account_id
            acc_auth_req.accessToken = self.access_token
            yield self.client.send(acc_auth_req)
            self.logger.info("Account Auth Success.")
            
            if self.on_authenticated:
                self.on_authenticated()
            
            self._start_heartbeat()
            
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            self.stop()
        
    def _on_disconnected(self, client, reason):
        self.logger.warning(f"Disconnected from cTrader: {reason}")
        self._stop_heartbeat()
        
        if self.retry_count < self.max_retries:
            self.retry_count += 1
            delay = self.base_delay * (2 ** (self.retry_count - 1))
            self.logger.info(f"Reconnecting in {delay} seconds (Attempt {self.retry_count}/{self.max_retries})...")
            reactor.callLater(delay, self.start)
        else:
            self.logger.error("Max reconnection retries reached. Stopping system.")
            # In a real scenario, we might want to trigger a notification here
            self.stop()
        
    def _on_message(self, client, message):
        """Handles incoming messages from cTrader."""
        try:
            payload = Protobuf.extract(message)
            
            if isinstance(payload, ProtoOASpotEvent):
                self._handle_spot_event(payload)
            elif isinstance(payload, ProtoOAExecutionEvent):
                if self.on_order_execution:
                    self.on_order_execution(payload)
            elif isinstance(payload, ProtoOAOrderErrorEvent):
                if self.on_order_error:
                    self.on_order_error(payload)
            
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            
    def _handle_spot_event(self, event):
        """Processes Spot Events for trendbar closes."""
        if event.trendbar:
            for bar in event.trendbar:
                if bar.period == ProtoOATrendbarPeriod.M5:
                    symbol_id = event.symbolId
                    # utcTimestampInMinutes identifies the bar (e.g. 07:10 for the 07:10-07:15 bar)
                    # We only process if this is a NEW bar we haven't seen yet.
                    current_ts = bar.utcTimestampInMinutes
                   
                    last_ts = self.last_bar_timestamps.get(symbol_id)
                    
                    if last_ts is None or current_ts > last_ts:
                        self.logger.info(f"M5 Candle closed for symbol {symbol_id} (TS: {current_ts})")
                        self.last_bar_timestamps[symbol_id] = current_ts
                        try:
                            if self.on_candle_closed:
                                self.on_candle_closed(symbol_id, bar)
                        except Exception as e:
                            self.logger.error(f"Error in on_candle_closed callback: {e}")
                    else:
                        # Duplicate event for the same candle (common with multiple spot updates)
                        pass
        
    def _start_heartbeat(self):
        """Schedules the first heartbeat."""
        self._stop_heartbeat()
        self.heartbeat_timer = reactor.callLater(self.heartbeat_interval, self._send_heartbeat)
        
    def _stop_heartbeat(self):
        """Cancels any scheduled heartbeat."""
        if self.heartbeat_timer and self.heartbeat_timer.active():
            self.heartbeat_timer.cancel()
        self.heartbeat_timer = None
        
    def _send_heartbeat(self):
        """Sends a heartbeat message and schedules the next one."""
        heartbeat = ProtoHeartbeatEvent()
        d = self.client.send(heartbeat)
        
        # Heartbeats don't have clientMsgId response matching, causing a TimeoutError
        # because the client waits for a response that never matches. We suppress this.
        def suppress_timeout(failure):
            if failure.check(defer.TimeoutError):
                return None # Checkmate, Timeout.
            return failure
            
        d.addErrback(suppress_timeout)
        d.addErrback(lambda f: self.logger.warning(f"Heartbeat failed: {f}"))

        self.heartbeat_timer = reactor.callLater(self.heartbeat_interval, self._send_heartbeat)

    def stop(self):
        """Stops the client service and heartbeat."""
        self._stop_heartbeat()
        self.client.stopService()

    @inlineCallbacks
    def send_request(self, request, retries=3, timeout=10):
        """Sends a request and returns a Deferred with retry logic."""
        from twisted.internet.task import deferLater
        last_error = None
        for i in range(retries):
            try:
                # Some versions of the library might support client.send(request, timeout=timeout)
                # If not, we rely on the internal timeout of the client or wrap it.
                # Here we assume the client.send returns a deferred.
                res = yield self.client.send(request)
                return res
            except Exception as e:
                last_error = e
                self.logger.warning(f"Request failed (attempt {i+1}/{retries}): {e}")
                if i < retries - 1:
                    # Exponential backoff: 1s, 2s, 4s...
                    yield deferLater(reactor, 1.0 * (2**i), lambda: None)
        
        raise last_error

    @inlineCallbacks
    def fetch_ohlcv(self, symbol_id, count=150, period=ProtoOATrendbarPeriod.M5):
        """Fetches historical trendbars for a symbol."""
        from datetime import datetime, timedelta
        
        req = ProtoOAGetTrendbarsReq()
        req.ctidTraderAccountId = self.account_id
        req.symbolId = symbol_id
        req.period = period
        
        to_timestamp = int(datetime.now().timestamp() * 1000)
        # Increase lookback range significantly to account for weekends/holidays (e.g. 5x the count)
        # 5 mins * count * 5
        from_timestamp = to_timestamp - (count * 5 * 5 * 60 * 1000)
        
        req.fromTimestamp = from_timestamp
        req.toTimestamp = to_timestamp
        
        res_msg = yield self.send_request(req)
        return Protobuf.extract(res_msg)

    @inlineCallbacks
    def fetch_account_summary(self):
        """Fetches account details (balance, leverage, etc)."""
        req = ProtoOATraderReq()
        req.ctidTraderAccountId = self.account_id
        res_msg = yield self.send_request(req)
        return Protobuf.extract(res_msg)

    @inlineCallbacks
    def fetch_open_positions(self):
        """Fetches currently open positions for the account."""
        req = ProtoOAReconcileReq()
        req.ctidTraderAccountId = self.account_id
        res_msg = yield self.send_request(req)
        return Protobuf.extract(res_msg)

    @inlineCallbacks
    def subscribe_spots(self, symbol_ids):
        """Subscribes to spot events (prerequisite for trendbar updates)."""
        req = ProtoOASubscribeSpotsReq()
        req.ctidTraderAccountId = self.account_id
        if isinstance(symbol_ids, list):
            req.symbolId.extend(symbol_ids)
        else:
            req.symbolId.append(symbol_ids)
            
        res_msg = yield self.send_request(req)
        return Protobuf.extract(res_msg)

    @inlineCallbacks
    def close_position(self, pos_id, volume):
        """Closes an existing position."""
        req = ProtoOAClosePositionReq()
        req.ctidTraderAccountId = self.account_id
        req.positionId = pos_id
        req.volume = int(volume)

        res_msg = yield self.send_request(req)
        return Protobuf.extract(res_msg)

    @inlineCallbacks
    def subscribe_trendbars(self, symbol_ids, period=ProtoOATrendbarPeriod.M5):
        """Subscribes to live trendbar updates for each symbol.
        
        IMPORTANT: Must call subscribe_spots() FIRST before calling this.
        This is required by cTrader API to receive trendbar close events.
        """
        if not isinstance(symbol_ids, list):
            symbol_ids = [symbol_ids]
        
        for symbol_id in symbol_ids:
            req = ProtoOASubscribeLiveTrendbarReq()
            req.ctidTraderAccountId = self.account_id
            req.symbolId = symbol_id
            req.period = period
            
            res_msg = yield self.send_request(req)
            self.logger.info(f"Subscribed to M5 trendbars for symbol {symbol_id}")
        
        return True

    @inlineCallbacks
    def subscribe(self, symbol_ids):
        """Full subscription: spots + M5 trendbars (convenience wrapper)."""
        # Step 1: Subscribe to spots (required first)
        yield self.subscribe_spots(symbol_ids)
        self.logger.info("Subscribed to spot events.")
        
        # Step 2: Subscribe to M5 trendbars (required to get candle close events)
        yield self.subscribe_trendbars(symbol_ids)
        self.logger.info("Subscribed to M5 trendbar events.")

    @inlineCallbacks
    def execute_market_order(self, symbol_id, volume, side, relative_sl=None, relative_tp=None):
        """Executes a Market Order."""
        req = ProtoOANewOrderReq()
        req.ctidTraderAccountId = self.account_id
        req.symbolId = symbol_id
        req.volume = volume
        req.tradeSide = side
        req.orderType = ProtoOAOrderType.MARKET
        
        if relative_sl is not None:
            req.relativeStopLoss = int(relative_sl)
        if relative_tp is not None:
            req.relativeTakeProfit = int(relative_tp)
            
        res_msg = yield self.send_request(req)
        return Protobuf.extract(res_msg)
