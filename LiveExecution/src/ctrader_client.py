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
            # Add more event handlers as needed
            
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            
    def _handle_spot_event(self, event):
        """Processes Spot Events for trendbar closes."""
        if event.trendbar:
            for bar in event.trendbar:
                if bar.period == ProtoOATrendbarPeriod.M5:
                    self.logger.info(f"M5 Candle closed for symbol {event.symbolId}")
                    if self.on_candle_closed:
                        self.on_candle_closed(event.symbolId)
        
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
        self.client.send(heartbeat)
        self.heartbeat_timer = reactor.callLater(self.heartbeat_interval, self._send_heartbeat)

    def stop(self):
        """Stops the client service and heartbeat."""
        self._stop_heartbeat()
        self.client.stopService()

    def send_request(self, request):
        """Sends a request and returns a Deferred."""
        return self.client.send(request)

    @inlineCallbacks
    def fetch_ohlcv(self, symbol_id, count=150, period=ProtoOATrendbarPeriod.M5):
        """Fetches historical trendbars for a symbol."""
        from datetime import datetime, timedelta
        
        req = ProtoOAGetTrendbarsReq()
        req.ctidTraderAccountId = self.account_id
        req.symbolId = symbol_id
        req.period = period
        
        to_timestamp = int(datetime.now().timestamp() * 1000)
        # Increase lookback range significantly to account for weekends/holidays (e.g. 2x the count)
        # 5 mins * count * 2
        from_timestamp = to_timestamp - (count * 2 * 5 * 60 * 1000)
        
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
    def execute_market_order(self, symbol_id, volume, side, sl_price=None, tp_price=None):
        """Executes a Market Order."""
        req = ProtoOANewOrderReq()
        req.ctidTraderAccountId = self.account_id
        req.symbolId = symbol_id
        req.volume = volume
        req.tradeSide = side
        req.orderType = ProtoOAOrderType.MARKET
        
        if sl_price:
            req.stopLoss = sl_price
        if tp_price:
            req.takeProfit = tp_price
            
        res_msg = yield self.send_request(req)
        return Protobuf.extract(res_msg)
