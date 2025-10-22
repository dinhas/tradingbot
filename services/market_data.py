import os
import json
from datetime import datetime, timedelta
from twisted.internet import reactor
from twisted.internet.defer import Deferred
from services.cTrader.client import create_ctrader_client
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOATrendbarPeriod, ProtoOAPayloadType
from ctrader_open_api.messages.OpenApiMessages_pb2 import (
    ProtoOAGetTrendbarsReq, ProtoOAGetTrendbarsRes,
    ProtoOASymbolsListReq, ProtoOASymbolsListRes
)
from ctrader_open_api.messages.OpenApiMessages_pb2 import (
    ProtoOASubscribeSpotsReq, ProtoOASubscribeSpotsRes, ProtoOASpotEvent
)
from utils.logger import system_logger as log
from utils.indicators import calculate_scalping_indicators
from collections import deque

class MarketDataService:
    def __init__(self, client):
        self._client = client
        self._client.add_message_listener(self.on_message)
        self.symbols_deferred = None
        self.spot_subscriptions = {}
        self.live_candles = {}
        # Store the last 100 candles for indicator calculation
        self.historical_candles = {}

    def on_message(self, client, message):
        """Callback for received messages."""
        if message.payloadType == ProtoOAPayloadType.PROTO_OA_GET_TRENDBARS_RES:
            self._on_trendbars_response(message)
        elif message.payloadType == ProtoOAPayloadType.PROTO_OA_SYMBOLS_LIST_RES:
            self._on_symbols_list_response(message)
        elif message.payloadType == ProtoOAPayloadType.PROTO_OA_SUBSCRIBE_SPOTS_RES:
            self._on_subscribe_spots_response(message)
        elif message.payloadType == ProtoOAPayloadType.PROTO_OA_SPOT_EVENT:
            self._on_spot_event(message)

    def get_trendbars(self, symbol_id, period, start_date, end_date):
        log.info(f"Requesting trendbars for symbol {symbol_id} from {start_date} to {end_date}.")
        request = ProtoOAGetTrendbarsReq()
        request.ctidTraderAccountId = self._client.account_id
        request.symbolId = symbol_id
        request.period = period
        request.fromTimestamp = int(start_date.timestamp() * 1000)
        request.toTimestamp = int(end_date.timestamp() * 1000)

        self._client.send(request)

    def get_symbols(self):
        log.info("Requesting all symbols...")
        self.symbols_deferred = Deferred()
        request = ProtoOASymbolsListReq()
        request.ctidTraderAccountId = self._client.account_id
        self._client.send(request)
        return self.symbols_deferred

    def _on_symbols_list_response(self, message):
        response = ProtoOASymbolsListRes()
        response.ParseFromString(message.payload)
        log.info(f"Received {len(response.symbol)} symbols.")

        symbols_data = [
            {
                "symbolId": symbol.symbolId,
                "symbolName": symbol.symbolName,
            }
            for symbol in response.symbol
        ]

        # Save symbols to a file
        try:
            logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            file_path = os.path.join(logs_dir, "symbols.json")
            with open(file_path, "w") as f:
                json.dump(symbols_data, f, indent=4)
            log.info(f"Symbols data saved to {file_path}")
        except Exception as e:
            log.error(f"Failed to save symbols data: {e}")

        if self.symbols_deferred:
            self.symbols_deferred.callback(file_path)
            self.symbols_deferred = None

    def _on_trendbars_response(self, message):
        response = ProtoOAGetTrendbarsRes()
        response.ParseFromString(message.payload)

        log.info(f"Received {len(response.trendbar)} trendbars.")

        trendbars = [{
            "timestamp": bar.utcTimestampInMinutes * 60 * 1000,
            "open": bar.low + bar.deltaOpen,
            "high": bar.low + bar.deltaHigh,
            "low": bar.low,
            "close": bar.low + bar.deltaClose,
            "volume": bar.volume
        } for bar in response.trendbar]

        # For now, we'll just log the data. Later, this will be processed.
        log.info(f"Sample trendbar data: {trendbars[0] if trendbars else 'No data'}")

        # In a real scenario, you would process or store this data.
        reactor.callLater(2, self._client.disconnect)

    def subscribe_to_spots(self, symbol_id):
        log.info(f"Subscribing to spot data for symbol {symbol_id}.")
        request = ProtoOASubscribeSpotsReq()
        request.ctidTraderAccountId = self._client.account_id
        request.symbolId.append(symbol_id)
        self._client.send(request)

    def _on_subscribe_spots_response(self, message):
        response = ProtoOASubscribeSpotsRes()
        response.ParseFromString(message.payload)
        log.info(f"Successfully subscribed to spot data.")

    def _on_spot_event(self, message):
        event = ProtoOASpotEvent()
        event.ParseFromString(message.payload)
        now = datetime.utcnow()

        price = (event.bid + event.ask) // 2 if event.ask and event.bid else None
        if price is None:
            return

        current_minute = (now.minute // 5) * 5
        candle_timestamp = now.replace(minute=current_minute, second=0, microsecond=0)

        self._initialize_historical_candles(event.symbolId)
        candle = self.live_candles.get(event.symbolId)

        if not candle or candle['timestamp'] != candle_timestamp:
            if candle:
                self.historical_candles[event.symbolId].append(candle)
                indicators = calculate_scalping_indicators(
                    list(self.historical_candles[event.symbolId])
                )
                if indicators:
                    candle['indicators'] = indicators
                    log.info(f"New 5-min candle for symbol {event.symbolId} with indicators: {candle}")
                else:
                    log.info(f"New 5-min candle for symbol {event.symbolId}. Not enough data for indicators.")

            new_candle = {
                "timestamp": candle_timestamp, "open": price, "high": price, "low": price, "close": price, "volume": 0
            }
            self.live_candles[event.symbolId] = new_candle
        else:
            candle['high'] = max(candle['high'], price)
            candle['low'] = min(candle['low'], price)
            candle['close'] = price
            candle['volume'] = candle.get('volume', 0) + 1

        log.debug(f"Spot Event: Symbol={event.symbolId}, Price={price}, Candle={self.live_candles[event.symbolId]}")

    def _initialize_historical_candles(self, symbol_id):
        if symbol_id not in self.historical_candles:
            self.historical_candles[symbol_id] = deque(maxlen=100)
            log.info(f"Initialized historical candle buffer for symbol {symbol_id}.")


if __name__ == '__main__':
    log.info("Starting Market Data Service for testing...")

    client = create_ctrader_client()
    market_data_service = MarketDataService(client)

    def on_symbols_loaded(symbols_path):
        log.info(f"Symbols file created at {symbols_path}. Subscribing to EURUSD spot data.")
        try:
            with open(symbols_path, 'r') as f:
                symbols = json.load(f)
            eurusd_symbol = next((s for s in symbols if s['symbolName'] == 'EURUSD'), None)
            if eurusd_symbol:
                symbol_id_to_fetch = eurusd_symbol['symbolId']
                market_data_service.subscribe_to_spots(symbol_id_to_fetch)
                reactor.callLater(310, client.disconnect)
            else:
                log.warning("EURUSD symbol not found. Cannot subscribe to spot data.")
                client.disconnect()
        except Exception as e:
            log.error(f"Failed to read symbols file: {e}")
            client.disconnect()

    def on_auth_success(client_instance):
        log.info("Authentication success. Fetching symbol list to get IDs...")
        deferred = market_data_service.get_symbols()
        deferred.addCallbacks(on_symbols_loaded, on_auth_error)

    def on_auth_error(failure):
        log.error(f"Operation failed: {failure}")
        if reactor.running:
            reactor.stop()

    deferred = client.connect()
    deferred.addCallbacks(on_auth_success, on_auth_error)

    if not reactor.running:
        reactor.run()

    log.info("Reactor stopped.")