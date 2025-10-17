import os
import json
from datetime import datetime, timedelta
from twisted.internet import reactor
from services.cTrader.client import create_ctrader_client
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOATrendbarPeriod
from ctrader_open_api.messages.OpenApiMessages_pb2 import (
    ProtoOAGetTrendbarsReq, ProtoOAGetTrendbarsRes,
    ProtoOASymbolsListReq, ProtoOASymbolsListRes
)
from utils.logger import system_logger as log

class MarketDataService:
    def __init__(self, client):
        self._client = client
        self._client.add_message_listener(self.on_message)
        self.symbols_deferred = None

    def on_message(self, client, message):
        """Callback for received messages."""
        if message.payloadType == ProtoOAGetTrendbarsRes.payloadType:
            self._on_trendbars_response(message)
        elif message.payloadType == ProtoOASymbolsListRes.payloadType:
            self._on_symbols_list_response(message)

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
        self.symbols_deferred = reactor.deferred()
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
            self.symbols_deferred.callback(symbols_data)
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
        # For this example, we stop the reactor after fetching data.
        reactor.callLater(2, self._client.disconnect)


if __name__ == '__main__':
    log.info("Starting Market Data Service for testing...")

    client = create_ctrader_client()
    market_data_service = MarketDataService(client)

    def on_symbols_loaded(symbols):
        log.info(f"{len(symbols)} symbols loaded. Fetching trendbars for the first symbol.")
        if symbols:
            eurusd_symbol = next((s for s in symbols if s['symbolName'] == 'EURUSD'), None)
            symbol_id_to_fetch = eurusd_symbol['symbolId'] if eurusd_symbol else 1

            now = datetime.utcnow()
            thirty_days_ago = now - timedelta(days=30)
            market_data_service.get_trendbars(symbol_id_to_fetch, ProtoOATrendbarPeriod.D1, thirty_days_ago, now)
        else:
            log.warning("No symbols found, cannot fetch trendbars.")
            client.disconnect()

    def on_auth_success(client_instance):
        log.info("Authentication success. Fetching symbol list...")
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