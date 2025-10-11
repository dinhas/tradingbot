






"""
Market Data Module - Legacy compatibility wrapper

This module provides backward compatibility while using the new symbols_manager module.
"""

from ctrader_open_api.messages import OpenApiMessages_pb2 as api_msgs
from ctrader.symbols_manager import get_symbols_manager

# Get the module-level symbols manager
symbols_manager = get_symbols_manager()

def onProtoOASymbolsListRes(client, message: api_msgs.ProtoOASymbolsListRes):
    """
    Legacy callback function - now delegates to symbols_manager
    """
    symbols_manager.on_symbols_received(client, message)

def get_symbols(client, ctidTraderAccountId, onError):
    """
    Legacy function to request symbols - now uses symbols_manager
    """
    request = api_msgs.ProtoOASymbolsListReq()
    request.ctidTraderAccountId = int(ctidTraderAccountId)
    deferred = client.send(request)
    deferred.addErrback(onError)