"""
Message Router for cTrader API

Routes incoming messages from cTrader API to appropriate service handlers
based on message type and registration.
"""

from typing import Dict, Callable, Optional
from ctrader_open_api import Protobuf


class MessageRouter:
    """Routes cTrader API messages to registered service handlers"""
    
    def __init__(self):
        self._handlers: Dict[int, Callable] = {}
        self._default_handler: Optional[Callable] = None
    
    def register_handler(self, message_type: int, handler: Callable):
        """
        Register a handler for a specific message type
        
        Args:
            message_type (int): cTrader API message type ID
            handler (Callable): Function to handle the message
        """
        self._handlers[message_type] = handler
        print(f"Registered handler for message type {message_type}")
    
    def unregister_handler(self, message_type: int):
        """
        Unregister a handler for a message type
        
        Args:
            message_type (int): cTrader API message type ID
        """
        if message_type in self._handlers:
            del self._handlers[message_type]
            print(f"🗑️ Unregistered handler for message type {message_type}")
    
    def set_default_handler(self, handler: Callable):
        """
        Set a default handler for unregistered message types
        
        Args:
            handler (Callable): Default handler function
        """
        self._default_handler = handler
    
    def route_message(self, client, message):
        """
        Route incoming message to appropriate handler
        
        Args:
            client: cTrader client instance
            message: Incoming message from cTrader API
        """
        message_type = message.payloadType
        
        if message_type in self._handlers:
            try:
                self._handlers[message_type](client, message)
            except Exception as e:
                print(f"❌ Error handling message type {message_type}: {e}")
        elif self._default_handler:
            try:
                self._default_handler(client, message)
            except Exception as e:
                print(f"❌ Error in default handler for message type {message_type}: {e}")
        else:
            print(f"⚠️ No handler registered for message type {message_type}")
            print("Message content:", Protobuf.extract(message))
    
    def get_registered_types(self) -> list:
        """Get list of registered message types"""
        return list(self._handlers.keys())
    
    def clear_handlers(self):
        """Clear all registered handlers"""
        self._handlers.clear()
        self._default_handler = None
        print("🧹 Cleared all message handlers")