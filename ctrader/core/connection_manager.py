"""
Connection Manager for cTrader API

Handles authentication, connection lifecycle, and provides a clean interface
for establishing and maintaining connections to the cTrader API.
"""

from ctrader_open_api import Client, TcpProtocol, EndPoints
from ctrader_open_api.messages import OpenApiMessages_pb2 as api_msgs
from dotenv import load_dotenv
import os
from typing import Optional, Callable


class ConnectionManager:
    """Manages connection and authentication with cTrader API"""
    
    def __init__(self, use_demo: bool = True):
        """
        Initialize Connection Manager
        
        Args:
            use_demo (bool): Whether to use demo or live environment
        """
        self._load_environment()
        self.use_demo = use_demo
        self.host = EndPoints.PROTOBUF_DEMO_HOST if use_demo else EndPoints.PROTOBUF_LIVE_HOST
        self.client = Client(self.host, EndPoints.PROTOBUF_PORT, TcpProtocol)
        self.is_authenticated = False
        self.is_connected = False
        self.is_account_authenticated = False
        
        # Callbacks
        self._on_connected_callback: Optional[Callable] = None
        self._on_disconnected_callback: Optional[Callable] = None
        self._on_app_auth_callback: Optional[Callable] = None
        self._on_account_auth_callback: Optional[Callable] = None
        
    def _load_environment(self):
        """Load environment variables"""
        dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env')
        load_dotenv(dotenv_path=dotenv_path)
        
        self.client_id = os.getenv("CTRADER_APP_CLIENT_ID")
        self.client_secret = os.getenv("CTRADER_APP_CLIENT_SECRET")
        self.trader_account_id = os.getenv("CTRADER_ACCOUNT_ID")
        self.access_token = os.getenv("ACCESS_TOKEN")
        
        if not all([self.client_id, self.client_secret, self.trader_account_id, self.access_token]):
            raise ValueError("Missing required environment variables for cTrader API")
    
    def set_connected_callback(self, callback: Callable):
        """Set callback for successful connection"""
        self._on_connected_callback = callback
        
    def set_disconnected_callback(self, callback: Callable):
        """Set callback for disconnection"""
        self._on_disconnected_callback = callback
        
    def set_app_auth_callback(self, callback: Callable):
        """Set callback for application authentication"""
        self._on_app_auth_callback = callback
        
    def set_account_auth_callback(self, callback: Callable):
        """Set callback for account authentication"""
        self._on_account_auth_callback = callback
    
    def connect(self):
        """Establish connection to cTrader API"""
        try:
            self.client.setConnectedCallback(self._on_connected)
            self.client.setDisconnectedCallback(self._on_disconnected)
            self.client.startService()
            print(f"Connecting to cTrader API ({'Demo' if self.use_demo else 'Live'})...")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from cTrader API"""
        if self.client and hasattr(self.client, 'stopService'):
            self.client.stopService()
            self.is_connected = False
            self.is_authenticated = False
    
    @property
    def account_id(self):
        """Get the trader account ID"""
        return self.trader_account_id
    
    def authenticate_application(self):
        """Authenticate the application with cTrader API"""
        request = api_msgs.ProtoOAApplicationAuthReq()
        request.clientId = self.client_id
        request.clientSecret = self.client_secret
        
        deferred = self.client.send(request)
        deferred.addCallbacks(self._on_app_auth_success, self._on_error)
        
    def authenticate_account(self):
        """Authenticate the trading account"""
        request = api_msgs.ProtoOAAccountAuthReq()
        request.ctidTraderAccountId = int(self.trader_account_id)
        request.accessToken = self.access_token
        
        deferred = self.client.send(request)
        deferred.addCallbacks(self._on_account_auth_success, self._on_error)
    
    def _on_connected(self, client):
        """Internal callback for connection establishment"""
        self.is_connected = True
        print("Connected to cTrader API")
        
        if self._on_connected_callback:
            self._on_connected_callback(client)
            
        # Start authentication flow
        self.authenticate_application()
    
    def _on_disconnected(self, client, reason):
        """Internal callback for disconnection"""
        self.is_connected = False
        self.is_authenticated = False
        self.is_account_authenticated = False
        print(f"Disconnected from cTrader API: {reason}")
        
        if self._on_disconnected_callback:
            self._on_disconnected_callback(client, reason)
    
    def _on_app_auth_success(self, message):
        """Internal callback for successful app authentication"""
        self.is_authenticated = True
        print("Application authenticated successfully")
        
        if self._on_app_auth_callback:
            self._on_app_auth_callback(self.client, message)
            
        # Proceed to account authentication
        self.authenticate_account()
    
    def _on_account_auth_success(self, message):
        """Internal callback for successful account authentication"""
        self.is_account_authenticated = True
        print(f"Account {self.account_id} authenticated successfully")
        
        if self._on_account_auth_callback:
            self._on_account_auth_callback(self.client, message)
    
    def _on_error(self, failure):
        """Internal callback for errors"""
        print(f"Authentication error: {failure}")
    
    @property
    def is_ready(self) -> bool:
        """Check if connection is ready for trading operations"""
        return self.is_connected and self.is_authenticated and self.is_account_authenticated