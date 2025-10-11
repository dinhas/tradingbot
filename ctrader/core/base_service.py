"""
Base Service Class for cTrader API Services

Provides common functionality and interface that all cTrader services inherit from.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import json
import os
from datetime import datetime


class BaseService(ABC):
    """Base class for all cTrader services"""
    
    def __init__(self, connection_manager, message_router, service_name: str):
        """
        Initialize base service
        
        Args:
            connection_manager: Connection manager instance
            message_router: Message router instance
            service_name (str): Name of the service
        """
        self.connection_manager = connection_manager
        self.message_router = message_router
        self.service_name = service_name
        self.is_initialized = False
        self._data_cache: Dict[str, Any] = {}
        
        # Setup data directory
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = os.path.join(self.project_root, 'data', self.service_name)
        os.makedirs(self.data_dir, exist_ok=True)
        
    @abstractmethod
    def initialize(self):
        """Initialize the service - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def get_message_handlers(self) -> Dict[int, callable]:
        """Return dict of message types and their handlers"""
        pass
    
    def register_message_handlers(self):
        """Register this service's message handlers with the message router"""
        handlers = self.get_message_handlers()
        for message_type, handler in handlers.items():
            self.message_router.register_handler(message_type, handler)
    
    def unregister_message_handlers(self):
        """Unregister this service's message handlers"""
        handlers = self.get_message_handlers()
        for message_type in handlers.keys():
            self.message_router.unregister_handler(message_type)
    
    def save_data(self, filename: str, data: Dict[str, Any]):
        """
        Save data to JSON file in service data directory
        
        Args:
            filename (str): Name of the file (without .json extension)
            data (Dict): Data to save
        """
        try:
            filepath = os.path.join(self.data_dir, f"{filename}.json")
            
            # Add metadata
            data_with_meta = {
                "service": self.service_name,
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data_with_meta, f, indent=4, ensure_ascii=False)
                
            print(f"{self.service_name}: Saved data to {filename}.json")
            
        except Exception as e:
            print(f"{self.service_name}: Error saving data to {filename}.json: {e}")
    
    def load_data(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Load data from JSON file
        
        Args:
            filename (str): Name of the file (without .json extension)
            
        Returns:
            Dict or None: Loaded data or None if file doesn't exist
        """
        try:
            filepath = os.path.join(self.data_dir, f"{filename}.json")
            
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                    return loaded_data.get('data', loaded_data)  # Handle both new and old formats
            else:
                print(f"{self.service_name}: No {filename}.json file found")
                return None
                
        except Exception as e:
            print(f"{self.service_name}: Error loading data from {filename}.json: {e}")
            return None
    
    def cache_data(self, key: str, data: Any):
        """Cache data in memory"""
        self._data_cache[key] = data
    
    def get_cached_data(self, key: str) -> Any:
        """Get cached data"""
        return self._data_cache.get(key)
    
    def clear_cache(self):
        """Clear all cached data"""
        self._data_cache.clear()
    
    def log(self, message: str, level: str = "INFO"):
        """
        Log a message with service context
        
        Args:
            message (str): Message to log
            level (str): Log level (INFO, WARN, ERROR)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level} - {self.service_name}: {message}")
    
    @property
    def client(self):
        """Get the cTrader client from connection manager"""
        return self.connection_manager.client
    
    @property
    def account_id(self) -> str:
        """Get the account ID from connection manager"""
        return self.connection_manager.account_id
    
    @property
    def is_ready(self) -> bool:
        """Check if service is ready (connection established and service initialized)"""
        return self.connection_manager.is_ready and self.is_initialized
    
    def on_error(self, failure):
        """Default error handler for service operations"""
        self.log(f"Operation failed: {failure}", "ERROR")