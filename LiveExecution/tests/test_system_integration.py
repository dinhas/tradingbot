import unittest
from unittest.mock import MagicMock, patch, call, ANY
import sys
from pathlib import Path
from twisted.internet.defer import Deferred, inlineCallbacks, succeed

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from LiveExecution.src.orchestrator import Orchestrator
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOATradeSide
import numpy as np

class TestSystemIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_fm = MagicMock()
        self.mock_ml = MagicMock()
        self.mock_notifier = MagicMock()
        
        # Mock fetch_open_positions to return a successful empty response
        mock_pos_res = MagicMock()
        mock_pos_res.position = []
        self.mock_client.fetch_open_positions.return_value = succeed(mock_pos_res)
        
        # Mock ATR - crucial to be a float to avoid comparison errors with mocks
        self.mock_fm.get_atr.return_value = 0.001
        
        # Initialize Orchestrator with mocked dependencies
        self.orchestrator = Orchestrator(
            self.mock_client, 
            self.mock_fm, 
            self.mock_ml,
            self.mock_notifier
        )
        
        # Mock asset universe in FeatureManager
        self.mock_fm.assets = ['EURUSD', 'GBPUSD', 'XAUUSD', 'USDCHF', 'USDJPY']
        
        # Mock client symbol IDs
        self.mock_client.symbol_ids = {
            'EURUSD': 1, 'GBPUSD': 2, 'XAUUSD': 41, 'USDCHF': 6, 'USDJPY': 4
        }

    def test_on_m5_candle_close_success_flow(self):
        """
        Test the full flow:
        Candle Close -> Fetch Data -> Inference (Allow) -> Execute Order -> Notify
        """
        symbol_id = 1 # EURUSD
        
        # 1. Setup Data Fetching Mocks
        d_ohlcv = Deferred()
        self.mock_client.fetch_ohlcv.return_value = d_ohlcv
        
        d_account = Deferred()
        self.mock_client.fetch_account_summary.return_value = d_account
        
        # Mock Account Response Object
        mock_acc_res = MagicMock()
        mock_acc_res.trader.balance = 100000
        
        # Alpha returns 1.0 (Buy)
        self.mock_fm.get_alpha_observation.return_value = "alpha_obs_data"
        self.mock_ml.get_alpha_action.return_value = [1.0, 0, 0, 0, 0] # Action for 1st asset
        
        # Risk returns size=0.1, sl=0.01, tp=0.02
        self.mock_fm.get_risk_observation.return_value = "risk_obs_data"
        self.mock_ml.get_risk_action.return_value = np.array([0.1, 0.01, 0.02])
        
        # TradeGuard returns 1 (Allow)
        # Mocking history to get current price for TG observation setup
        import pandas as pd
        self.mock_fm.history = {'EURUSD': pd.DataFrame({'close': [1.1000]})}
        self.mock_fm.get_tradeguard_observation.return_value = "tg_obs_data"
        self.mock_ml.get_tradeguard_action.return_value = 1 # Allow
        
        # 3. Setup Execution Mock
        d_exec = Deferred()
        self.mock_client.execute_market_order.return_value = d_exec
        
        # --- Trigger the Event ---
        # The orchestrator should return a Deferred that resolves when the flow is done
        d_flow = self.orchestrator.on_m5_candle_close(symbol_id)
        
        # Resolve Data Fetches
        d_ohlcv.callback("ohlcv_data")
        d_account.callback(mock_acc_res)
        
        # Resolve Execution
        d_exec.callback("order_result")
        
        # --- Verifications ---
        
        # Verify Data Fetching
        self.mock_client.fetch_ohlcv.assert_called_with(symbol_id)
        self.mock_client.fetch_account_summary.assert_called()
        self.mock_fm.update_data.assert_called_with(1, "ohlcv_data")
        
        # Verify Inference Chain
        self.mock_ml.get_alpha_action.assert_called()
        self.mock_ml.get_risk_action.assert_called()
        self.mock_ml.get_tradeguard_action.assert_called()
        
        # Verify Execution
        # Expected: volume calculated (let's verify logic in implementation), side=BUY
        self.mock_client.execute_market_order.assert_called_once()
        args, kwargs = self.mock_client.execute_market_order.call_args
        self.assertEqual(args[0], symbol_id) # Symbol
        self.assertEqual(args[2], ProtoOATradeSide.BUY) # Side
        
        # Verify Notification
        self.mock_notifier.send_trade_event.assert_called_once()

    def test_on_m5_candle_close_blocked_trade(self):
        """Test flow when TradeGuard blocks the trade."""
        symbol_id = 1
        
        mock_acc_res = MagicMock()
        mock_acc_res.trader.balance = 100000
        
        # Setup mocks similar to above
        self.mock_client.fetch_ohlcv.return_value = succeed("ohlcv_data")
        self.mock_client.fetch_account_summary.return_value = succeed(mock_acc_res)
        
        # Alpha Buy
        self.mock_fm.get_alpha_observation.return_value = "alpha_obs_data"
        self.mock_ml.get_alpha_action.return_value = [1.0, 0, 0, 0, 0] 
        
        # Risk
        self.mock_fm.get_risk_observation.return_value = "risk_obs_data"
        self.mock_ml.get_risk_action.return_value = np.array([0.1, 0.01, 0.02])
        
        # Mock History
        import pandas as pd
        self.mock_fm.history = {'EURUSD': pd.DataFrame({'close': [1.1000]})}
        
        # TradeGuard BLOCKS (0)
        self.mock_fm.get_tradeguard_observation.return_value = "tg_obs_data"
        self.mock_ml.get_tradeguard_action.return_value = 0 # Block
        
        # Trigger
        self.orchestrator.on_m5_candle_close(symbol_id)
        
        # Verify NO Execution
        self.mock_client.execute_market_order.assert_not_called()
        
        # Verify Block Notification
        self.mock_notifier.send_block_event.assert_called_once()

    def test_asset_locked_check(self):
        """Test that locked assets skip inference."""
        symbol_id = 1
        
        # Mock fetch_open_positions to return a list containing this symbol
        mock_pos = MagicMock()
        mock_pos.symbolId = 1
        mock_pos.positionId = 123
        
        mock_res = MagicMock()
        mock_res.position = [mock_pos]
        self.mock_client.fetch_open_positions.return_value = succeed(mock_res)
        
        self.orchestrator.on_m5_candle_close(symbol_id)
        
        # Should NOT fetch data or run inference beyond syncing positions
        self.mock_client.fetch_ohlcv.assert_not_called()
        self.mock_ml.get_alpha_action.assert_not_called()

if __name__ == '__main__':
    unittest.main()
