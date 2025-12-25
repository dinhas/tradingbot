import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# We can't easily test main.py execution without running it, 
# but we can import a 'run_system' function from it if we structure it right.

class TestMainEntry(unittest.TestCase):
    @patch('LiveExecution.main.CTraderClient')
    @patch('LiveExecution.main.FeatureManager')
    @patch('LiveExecution.main.ModelLoader')
    @patch('LiveExecution.main.DiscordNotifier')
    @patch('LiveExecution.main.Orchestrator')
    @patch('LiveExecution.main.reactor')
    def test_main_initialization(self, mock_reactor, mock_orch, mock_notifier, mock_ml, mock_fm, mock_client):
        from LiveExecution.main import main
        
        # Mock environment variables
        with patch.dict('os.environ', {
            'CT_APP_ID': 'test',
            'CT_APP_SECRET': 'test',
            'CT_ACCOUNT_ID': '123',
            'CT_ACCESS_TOKEN': 'test',
            'CT_HOST_TYPE': 'demo',
            'DISCORD_WEBHOOK_URL': 'test'
        }):
            main()
            
            # Verify initializations
            mock_client.assert_called()
            mock_fm.assert_called()
            mock_ml.assert_called()
            mock_notifier.assert_called()
            mock_orch.assert_called()
            
            # Verify wiring
            client_instance = mock_client.return_value
            orch_instance = mock_orch.return_value
            self.assertEqual(client_instance.on_candle_closed, orch_instance.on_m5_candle_close)
            
            # Verify start
            client_instance.start.assert_called_once()
            mock_reactor.run.assert_called_once()

if __name__ == '__main__':
    unittest.main()
