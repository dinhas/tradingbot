import pytest
import math
from unittest.mock import MagicMock, patch
from LiveExecution.src.ctrader_client import CTraderClient, CTraderAmendError

def test_ctrader_client_init():
    config = {
        "CT_APP_ID": "app_id",
        "CT_APP_SECRET": "secret",
        "CT_ACCOUNT_ID": 12345,
        "CT_ACCESS_TOKEN": "token",
        "CT_HOST_TYPE": "demo"
    }
    client = CTraderClient(config)
    assert client.app_id == "app_id"
    assert client.app_secret == "secret"
    assert client.account_id == 12345
    assert client.access_token == "token"
    # EURUSD: 1, GBPUSD: 2, USDCHF: 6, USDJPY: 4
    assert client.symbol_ids == {'EURUSD': 1, 'GBPUSD': 2, 'USDCHF': 6, 'USDJPY': 4}
    assert client.heartbeat_interval == 25.0
    assert client.max_retries == 5
    assert client.base_delay == 5.0

def test_ctrader_client_reconnection_backoff(caplog):
    config = {
        "CT_APP_ID": "app_id", "CT_APP_SECRET": "secret",
        "CT_ACCOUNT_ID": 12345, "CT_ACCESS_TOKEN": "token",
        "CT_HOST_TYPE": "demo"
    }
    client = CTraderClient(config)

    # Test the custom retry policy with exponential backoff directly:
    retry_policy = client.custom_retry_policy
    assert retry_policy is not None

    # For failed attempts 0, 1, 2, 3:
    for attempt in range(4):
        delay = retry_policy(attempt)
        # Delay should be around min(5.0 * 2^attempt, 60.0) with some jitter +/- 1.0s
        expected_base = min(5.0 * (2 ** attempt), 60.0)
        assert expected_base - 1.1 <= delay <= expected_base + 1.1

def test_ctrader_client_last_bar_timestamps_deduplication():
    config = {
        "CT_APP_ID": "app_id", "CT_APP_SECRET": "secret",
        "CT_ACCOUNT_ID": 12345, "CT_ACCESS_TOKEN": "token",
        "CT_HOST_TYPE": "demo"
    }
    client = CTraderClient(config)

    mock_callback = MagicMock()
    client.on_candle_closed = mock_callback

    # Mock event
    from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOATrendbarPeriod
    mock_bar = MagicMock()
    mock_bar.period = ProtoOATrendbarPeriod.M5
    mock_bar.utcTimestampInMinutes = 1000

    mock_event = MagicMock()
    mock_event.trendbar = [mock_bar]
    mock_event.symbolId = 1 # EURUSD

    # 1. First event -> triggers callback
    client._handle_spot_event(mock_event)
    mock_callback.assert_called_once_with(1, mock_bar)
    assert client.last_bar_timestamps[1] == 1000

    # 2. Duplicate event same timestamp -> does NOT trigger callback
    mock_callback.reset_mock()
    client._handle_spot_event(mock_event)
    mock_callback.assert_not_called()

    # 3. New event higher timestamp -> triggers callback
    mock_bar_new = MagicMock()
    mock_bar_new.period = ProtoOATrendbarPeriod.M5
    mock_bar_new.utcTimestampInMinutes = 1005
    mock_event_new = MagicMock()
    mock_event_new.trendbar = [mock_bar_new]
    mock_event_new.symbolId = 1

    client._handle_spot_event(mock_event_new)
    mock_callback.assert_called_once_with(1, mock_bar_new)
    assert client.last_bar_timestamps[1] == 1005

def test_ctrader_amend_error_exception():
    err = CTraderAmendError(123, "ERR_CODE", "Invalid SL")
    assert err.pos_id == 123
    assert err.error_code == "ERR_CODE"
    assert err.description == "Invalid SL"
    assert str(err) == "Amend rejected for position 123: [ERR_CODE] Invalid SL"
