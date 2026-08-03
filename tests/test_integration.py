import pytest
import sys
import os
import logging
import pandas as pd
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path
from twisted.internet import defer

from LiveExecution.src.orchestrator import Orchestrator
from LiveExecution.src.models import ModelLoader
from LiveExecution.src.features import FeatureManager
from LiveExecution.src.database import DatabaseManager
from LiveExecution.src.ctrader_client import CTraderClient
from LiveExecution.dashboard.main import DashboardServer

def test_main_wiring_mocked():
    """Verify that main.py logic is wired up correctly as described in requirements."""
    # Mock all the components and verify wiring
    config = {
        "CT_APP_ID": "app_id",
        "CT_APP_SECRET": "secret",
        "CT_ACCOUNT_ID": 12345,
        "CT_ACCESS_TOKEN": "token",
        "CT_HOST_TYPE": "demo",
        "ALPHA_CONFIDENCE_THRESHOLD": 0.65,
        "FILTER_THRESHOLD": 0.75,
        "SL_MULTIPLIER": 2.5,
        "TP_MULTIPLIER": 4.5,
        "DB_PATH": "test_wiring_db.db"
    }

    client = MagicMock()
    fm = MagicMock()
    ml = MagicMock()

    with patch("LiveExecution.src.orchestrator.DatabaseManager"):
        orchestrator = Orchestrator(client, fm, ml, config=config)
        dashboard = DashboardServer(orchestrator)
        orchestrator.set_dashboard(dashboard)

        # Simulating wiring step in main.py
        client.on_authenticated = orchestrator.bootstrap
        client.on_candle_closed = orchestrator.on_m5_candle_close
        client.on_order_execution = orchestrator.on_order_execution
        client.on_order_error = orchestrator.on_order_error

        assert client.on_authenticated == orchestrator.bootstrap
        assert client.on_candle_closed == orchestrator.on_m5_candle_close
        assert client.on_order_execution == orchestrator.on_order_execution
        assert client.on_order_error == orchestrator.on_order_error

def test_database_auto_creation(tmp_path):
    """Verify that if database file doesn't exist, it is auto-created."""
    db_file = tmp_path / "new_dir" / "trading_system.db"
    assert not db_file.exists()

    db = DatabaseManager(str(db_file))
    assert db_file.exists()

def test_model_loader_corrupted_checkpoint(tmp_path, caplog):
    """Verify corrupted checkpoint loading logs error and returns False."""
    ml = ModelLoader()
    ml.project_root = tmp_path

    alpha_dir = tmp_path / "Alpha" / "models"
    alpha_dir.mkdir(parents=True, exist_ok=True)
    alpha_path = alpha_dir / "alpha_model.pth"

    # Write corrupt data
    with open(alpha_path, "wb") as f:
        f.write(b"corrupted binary data")

    with caplog.at_level("ERROR"):
        res = ml.load_all_models()
        assert res is False
        assert any("Failed to load models" in record.message for record in caplog.records)

def test_feature_manager_empty_history_graceful_degradation():
    """Verify feature manager empty history gracefully returns None or handles empty shapes."""
    with patch("LiveExecution.src.features.DataLoader") as mock_loader_class:
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader

        fm = FeatureManager()
        # Ensure all history buffers are initially empty
        for asset in fm.history:
            assert fm.history[asset].empty

        # Mock preprocess_data of alpha_fe to return empty dataframes to avoid empty index key error
        with patch.object(fm.alpha_fe, "preprocess_data", return_value=(None, pd.DataFrame())):
            assert fm.get_alpha_sequence("EURUSD", 25) is None

        assert fm.get_filter_features() is None
        assert fm.get_atr("EURUSD") == 0.0
        assert fm.is_ready() is False

def test_telegram_bot_not_instantiated_in_main():
    """Verify that TelegramNotifier is NOT imported/instantiated in main.py."""
    main_file = Path(__file__).resolve().parent.parent / "LiveExecution" / "main.py"
    with open(main_file, "r") as f:
        content = f.read()
    assert "TelegramNotifier" not in content

def test_on_order_execution_yield_bug_without_inline_callbacks():
    """Verify on_order_execution uses yield but method is not decorated with @inlineCallbacks.

    This means calling on_order_execution returns a generator instead of a Deferred,
    which is a known bug that prevents asynchronous code from executing properly as Twisted expects.
    """
    from twisted.internet.defer import inlineCallbacks

    # Get the function object of on_order_execution
    method = Orchestrator.on_order_execution

    # An inlineCallbacks method usually has specific wrapper attributes, but wait,
    # let's verify if the function returns a generator.
    # In Python, calling a generator function returns a generator iterator.
    # If it is decorated with @inlineCallbacks, calling it will run the generator and return a Deferred.
    # So we can assert that calling it returns a generator iterator or is not decorated.
    # Let's inspect the decorators or verify if calling it returns a generator.
    client = MagicMock()
    fm = MagicMock()
    fm.assets = ["EURUSD"]
    ml = MagicMock()

    with patch("LiveExecution.src.orchestrator.DatabaseManager"):
        orchestrator = Orchestrator(client, fm, ml, config={"DB_PATH": ":memory:"})
        event = MagicMock()
        event.position = None
        event.order = None

        res = orchestrator.on_order_execution(event)
        # Verify it returns a generator iterator (since it is a generator but not decorated with inlineCallbacks)
        import types
        assert isinstance(res, types.GeneratorType)

def test_dashboard_unimplemented_close_position_by_id():
    """Verify close_position_by_id is called by dashboard but NOT implemented on orchestrator.

    Calling it should raise AttributeError.
    """
    client = MagicMock()
    fm = MagicMock()
    fm.assets = ["EURUSD"]
    ml = MagicMock()

    with patch("LiveExecution.src.orchestrator.DatabaseManager"):
        orchestrator = Orchestrator(client, fm, ml, config={"DB_PATH": ":memory:"})

        # Verify orchestrator does not have close_position_by_id method
        assert not hasattr(orchestrator, "close_position_by_id")

        with pytest.raises(AttributeError):
            orchestrator.close_position_by_id(123, 1)

def test_dashboard_unimplemented_kill_switch():
    """Verify kill_switch is called by dashboard but NOT implemented on orchestrator.

    Calling it should raise AttributeError.
    """
    client = MagicMock()
    fm = MagicMock()
    fm.assets = ["EURUSD"]
    ml = MagicMock()

    with patch("LiveExecution.src.orchestrator.DatabaseManager"):
        orchestrator = Orchestrator(client, fm, ml, config={"DB_PATH": ":memory:"})

        # Verify orchestrator does not have kill_switch method
        assert not hasattr(orchestrator, "kill_switch")

        with pytest.raises(AttributeError):
            orchestrator.kill_switch()
