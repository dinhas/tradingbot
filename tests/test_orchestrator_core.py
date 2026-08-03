import pytest
import time
from unittest.mock import MagicMock, patch
from LiveExecution.src.orchestrator import Orchestrator

def test_orchestrator_init(tmp_path):
    client = MagicMock()
    fm = MagicMock()
    fm.assets = ["EURUSD", "GBPUSD", "USDCHF", "USDJPY"]
    ml = MagicMock()
    db_file = tmp_path / "test_live_trading.db"
    config = {"DB_PATH": str(db_file)}

    orchestrator = Orchestrator(client, fm, ml, config=config)
    assert orchestrator.client == client
    assert orchestrator.fm == fm
    assert orchestrator.ml == ml
    assert orchestrator.config == config
    assert orchestrator.active_positions == {}
    assert orchestrator.entry_prices == {}
    assert isinstance(orchestrator.portfolio_state, dict)
    assert "EURUSD" in orchestrator.portfolio_state

def test_update_account_state(tmp_path):
    client = MagicMock()
    fm = MagicMock()
    fm.assets = ["EURUSD"]
    ml = MagicMock()
    db_file = tmp_path / "test_live_trading.db"

    orchestrator = Orchestrator(client, fm, ml, config={"DB_PATH": str(db_file)})

    account_res = MagicMock()
    account_res.trader.balance = 1000000 # $10,000.00

    orchestrator.update_account_state(account_res)
    assert orchestrator.portfolio_state['balance'] == 10000.0
    assert orchestrator.portfolio_state['equity'] == 10000.0
    assert orchestrator.portfolio_state['initial_equity'] == 10000.0
    assert orchestrator.portfolio_state['peak_equity'] == 10000.0

    # update again with slightly different equity, verify peak_equity holds max
    orchestrator.portfolio_state['equity'] = 9500.0
    orchestrator.update_account_state(account_res) # balance is still 10000.0
    assert orchestrator.portfolio_state['peak_equity'] == 10000.0

def test_is_asset_locked(tmp_path):
    client = MagicMock()
    fm = MagicMock()
    fm.assets = ["EURUSD"]
    ml = MagicMock()
    db_file = tmp_path / "test_live_trading.db"

    orchestrator = Orchestrator(client, fm, ml, config={"DB_PATH": str(db_file)})

    # 1 is EURUSD symbolId
    assert orchestrator.is_asset_locked(1) is False

    orchestrator.active_positions[1] = 12345
    assert orchestrator.is_asset_locked(1) is True

def test_get_symbol_name(tmp_path):
    client = MagicMock()
    client.symbol_ids = {'EURUSD': 1, 'GBPUSD': 2}
    fm = MagicMock()
    fm.assets = ["EURUSD", "GBPUSD"]
    ml = MagicMock()
    db_file = tmp_path / "test_live_trading.db"

    orchestrator = Orchestrator(client, fm, ml, config={"DB_PATH": str(db_file)})
    assert orchestrator._get_symbol_name(1) == "EURUSD"
    assert orchestrator._get_symbol_name(2) == "GBPUSD"
    assert orchestrator._get_symbol_name(999) == "Unknown"

def test_on_order_error(tmp_path, caplog):
    client = MagicMock()
    fm = MagicMock()
    fm.assets = ["EURUSD"]
    ml = MagicMock()
    db_file = tmp_path / "test_live_trading.db"

    orchestrator = Orchestrator(client, fm, ml, config={"DB_PATH": str(db_file)})

    mock_event = MagicMock()
    mock_event.errorCode = "ERROR_CODE"
    mock_event.description = "Test error"

    with caplog.at_level("ERROR"):
        orchestrator.on_order_error(mock_event)
        assert any("Order rejected: ERROR_CODE - Test error" in record.message for record in caplog.records)
