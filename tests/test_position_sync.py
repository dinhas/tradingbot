import pytest
from unittest.mock import MagicMock, patch
from twisted.internet import defer
from LiveExecution.src.orchestrator import Orchestrator

@pytest.mark.asyncio
async def test_sync_active_positions(tmp_path):
    client = MagicMock()
    fm = MagicMock()
    fm.assets = ["EURUSD", "GBPUSD"]
    ml = MagicMock()

    db_file = tmp_path / "test_live_trading.db"
    orchestrator = Orchestrator(client, fm, ml, config={"DB_PATH": str(db_file)})

    # Setup mock reconcile response with one open position
    pos = MagicMock()
    pos.positionId = 123
    pos.price = 1.0500
    pos.tradeData.symbolId = 1
    pos.tradeData.symbolId = 1
    pos.tradeData.volume = 10000000
    # stopLoss and takeProfit set to verify they are already present (meaning no attach_missing_sltp is needed)
    pos.stopLoss = 1.0450
    pos.takeProfit = 1.0550

    reconcile_res = MagicMock()
    reconcile_res.position = [pos]

    client.fetch_open_positions.return_value = defer.succeed(reconcile_res)

    with patch.object(Orchestrator, "_attach_missing_sltp_for_pos") as mock_attach:
        mock_attach.return_value = defer.succeed(None)

        await orchestrator.sync_active_positions()

        # Check active positions and entry prices synced
        assert orchestrator.active_positions == {1: 123}
        assert orchestrator.entry_prices == {123: 1.0500}
        assert orchestrator.portfolio_state['num_open_positions'] == 1
        mock_attach.assert_not_called()

@pytest.mark.asyncio
async def test_sync_active_positions_missing_sltp(tmp_path):
    client = MagicMock()
    client.symbol_ids = {'EURUSD': 1}
    fm = MagicMock()
    fm.assets = ["EURUSD"]
    ml = MagicMock()

    db_file = tmp_path / "test_live_trading.db"
    orchestrator = Orchestrator(client, fm, ml, config={"DB_PATH": str(db_file)})

    pos = MagicMock()
    pos.positionId = 123
    pos.price = 1.0500
    pos.tradeData.symbolId = 1
    pos.tradeData.volume = 10000000
    # missing SL/TP
    pos.stopLoss = 0
    pos.takeProfit = 0

    reconcile_res = MagicMock()
    reconcile_res.position = [pos]

    client.fetch_open_positions.return_value = defer.succeed(reconcile_res)

    with patch.object(Orchestrator, "_attach_missing_sltp_for_pos") as mock_attach:
        mock_attach.return_value = defer.succeed(None)

        await orchestrator.sync_active_positions()

        # Verify sync still completes and attempts to attach missing SL/TP
        assert orchestrator.active_positions == {1: 123}
        mock_attach.assert_called_once_with(pos)

@pytest.mark.asyncio
async def test_sync_active_positions_empty(tmp_path):
    client = MagicMock()
    fm = MagicMock()
    fm.assets = ["EURUSD"]
    ml = MagicMock()

    db_file = tmp_path / "test_live_trading.db"
    orchestrator = Orchestrator(client, fm, ml, config={"DB_PATH": str(db_file)})
    orchestrator.active_positions = {1: 123}
    orchestrator.entry_prices = {123: 1.0500}

    # Reconcile returns empty position list
    reconcile_res = MagicMock()
    reconcile_res.position = []
    client.fetch_open_positions.return_value = defer.succeed(reconcile_res)

    await orchestrator.sync_active_positions()

    # Active positions and entry prices cleared
    assert orchestrator.active_positions == {}
    assert orchestrator.entry_prices == {}
    assert orchestrator.portfolio_state['num_open_positions'] == 0
