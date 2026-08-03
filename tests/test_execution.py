import pytest
import sqlite3
from unittest.mock import MagicMock, patch
from twisted.internet import defer
from twisted.internet.defer import inlineCallbacks
from LiveExecution.src.orchestrator import Orchestrator
from LiveExecution.src.ctrader_client import CTraderAmendError

# Helper to run twisted deferreds in testing
def run_deferred(d):
    results = []
    errors = []
    d.addCallback(results.append)
    d.addErrback(errors.append)
    if errors:
        raise errors[0].value
    return results[0] if results else None

@pytest.mark.asyncio
async def test_execute_decision_success(tmp_path):
    client = MagicMock()
    fm = MagicMock()
    fm.assets = ["EURUSD"]
    ml = MagicMock()

    # Mock database to use a test file
    db_file = tmp_path / "test_live_trading.db"
    orchestrator = Orchestrator(client, fm, ml, config={"DB_PATH": str(db_file)})

    # Mock client.execute_market_order deferred return
    execution_res = MagicMock()
    # Mock position properties to avoid MagicMock database types
    execution_res.position.positionId = 123
    execution_res.position.price = 1.0500
    execution_res.position.tradeData.volume = 10000000
    execution_res.position.tradeData.tradeSide = 1 # BUY

    client.execute_market_order.return_value = defer.succeed(execution_res)
    client.fetch_open_positions.return_value = defer.succeed(MagicMock())

    decision = {
        'asset': 'EURUSD',
        'action': 1, # BUY
        'lots': 1.0,
        'sl': 1.0450,
        'tp': 1.0550,
        'relative_sl': 50,
        'relative_tp': 50
    }

    # Execute decision
    await orchestrator.execute_decision(decision, 1)

    # Verify DB logged trade opening
    trade = orchestrator.db.get_trade_by_pos_id(123)
    assert trade is not None
    assert trade["symbol"] == "EURUSD"
    assert trade["action"] == "BUY"
    assert trade["entry_price"] == 1.0500
    assert trade["sl"] == 1.0450
    assert trade["tp"] == 1.0550

@pytest.mark.asyncio
async def test_attach_missing_sltp_recovery_from_db(tmp_path):
    client = MagicMock()
    fm = MagicMock()
    fm.assets = ["EURUSD"]
    ml = MagicMock()

    db_file = tmp_path / "test_live_trading.db"
    orchestrator = Orchestrator(client, fm, ml, config={"DB_PATH": str(db_file)})

    # Mock DB trade existing with SL/TP
    orchestrator.db.log_trade_opening(123, "EURUSD", "BUY", 1.0, 1.0500, sl=1.0450, tp=1.0550)

    # Pos mock
    pos = MagicMock()
    pos.positionId = 123
    pos.price = 1.0500
    pos.tradeData.symbolId = 1
    pos.tradeData.tradeSide = 1 # BUY
    pos.tradeData.volume = 10000000 # 1 lot

    client.amend_position_sltp.return_value = defer.succeed(MagicMock())

    await orchestrator._attach_missing_sltp_for_pos(pos)

    # Verify amend was called with the recovered sl/tp from db
    client.amend_position_sltp.assert_called_once_with(123, stop_loss=1.0450, take_profit=1.0550)

@pytest.mark.asyncio
async def test_attach_missing_sltp_invalid_tp_closes_position(tmp_path):
    client = MagicMock()
    fm = MagicMock()
    fm.assets = ["EURUSD"]
    ml = MagicMock()

    db_file = tmp_path / "test_live_trading.db"
    orchestrator = Orchestrator(client, fm, ml, config={"DB_PATH": str(db_file)})

    # Invalid TP for LONG: TP (1.0400) < entry (1.0500)
    orchestrator.db.log_trade_opening(123, "EURUSD", "BUY", 1.0, 1.0500, sl=1.0450, tp=1.0400)

    pos = MagicMock()
    pos.positionId = 123
    pos.price = 1.0500
    pos.tradeData.symbolId = 1
    pos.tradeData.tradeSide = 1 # BUY
    pos.tradeData.volume = 10000000

    client.close_position.return_value = defer.succeed(MagicMock())

    await orchestrator._attach_missing_sltp_for_pos(pos)

    # Position closed due to invalid TP direction
    client.close_position.assert_called_once_with(123, 10000000)

@pytest.mark.asyncio
async def test_attach_missing_sltp_amend_error_closes_position(tmp_path):
    client = MagicMock()
    fm = MagicMock()
    fm.assets = ["EURUSD"]
    ml = MagicMock()

    db_file = tmp_path / "test_live_trading.db"
    orchestrator = Orchestrator(client, fm, ml, config={"DB_PATH": str(db_file)})

    orchestrator.db.log_trade_opening(123, "EURUSD", "BUY", 1.0, 1.0500, sl=1.0450, tp=1.0550)

    pos = MagicMock()
    pos.positionId = 123
    pos.price = 1.0500
    pos.tradeData.symbolId = 1
    pos.tradeData.tradeSide = 1 # BUY
    pos.tradeData.volume = 10000000

    # Simulate CTraderAmendError on amend
    client.amend_position_sltp.side_effect = lambda *args, **kwargs: defer.fail(CTraderAmendError(123, "BAD_SLTP", "Invalid stop loss"))
    client.close_position.return_value = defer.succeed(MagicMock())

    await orchestrator._attach_missing_sltp_for_pos(pos)

    # Position closed for capital protection
    client.close_position.assert_called_once_with(123, 10000000)

@pytest.mark.asyncio
async def test_on_order_execution_filled_open(tmp_path):
    client = MagicMock()
    client.symbol_ids = {'EURUSD': 1}
    fm = MagicMock()
    fm.assets = ["EURUSD"]
    ml = MagicMock()

    db_file = tmp_path / "test_live_trading.db"
    orchestrator = Orchestrator(client, fm, ml, config={"DB_PATH": str(db_file)})

    # Setup execution event for a newly filled open position
    event = MagicMock()

    from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOAExecutionType, ProtoOAPositionStatus
    event.executionType = ProtoOAExecutionType.ORDER_FILLED

    # Explicitly mock spec/attributes so hasattr checks behave correctly or explicitly delete stopLoss/takeProfit
    # Since we are using MagicMock, we can configure stopLoss and takeProfit to be None/0
    event.position.positionId = 123
    event.position.positionStatus = ProtoOAPositionStatus.POSITION_STATUS_OPEN
    event.position.tradeData.symbolId = 1
    event.position.price = 1.0500
    event.position.tradeData.volume = 10000000 # 1 lot (contract_size * 100) -> 100000 * 100 = 10000000
    event.position.tradeData.tradeSide = 1 # BUY
    event.position.stopLoss = 0
    event.position.takeProfit = 0

    # Mock attach_missing method to avoid side-effects
    with patch.object(Orchestrator, "_attach_missing_sltp_for_pos") as mock_attach:
        mock_attach.return_value = defer.succeed(None)

        # Call on_order_execution
        # Since on_order_execution is a generator under the hood or may be inlineCallbacks-like,
        # we can await/run it. Actually, wait! The bug list says:
        # "on_order_execution() uses yield but method is not decorated with @inlineCallbacks"
        # Let's see if we can run it safely. It's a generator so we can consume it if it doesn't have the decorator.
        res = orchestrator.on_order_execution(event)
        if hasattr(res, '__next__'):
            # It's a generator because it uses yield without inlineCallbacks decorator!
            # Let's exhaust it.
            list(res)

        # Verify state updated
        assert orchestrator.active_positions[1] == 123
        assert orchestrator.entry_prices[123] == 1.0500
        mock_attach.assert_called_once()
