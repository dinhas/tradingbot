import pytest
import os
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.testclient import TestClient
from LiveExecution.dashboard.main import DashboardServer

def test_dashboard_init():
    orchestrator = MagicMock()
    server = DashboardServer(orchestrator)
    assert server.orchestrator == orchestrator
    assert server.app.title == "TradeGuard Dashboard"
    assert len(server.active_connections) == 0

def test_dashboard_routes(tmp_path):
    orchestrator = MagicMock()
    # Mock database responses
    orchestrator.db.get_recent_trades.return_value = [{"pos_id": 1, "symbol": "EURUSD", "action": "BUY", "size": 0.1, "entry_price": 1.0500, "exit_price": 1.0550, "pnl": 50.0, "net_pnl": 48.0, "reason": "TP", "exit_time": "2023-10-01T12:00:00"}]
    orchestrator.db.get_active_trades.return_value = [{"pos_id": 2, "symbol": "EURUSD", "action": "BUY", "size": 0.1, "entry_price": 1.0500, "sl": 1.0450, "tp": 1.0550}]
    orchestrator.portfolio_state = {"balance": 10000.0, "equity": 10050.0}
    orchestrator.db.get_daily_stats.return_value = {"count": 1, "pnl": 48.0, "win_rate": 100.0}
    orchestrator.db.get_performance_metrics.return_value = {"total_trades": 1, "total_pnl": 48.0, "win_rate": 100.0}
    orchestrator.fm.history = {}

    server = DashboardServer(orchestrator)
    client = TestClient(server.app)

    # 1. GET / index page template rendering
    # Patch TemplateResponse to verify correct variables are passed
    with patch.object(server.templates, "TemplateResponse") as mock_template_resp:
        # FastAPI TestClient serializes return values, and TemplateResponse typically returns HTMLResponse
        from fastapi.responses import HTMLResponse
        mock_template_resp.return_value = HTMLResponse(content="Rendered Template", status_code=200)
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.text == "Rendered Template"
        mock_template_resp.assert_called_once()
        context = mock_template_resp.call_args[0][1]
        assert context["state"] == {"balance": 10000.0, "equity": 10050.0}
        assert context["recent_trades"][0]["pos_id"] == 1
        assert context["active_trades"][0]["pos_id"] == 2
        assert context["daily_stats"]["pnl"] == 48.0

    # 2. GET /api/equity_history
    orchestrator.db.get_equity_history.return_value = [{"timestamp": "2023-10-01T10:00:00", "equity": 10000.0}]
    resp = client.get("/api/equity_history")
    assert resp.status_code == 200
    assert resp.json() == [{"timestamp": "2023-10-01T10:00:00", "equity": 10000.0}]

    # 3. GET /api/system_health
    orchestrator.start_time = 1000.0
    orchestrator.last_inference_time = 1050.0
    orchestrator.client.is_connected = True
    with patch("time.time", return_value=1100.0):
        resp = client.get("/api/system_health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["uptime"] == 100.0
        assert data["last_inference"] == "50s ago"
        assert data["connection_status"] == "Connected"

@pytest.mark.asyncio
async def test_dashboard_ws_connect_disconnect():
    orchestrator = MagicMock()
    server = DashboardServer(orchestrator)

    mock_websocket = AsyncMock(spec=WebSocket)
    # mock receive_text to raise WebSocketDisconnect on first call to simulate disconnect
    mock_websocket.receive_text.side_effect = WebSocketDisconnect()

    # Run websocket_endpoint
    # We retrieve the websocket handler registered in server._setup_routes()
    # Or we can just call the endpoint function
    # Let's extract ws_endpoint route
    ws_route = None
    for route in server.app.routes:
        if route.path == "/ws":
            ws_route = route.endpoint
            break

    assert ws_route is not None
    await ws_route(mock_websocket)

    # WebSocket accept called, but on disconnect connection removed from active set
    mock_websocket.accept.assert_called_once()
    assert len(server.active_connections) == 0

@pytest.mark.asyncio
async def test_dashboard_broadcast_update():
    orchestrator = MagicMock()
    server = DashboardServer(orchestrator)

    ws1 = AsyncMock(spec=WebSocket)
    ws2 = AsyncMock(spec=WebSocket)
    server.active_connections.add(ws1)
    server.active_connections.add(ws2)

    # ws2 raises exception on send_text (e.g. disconnected client)
    ws2.send_text.side_effect = Exception("Disconnected")

    await server.broadcast_update("trade_opened", {"symbol": "EURUSD"})

    # ws1 should receive message
    ws1.send_text.assert_called_once_with('{"type": "trade_opened", "data": {"symbol": "EURUSD"}}')
    # ws2 failed, so should be removed from active_connections
    assert ws1 in server.active_connections
    assert ws2 not in server.active_connections

def test_dashboard_api_actions():
    orchestrator = MagicMock()
    orchestrator.active_positions = {1: 123} # symbolId 1 -> pos_id 123
    server = DashboardServer(orchestrator)
    client = TestClient(server.app)

    # Mock twisted reactor.callFromThread
    with patch("LiveExecution.dashboard.main.reactor.callFromThread") as mock_call:
        # 1. POST /api/close/{pos_id}
        resp = client.post("/api/close/123")
        assert resp.status_code == 200
        assert resp.json() == {"status": "closing triggered"}
        mock_call.assert_called_once_with(orchestrator.close_position_by_id, 123, 1)

        # 2. POST /api/kill
        mock_call.reset_mock()
        resp = client.post("/api/kill")
        assert resp.status_code == 200
        assert resp.json() == {"status": "kill switch activated"}
        mock_call.assert_called_once_with(orchestrator.kill_switch)
