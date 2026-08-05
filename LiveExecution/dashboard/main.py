import threading
import uvicorn
import logging
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from twisted.internet import reactor
import json

class DashboardServer:
    """
    FastAPI-based monitoring dashboard for the trading bot.
    Runs in a separate thread to avoid blocking the Twisted event loop.
    """
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.logger = logging.getLogger("LiveExecution")
        self.app = FastAPI(title="TradeGuard Dashboard")
        self.active_connections = set()
        
        # Use absolute path for templates
        base_path = Path(__file__).resolve().parent
        self.templates = Jinja2Templates(directory=str(base_path / "templates"))
        
        self._setup_routes()

    def _setup_routes(self):
        from fastapi.responses import FileResponse, PlainTextResponse

        @self.app.get("/health")
        @self.app.get("/api/health")
        async def health():
            return {"status": "healthy"}

        @self.app.get("/")
        async def index(request: Request, asset: str = None, status: str = None):
            try:
                # Get the logs directory
                log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
                log_files = []
                if log_dir.exists():
                    for f in sorted(log_dir.glob("tradebot-*.log"), reverse=True):
                        sz_bytes = f.stat().st_size
                        # Convert to human readable format
                        if sz_bytes >= 1024 * 1024:
                            sz_str = f"{sz_bytes / (1024 * 1024):.1f} MB"
                        else:
                            sz_str = f"{sz_bytes / 1024:.1f} KB"
                        # Extract date from filename
                        date_part = f.name[len("tradebot-"):-4]
                        log_files.append({
                            "filename": f.name,
                            "date": date_part,
                            "size": sz_str
                        })

                # Check if orchestrator is initialized yet
                if self.orchestrator is None:
                    return self.templates.TemplateResponse(
                        request=request,
                        name="index.html",
                        context={
                            "state": {"balance": 0.0, "equity": 0.0, "initial_equity": 1.0, "peak_equity": 0.0},
                            "recent_trades": [],
                            "active_trades": [],
                            "daily_stats": {"count": 0, "pnl": 0.0, "win_rate": 0.0},
                            "performance": {"total_trades": 0, "total_pnl": 0.0, "win_rate": 0.0},
                            "log_files": log_files,
                            "selected_asset": asset or "",
                            "selected_status": status or ""
                        }
                    )

                # Use data from orchestrator
                recent_trades = self.orchestrator.db.get_recent_trades(limit=100)
                active_trades = self.orchestrator.db.get_active_trades()
                state = self.orchestrator.portfolio_state
                
                # Fetch new stats
                daily_stats = self.orchestrator.db.get_daily_stats()
                performance_metrics = self.orchestrator.db.get_performance_metrics()

                # Enrich active trades with current price and unrealized PnL
                enriched_active_trades = []
                for trade in active_trades:
                    symbol = trade['symbol']
                    entry_price = trade['entry_price']
                    size = trade['size']
                    direction = 1 if trade['action'] == 'BUY' else -1
                    
                    current_price = entry_price
                    if symbol in self.orchestrator.fm.history and not self.orchestrator.fm.history[symbol].empty:
                        current_price = self.orchestrator.fm.history[symbol].iloc[-1]['close']
                    
                    # Rough PnL estimation
                    contract_size = 100000
                    pnl = (current_price - entry_price) * direction * contract_size * size
                    
                    trade_copy = dict(trade)
                    trade_copy['current_price'] = current_price
                    trade_copy['unrealized_pnl'] = pnl
                    enriched_active_trades.append(trade_copy)

                # Filter by asset
                if asset:
                    enriched_active_trades = [t for t in enriched_active_trades if t['symbol'] == asset]
                    recent_trades = [t for t in recent_trades if t['symbol'] == asset]

                # Filter by status
                show_active = (status != 'closed')
                show_recent = (status != 'open')

                return self.templates.TemplateResponse(
                    request=request,
                    name="index.html",
                    context={
                        "state": state,
                        "recent_trades": recent_trades if show_recent else [],
                        "active_trades": enriched_active_trades if show_active else [],
                        "daily_stats": daily_stats,
                        "performance": performance_metrics,
                        "log_files": log_files,
                        "selected_asset": asset or "",
                        "selected_status": status or ""
                    }
                )
            except Exception as e:
                self.logger.exception(f"Dashboard error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/logs/{filename}")
        async def download_log(filename: str):
            log_dir = (Path(__file__).resolve().parent.parent.parent / "logs").resolve()
            file_path = (log_dir / filename).resolve()

            # Path traversal prevention check
            if not str(file_path).startswith(str(log_dir)):
                raise HTTPException(status_code=403, detail="Access denied: path traversal attempt detected.")

            if not file_path.exists() or not file_path.is_file():
                raise HTTPException(status_code=404, detail="Log file not found.")

            return FileResponse(file_path, media_type="text/plain", filename=filename)

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.active_connections.add(websocket)
            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
                self.active_connections.remove(websocket)

        @self.app.get("/api/equity_history")
        async def equity_history():
            if self.orchestrator is None:
                return []
            return self.orchestrator.db.get_equity_history(limit=200)

        @self.app.get("/api/system_health")
        async def system_health():
            if self.orchestrator is None:
                return {
                    "uptime": 0.0,
                    "last_inference": "System Starting...",
                    "connection_status": "Disconnected",
                    "active_assets": 0
                }
            import time
            uptime = time.time() - self.orchestrator.start_time
            last_inference = "Never"
            if self.orchestrator.last_inference_time > 0:
                last_inference = f"{int(time.time() - self.orchestrator.last_inference_time)}s ago"
            
            return {
                "uptime": uptime,
                "last_inference": last_inference,
                "connection_status": "Connected" if self.orchestrator.client.is_connected else "Disconnected",
                "active_assets": len(self.orchestrator.fm.assets)
            }

        @self.app.post("/api/close/{pos_id}")
        async def close_position(pos_id: int):
            if self.orchestrator is None:
                raise HTTPException(status_code=503, detail="System is starting up")
            symbol_id = None
            for sid, pid in self.orchestrator.active_positions.items():
                if pid == pos_id:
                    symbol_id = sid
                    break

            if symbol_id or pos_id: # Try even if not in active_positions cache
                # Schedule closure in Twisted thread
                reactor.callFromThread(self.orchestrator.close_position_by_id, pos_id, symbol_id)
                return {"status": "closing triggered"}
            else:
                raise HTTPException(status_code=404, detail="Position not found")

        @self.app.post("/api/kill")
        async def kill_switch():
            if self.orchestrator is None:
                raise HTTPException(status_code=503, detail="System is starting up")
            # Schedule kill switch in Twisted thread
            reactor.callFromThread(self.orchestrator.kill_switch)
            return {"status": "kill switch activated"}

    async def broadcast_update(self, event_type, data):
        """Broadcasts an update to all connected WebSocket clients."""
        if not self.active_connections:
            return
            
        message = json.dumps({"type": event_type, "data": data})
        disconnected = []
        for websocket in self.active_connections:
            try:
                await websocket.send_text(message)
            except Exception:
                disconnected.append(websocket)
        
        for ws in disconnected:
            self.active_connections.remove(ws)

    def start(self):
        """Starts the Uvicorn server in a background thread."""
        import os
        port = int(os.getenv("PORT", 8080))
        config = uvicorn.Config(self.app, host="0.0.0.0", port=port, log_level="warning")
        server = uvicorn.Server(config)

        thread = threading.Thread(target=server.run, daemon=True, name="DashboardThread")
        thread.start()
        self.logger.info(f"Web Dashboard started on http://0.0.0.0:{port}")
