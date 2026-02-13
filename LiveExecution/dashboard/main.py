import threading
import uvicorn
import logging
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from twisted.internet import reactor

class DashboardServer:
    """
    FastAPI-based monitoring dashboard for the trading bot.
    Runs in a separate thread to avoid blocking the Twisted event loop.
    """
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.logger = logging.getLogger("LiveExecution")
        self.app = FastAPI(title="TradeGuard Dashboard")
        
        # Use absolute path for templates
        base_path = Path(__file__).resolve().parent
        self.templates = Jinja2Templates(directory=str(base_path / "templates"))
        
        self._setup_routes()

    def _setup_routes(self):
        @self.app.get("/")
        async def index(request: Request):
            try:
                # Use data from orchestrator
                recent_trades = self.orchestrator.db.get_recent_trades(limit=10)
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
                    
                    # Rough PnL estimation (ignoring fees/swap for display)
                    # Forex: (Price Diff) * Contract Size * Lots
                    # Gold: (Price Diff) * Contract Size * Lots
                    contract_size = 100 if symbol == 'XAUUSD' else 100000
                    pnl = (current_price - entry_price) * direction * contract_size * size
                    
                    trade_copy = dict(trade)
                    trade_copy['current_price'] = current_price
                    trade_copy['unrealized_pnl'] = pnl
                    enriched_active_trades.append(trade_copy)

                return self.templates.TemplateResponse("index.html", {
                    "request": request,
                    "state": state,
                    "recent_trades": recent_trades,
                    "active_trades": enriched_active_trades,
                    "daily_stats": daily_stats,
                    "performance": performance_metrics
                })
            except Exception as e:
                self.logger.error(f"Dashboard error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/equity_history")
        async def equity_history():
            return self.orchestrator.db.get_equity_history(limit=200)

        @self.app.post("/api/close/{pos_id}")
        async def close_position(pos_id: int):
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
            # Schedule kill switch in Twisted thread
            reactor.callFromThread(self.orchestrator.kill_switch)
            return {"status": "kill switch activated"}

    def start(self):
        """Starts the Uvicorn server in a background thread."""
        config = uvicorn.Config(self.app, host="0.0.0.0", port=8000, log_level="warning")
        server = uvicorn.Server(config)

        thread = threading.Thread(target=server.run, daemon=True, name="DashboardThread")
        thread.start()
        self.logger.info("Web Dashboard started on http://0.0.0.0:8000")
