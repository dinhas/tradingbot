import sqlite3
import logging
from datetime import datetime

class DatabaseManager:
    """
    Handles local persistence for account state and trade history.
    """
    def __init__(self, db_path):
        self.db_path = db_path
        self.logger = logging.getLogger("LiveExecution")

        # Ensure data directory exists
        import os
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Account History table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS account_history (
                        timestamp DATETIME PRIMARY KEY,
                        balance REAL,
                        equity REAL,
                        drawdown REAL,
                        margin REAL
                    )
                ''')

                # Trades table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        pos_id INTEGER PRIMARY KEY,
                        symbol TEXT,
                        action TEXT,
                        size REAL,
                        entry_price REAL,
                        entry_time DATETIME,
                        exit_price REAL,
                        exit_time DATETIME,
                        pnl REAL,
                        net_pnl REAL,
                        reason TEXT
                    )
                ''')
                conn.commit()
                self.logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")

    def log_account_state(self, balance, equity, drawdown, margin):
        """Logs a snapshot of the account state."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO account_history (timestamp, balance, equity, drawdown, margin)
                    VALUES (?, ?, ?, ?, ?)
                ''', (datetime.now().isoformat(), balance, equity, drawdown, margin))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Database error logging account state: {e}")

    def log_trade_opening(self, pos_id, symbol, action, size, entry_price):
        """Logs a new trade opening."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO trades (pos_id, symbol, action, size, entry_price, entry_time)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (pos_id, symbol, action, size, entry_price, datetime.now().isoformat()))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Database error logging trade opening: {e}")

    def log_trade_closure(self, pos_id, exit_price, pnl, net_pnl, reason):
        """Updates a trade record with closure details."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE trades
                    SET exit_price = ?, exit_time = ?, pnl = ?, net_pnl = ?, reason = ?
                    WHERE pos_id = ?
                ''', (exit_price, datetime.now().isoformat(), pnl, net_pnl, reason, pos_id))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Database error logging trade closure: {e}")

    def get_active_trades(self):
        """Retrieves currently open trades from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM trades WHERE exit_time IS NULL')
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Database error getting active trades: {e}")
            return []

    def get_recent_trades(self, limit=50):
        """Retrieves recent completed trades."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM trades WHERE exit_time IS NOT NULL ORDER BY exit_time DESC LIMIT ?', (limit,))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Database error getting recent trades: {e}")
            return []

    def get_equity_history(self, limit=1000):
        """Retrieves equity curve data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('SELECT timestamp, equity FROM account_history ORDER BY timestamp ASC LIMIT ?', (limit,))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Database error getting equity history: {e}")
            return []
