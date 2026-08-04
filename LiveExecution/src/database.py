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
                        reason TEXT,
                        sl REAL,
                        tp REAL,
                        relative_sl INTEGER,
                        relative_tp INTEGER,
                        confidence REAL
                    )
                ''')

                # Migration check for existing databases missing SL/TP columns
                cursor.execute("PRAGMA table_info(trades)")
                existing_cols = [row[1] for row in cursor.fetchall()]
                if 'sl' not in existing_cols:
                    cursor.execute("ALTER TABLE trades ADD COLUMN sl REAL")
                if 'tp' not in existing_cols:
                    cursor.execute("ALTER TABLE trades ADD COLUMN tp REAL")
                if 'relative_sl' not in existing_cols:
                    cursor.execute("ALTER TABLE trades ADD COLUMN relative_sl INTEGER")
                if 'relative_tp' not in existing_cols:
                    cursor.execute("ALTER TABLE trades ADD COLUMN relative_tp INTEGER")
                if 'confidence' not in existing_cols:
                    cursor.execute("ALTER TABLE trades ADD COLUMN confidence REAL")

                conn.commit()
                self.logger.debug(f"Database initialized at {self.db_path}")
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

    def log_trade_opening(self, pos_id, symbol, action, size, entry_price, sl=None, tp=None, relative_sl=None, relative_tp=None, confidence=None):
        """Logs a new trade opening with optional SL, TP and model confidence levels."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Check if position already exists to avoid overwriting existing SL/TP with None
                cursor.execute('SELECT sl, tp, relative_sl, relative_tp, confidence FROM trades WHERE pos_id = ?', (pos_id,))
                existing = cursor.fetchone()
                if existing:
                    sl = sl if sl is not None else existing[0]
                    tp = tp if tp is not None else existing[1]
                    relative_sl = relative_sl if relative_sl is not None else existing[2]
                    relative_tp = relative_tp if relative_tp is not None else existing[3]
                    confidence = confidence if confidence is not None else existing[4]

                cursor.execute('''
                    INSERT OR REPLACE INTO trades (pos_id, symbol, action, size, entry_price, entry_time, sl, tp, relative_sl, relative_tp, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (pos_id, symbol, action, size, entry_price, datetime.now().isoformat(), sl, tp, relative_sl, relative_tp, confidence))
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

    def get_trade_by_pos_id(self, pos_id):
        """Retrieves a single trade record by pos_id."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM trades WHERE pos_id = ?', (pos_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            self.logger.error(f"Database error getting trade {pos_id}: {e}")
            return None

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

    def get_daily_stats(self):
        """Calculates statistics for trades closed today (UTC)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                # Get trades closed today
                cursor.execute('''
                    SELECT count(*) as count, sum(net_pnl) as total_pnl, 
                    sum(case when net_pnl > 0 then 1 else 0 end) as wins 
                    FROM trades 
                    WHERE exit_time >= date('now')
                ''')
                row = cursor.fetchone()
                if row:
                    count = row['count']
                    total_pnl = row['total_pnl'] or 0.0
                    wins = row['wins']
                    win_rate = (wins / count * 100) if count > 0 else 0
                    return {'count': count, 'pnl': total_pnl, 'win_rate': win_rate}
                return {'count': 0, 'pnl': 0.0, 'win_rate': 0.0}
        except Exception as e:
            self.logger.error(f"Database error getting daily stats: {e}")
            return {'count': 0, 'pnl': 0.0, 'win_rate': 0.0}

    def get_performance_metrics(self):
        """Calculates all-time performance metrics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT count(*) as count, sum(net_pnl) as total_pnl,
                    sum(case when net_pnl > 0 then 1 else 0 end) as wins
                    FROM trades
                    WHERE exit_time IS NOT NULL
                ''')
                row = cursor.fetchone()
                if row:
                    count = row['count']
                    total_pnl = row['total_pnl'] or 0.0
                    wins = row['wins']
                    win_rate = (wins / count * 100) if count > 0 else 0
                    return {'total_trades': count, 'total_pnl': total_pnl, 'win_rate': win_rate}
                return {'total_trades': 0, 'total_pnl': 0.0, 'win_rate': 0.0}
        except Exception as e:
            self.logger.error(f"Database error getting performance metrics: {e}")
            return {'total_trades': 0, 'total_pnl': 0.0, 'win_rate': 0.0}
