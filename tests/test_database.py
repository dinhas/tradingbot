import os
import pytest
import sqlite3
from datetime import datetime
from LiveExecution.src.database import DatabaseManager

def test_database_init(tmp_path):
    db_file = tmp_path / "test_live_trading.db"
    db = DatabaseManager(str(db_file))

    # Verify tables exist
    with sqlite3.connect(db.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        assert "account_history" in tables
        assert "trades" in tables

def test_database_migration_adds_missing_columns(tmp_path):
    db_file = tmp_path / "test_migration.db"

    # Create database with older schema
    with sqlite3.connect(str(db_file)) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE trades (
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

    # Instantiate DatabaseManager (triggering migration)
    db = DatabaseManager(str(db_file))

    # Verify new columns added
    with sqlite3.connect(db.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(trades)")
        cols = [row[1] for row in cursor.fetchall()]
        assert 'sl' in cols
        assert 'tp' in cols
        assert 'relative_sl' in cols
        assert 'relative_tp' in cols

def test_database_operations(tmp_path):
    db_file = tmp_path / "test_ops.db"
    db = DatabaseManager(str(db_file))

    # 1. log_trade_opening INSERT
    db.log_trade_opening(123, "EURUSD", "BUY", 1.0, 1.0500, sl=1.0450, tp=1.0550)

    trade = db.get_trade_by_pos_id(123)
    assert trade is not None
    assert trade["symbol"] == "EURUSD"
    assert trade["entry_price"] == 1.0500
    assert trade["sl"] == 1.0450
    assert trade["tp"] == 1.0550
    assert trade["exit_price"] is None

    # 2. log_trade_opening with existing pos_id + None SL/TP -> verify preserved
    db.log_trade_opening(123, "EURUSD", "BUY", 1.0, 1.0500, sl=None, tp=None)
    trade_preserved = db.get_trade_by_pos_id(123)
    assert trade_preserved["sl"] == 1.0450
    assert trade_preserved["tp"] == 1.0550

    # 3. get_active_trades
    active = db.get_active_trades()
    assert len(active) == 1
    assert active[0]["pos_id"] == 123

    # 4. log_trade_closure UPDATE with exit details
    db.log_trade_closure(123, 1.0520, 200.0, 195.0, "TP")
    closed_trade = db.get_trade_by_pos_id(123)
    assert closed_trade["exit_price"] == 1.0520
    assert closed_trade["pnl"] == 200.0
    assert closed_trade["net_pnl"] == 195.0
    assert closed_trade["reason"] == "TP"
    assert closed_trade["exit_time"] is not None

    # 5. get_recent_trades with limit
    db.log_trade_opening(124, "GBPUSD", "SELL", 0.5, 1.2500)
    db.log_trade_closure(124, 1.2400, 50.0, 48.0, "SIGNAL")

    recent_all = db.get_recent_trades(limit=10)
    assert len(recent_all) == 2

    recent_limited = db.get_recent_trades(limit=1)
    assert len(recent_limited) == 1
    # Should return most recent first
    assert recent_limited[0]["pos_id"] == 124

    # 6. log_account_state & get_equity_history
    db.log_account_state(10000.0, 10200.0, 0.0, 1000.0)
    db.log_account_state(10000.0, 10250.0, 0.0, 1000.0)

    eq_history = db.get_equity_history()
    assert len(eq_history) == 2
    assert eq_history[0]["equity"] == 10200.0
    assert eq_history[1]["equity"] == 10250.0

    # 7. get_daily_stats
    stats = db.get_daily_stats()
    assert stats["count"] == 2
    assert stats["pnl"] == 243.0 # 195.0 + 48.0
    assert stats["win_rate"] == 100.0

    # 8. get_performance_metrics
    metrics = db.get_performance_metrics()
    assert metrics["total_trades"] == 2
    assert metrics["total_pnl"] == 243.0
    assert metrics["win_rate"] == 100.0

def test_get_trade_by_pos_id_missing(tmp_path):
    db_file = tmp_path / "test_ops.db"
    db = DatabaseManager(str(db_file))
    assert db.get_trade_by_pos_id(999) is None
