import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from LiveExecution.src.orchestrator import Orchestrator

def test_run_inference_chain_filter_rejects(tmp_path):
    client = MagicMock()
    client.symbol_ids = {'EURUSD': 1}
    fm = MagicMock()
    fm.assets = ["EURUSD"]

    ml = MagicMock()
    # Mock filter loaded and rejects (should_trade=False)
    ml.filter_ensemble = {}
    ml.get_filter_signal.return_value = {"should_trade": False, "confidence": 0.50}
    ml.alpha_sequence_length = 25

    # We must patch DatabaseManager initialization as it relies on directory creation
    with patch("LiveExecution.src.orchestrator.DatabaseManager"):
        orchestrator = Orchestrator(client, fm, ml, config={"DB_PATH": str(tmp_path / "test.db")})

        # 1 is EURUSD
        res = orchestrator.run_inference_chain(1)
        assert res == {"action": 0}

def test_run_inference_chain_alpha_rejects(tmp_path):
    client = MagicMock()
    client.symbol_ids = {'EURUSD': 1}
    fm = MagicMock()
    fm.assets = ["EURUSD"]

    ml = MagicMock()
    # Mock filter passes
    ml.filter_ensemble = {}
    ml.get_filter_signal.return_value = {"should_trade": True, "confidence": 0.80}

    # Mock alpha rejects (confidence < alpha_threshold)
    ml.alpha_sequence_length = 25
    ml.alpha_threshold = 0.60
    ml.get_alpha_signal.return_value = {
        "direction": np.array([1.0]),
        "confidence": np.array([0.55]),
        "action": np.array([1]),
        "buy_prob": np.array([0.55])
    }

    with patch("LiveExecution.src.orchestrator.DatabaseManager"):
        orchestrator = Orchestrator(client, fm, ml, config={"DB_PATH": str(tmp_path / "test.db")})

        res = orchestrator.run_inference_chain(1)
        assert res == {"action": 0}

def test_run_inference_chain_buy_pass(tmp_path):
    client = MagicMock()
    client.symbol_ids = {'EURUSD': 1}
    fm = MagicMock()
    fm.assets = ["EURUSD"]
    # Mock history and ATR
    idx = pd.date_range(start='2023-10-01 10:00:00', periods=2, freq='5min')
    fm.history = {"EURUSD": pd.DataFrame({'close': [1.0500, 1.0520]}, index=idx)}
    fm.get_atr.return_value = 0.0010
    fm.get_filter_features.return_value = np.zeros((1, 26)) # non-empty filter features
    fm.get_alpha_sequence.return_value = np.zeros((25, 43))

    ml = MagicMock()
    ml.filter_ensemble = {}
    ml.get_filter_signal.return_value = {"should_trade": True, "confidence": 0.80}
    ml.alpha_sequence_length = 25
    ml.alpha_threshold = 0.60
    # direction 1 means BUY
    ml.get_alpha_signal.return_value = {
        "direction": np.array([1.0]),
        "confidence": np.array([0.75]),
        "action": np.array([1]),
        "buy_prob": np.array([0.75])
    }
    ml.sl_multiplier = 2.0
    ml.tp_multiplier = 4.0

    with patch("LiveExecution.src.orchestrator.DatabaseManager"):
        orchestrator = Orchestrator(client, fm, ml, config={"DB_PATH": str(tmp_path / "test.db")})
        orchestrator.portfolio_state['equity'] = 10000.0
        orchestrator.symbol_digits = {'EURUSD': 5}

        res = orchestrator.run_inference_chain(1)
        # BUY action should map to 1
        assert res["action"] == 1
        assert res["asset"] == "EURUSD"
        # SL = price - direction * sl_multiplier * ATR -> 1.0520 - 1 * 2 * 0.001 = 1.0500
        assert np.allclose(res["sl"], 1.0500)
        # TP = price + direction * tp_multiplier * ATR -> 1.0520 + 1 * 4 * 0.001 = 1.0560
        assert np.allclose(res["tp"], 1.0560)
        assert res["lots"] > 0

def test_run_inference_chain_sell_pass(tmp_path):
    client = MagicMock()
    client.symbol_ids = {'EURUSD': 1}
    fm = MagicMock()
    fm.assets = ["EURUSD"]
    idx = pd.date_range(start='2023-10-01 10:00:00', periods=2, freq='5min')
    fm.history = {"EURUSD": pd.DataFrame({'close': [1.0500, 1.0520]}, index=idx)}
    fm.get_atr.return_value = 0.0010
    fm.get_filter_features.return_value = np.zeros((1, 26))
    fm.get_alpha_sequence.return_value = np.zeros((25, 43))

    ml = MagicMock()
    ml.filter_ensemble = {}
    ml.get_filter_signal.return_value = {"should_trade": True, "confidence": 0.80}
    ml.alpha_sequence_length = 25
    ml.alpha_threshold = 0.60
    # direction -1 means SELL
    ml.get_alpha_signal.return_value = {
        "direction": np.array([-1.0]),
        "confidence": np.array([0.75]),
        "action": np.array([-1]),
        "buy_prob": np.array([0.25])
    }
    ml.sl_multiplier = 2.0
    ml.tp_multiplier = 4.0

    with patch("LiveExecution.src.orchestrator.DatabaseManager"):
        orchestrator = Orchestrator(client, fm, ml, config={"DB_PATH": str(tmp_path / "test.db")})
        orchestrator.portfolio_state['equity'] = 10000.0
        orchestrator.symbol_digits = {'EURUSD': 5}

        res = orchestrator.run_inference_chain(1)
        # SELL action maps to 2
        assert res["action"] == 2
        assert res["asset"] == "EURUSD"
        # For SELL: SL is above entry, TP is below entry
        # SL = price - direction * sl_multiplier * ATR -> 1.0520 - (-1) * 2 * 0.001 = 1.0540
        assert np.allclose(res["sl"], 1.0540)
        # TP = price + direction * tp_multiplier * ATR -> 1.0520 + (-1) * 4 * 0.001 = 1.0480
        assert np.allclose(res["tp"], 1.0480)

def test_run_inference_chain_atr_zero(tmp_path):
    client = MagicMock()
    client.symbol_ids = {'EURUSD': 1}
    fm = MagicMock()
    fm.assets = ["EURUSD"]
    idx = pd.date_range(start='2023-10-01 10:00:00', periods=2, freq='5min')
    fm.history = {"EURUSD": pd.DataFrame({'close': [1.0500, 1.0520]}, index=idx)}
    fm.get_atr.return_value = 0.0 # ATR=0
    fm.get_filter_features.return_value = np.zeros((1, 26))
    fm.get_alpha_sequence.return_value = np.zeros((25, 43))

    ml = MagicMock()
    ml.filter_ensemble = {}
    ml.get_filter_signal.return_value = {"should_trade": True, "confidence": 0.80}
    ml.alpha_sequence_length = 25
    ml.alpha_threshold = 0.60
    ml.get_alpha_signal.return_value = {
        "direction": np.array([1.0]),
        "confidence": np.array([0.75]),
        "action": np.array([1]),
        "buy_prob": np.array([0.75])
    }
    ml.sl_multiplier = 2.0
    ml.tp_multiplier = 4.0

    with patch("LiveExecution.src.orchestrator.DatabaseManager"):
        orchestrator = Orchestrator(client, fm, ml, config={"DB_PATH": str(tmp_path / "test.db")})
        orchestrator.portfolio_state['equity'] = 10000.0
        orchestrator.symbol_digits = {'EURUSD': 5}

        res = orchestrator.run_inference_chain(1)
        # Fallback atr_scaled = price * 0.0001 = 1.052 * 0.0001 = 0.0001052
        # sl_dist = 2 * 0.0001052 = 0.0002104
        # relative_sl = max(int(round(sl_dist * 100000 / 1) * 1), 1) = 21 pips
        # sl_price = 1.0520 - 21 * 0.00001 = 1.05179
        assert res["action"] == 1
        assert res["relative_sl"] == 21
