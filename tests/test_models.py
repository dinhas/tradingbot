import os
import sys
import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from LiveExecution.src.models import ModelLoader

def test_model_loader_init():
    ml = ModelLoader()
    assert ml.alpha_threshold == 0.60
    assert ml.filter_threshold == 0.72
    assert ml.sl_multiplier == 2.0
    assert ml.tp_multiplier == 4.0
    assert isinstance(ml.device, torch.device)

def test_load_all_models_missing_alpha(tmp_path):
    ml = ModelLoader()
    ml.project_root = tmp_path

    # Assert return False when alpha model file is missing
    assert ml.load_all_models() is False

def test_load_all_models_missing_filter(tmp_path, caplog):
    ml = ModelLoader()
    ml.project_root = tmp_path

    # Create fake alpha checkpoint
    alpha_dir = tmp_path / "Alpha" / "models"
    alpha_dir.mkdir(parents=True, exist_ok=True)
    alpha_path = alpha_dir / "alpha_model.pth"
    alpha_path.touch() # make sure exists() is True

    fake_checkpoint = {
        "format_version": 7,
        "sequence_length": 25,
        "model_config": {
            "input_dim": 43,
            "lstm_units": 64,
            "dense_units": 32,
            "dropout": 0.1,
            "num_assets": 4,
            "asset_embedding_dim": 4,
            "num_layers": 3,
            "num_heads": 4,
            "bidirectional": True
        },
        "model_state_dict": {}
    }

    # We need to mock _build_alpha_model since state_dict is empty
    with patch.object(ModelLoader, "_build_alpha_model") as mock_build, \
         patch("torch.load", return_value=fake_checkpoint):
        mock_model = MagicMock()
        mock_build.return_value = mock_model

        # When filter is missing, load_all_models warning logged, returns True (since filter is optional)
        with caplog.at_level("WARNING"):
            res = ml.load_all_models()
            assert res is True
            assert ml.alpha_model is not None
            assert ml.filter_ensemble is None
            assert any("Filter ensemble not found" in record.message for record in caplog.records)

def test_load_all_models_both_exist(tmp_path):
    ml = ModelLoader()
    ml.project_root = tmp_path

    alpha_dir = tmp_path / "Alpha" / "models"
    alpha_dir.mkdir(parents=True, exist_ok=True)
    alpha_path = alpha_dir / "alpha_model.pth"

    fake_checkpoint = {
        "format_version": 7,
        "sequence_length": 25,
        "model_config": {
            "input_dim": 43, "lstm_units": 64, "dense_units": 32, "dropout": 0.1
        },
        "model_state_dict": {}
    }

    filter_dir = tmp_path / "Filter" / "models"
    filter_dir.mkdir(parents=True, exist_ok=True)
    filter_path = filter_dir / "filter_rf_ensemble.joblib"

    fake_ensemble = {
        "rf1": MagicMock(),
        "rf2": MagicMock(),
        "gb": MagicMock(),
        "meta": MagicMock(),
        "threshold": 0.65
    }

    with patch.object(ModelLoader, "_build_alpha_model") as mock_build, \
         patch("torch.load", return_value=fake_checkpoint), \
         patch("joblib.load", return_value=fake_ensemble):
        mock_model = MagicMock()
        mock_build.return_value = mock_model

        # Write empty file to make exists() happy
        alpha_path.touch()
        filter_path.touch()

        res = ml.load_all_models()
        assert res is True
        assert ml.alpha_model is not None
        assert ml.filter_ensemble is not None
        assert ml.filter_threshold == 0.65

def test_get_alpha_signal_not_loaded():
    ml = ModelLoader()
    with pytest.raises(RuntimeError, match="Alpha model not loaded"):
        ml.get_alpha_signal(np.zeros((25, 43)))

def test_get_alpha_signal_success():
    ml = ModelLoader()
    ml.alpha_model = MagicMock()

    # mock forward pass of alpha model
    # out = self.alpha_model(...) -> return_dict=True -> out["action_logits"]
    fake_out = {"action_logits": torch.tensor([[1.5], [-0.5]])}
    ml.alpha_model.return_value = fake_out

    # test with shape (2, 25, 43)
    obs = np.zeros((2, 25, 43))
    res = ml.get_alpha_signal(obs, threshold=0.60)

    # 1.5 sigmoid is ~0.817 (>= 0.60 -> action=1, confidence=0.817)
    # -0.5 sigmoid is ~0.377 (< 0.60 -> action=-1, confidence=1 - 0.377 = 0.623)
    assert res["action"][0] == 1
    assert res["action"][1] == -1
    assert np.allclose(res["confidence"][0], 1.0 / (1.0 + np.exp(-1.5)))
    assert np.allclose(res["confidence"][1], 1.0 - (1.0 / (1.0 + np.exp(0.5))))

def test_get_filter_signal_no_filter():
    ml = ModelLoader()
    ml.filter_ensemble = None
    res = ml.get_filter_signal(np.zeros(26))
    assert res["should_trade"] is True
    assert res["confidence"] == 1.0

def test_get_filter_signal_success():
    ml = ModelLoader()
    rf1 = MagicMock()
    rf2 = MagicMock()
    gb = MagicMock()
    meta = MagicMock()

    rf1.predict_proba.return_value = np.array([[0.3, 0.7]]) # prob for 1 is 0.7
    rf2.predict_proba.return_value = np.array([[0.4, 0.6]]) # prob for 1 is 0.6
    gb.predict_proba.return_value = np.array([[0.2, 0.8]])  # prob for 1 is 0.8
    meta.predict_proba.return_value = np.array([[0.2, 0.8]]) # prob for 1 is 0.8

    ml.filter_ensemble = {
        "rf1": rf1,
        "rf2": rf2,
        "gb": gb,
        "meta": meta,
        "threshold": 0.72
    }
    ml.filter_threshold = 0.72

    res = ml.get_filter_signal(np.zeros(26))
    assert res["should_trade"] is True
    assert res["confidence"] == 0.8

    # Verify meta learner inputs
    meta.predict_proba.assert_called_once()
    meta_input = meta.predict_proba.call_args[0][0]
    assert np.allclose(meta_input, np.array([[0.7, 0.6, 0.8]]))
