import os
import pytest
import json
from pathlib import Path
from LiveExecution.src.config import load_config, get_thresholds, ConfigError, DEFAULT_THRESHOLDS

def test_load_config_valid():
    override_env = {
        "CT_APP_ID": "app_id",
        "CT_APP_SECRET": "secret",
        "CT_ACCOUNT_ID": "12345",
        "CT_ACCESS_TOKEN": "token",
        "CT_HOST_TYPE": "LIVE",
        "DB_PATH": "test_db.db",
        "ALPHA_CONFIDENCE_THRESHOLD": "0.65",
        "FILTER_THRESHOLD": "0.75",
        "SL_MULTIPLIER": "1.5",
        "TP_MULTIPLIER": "3.5"
    }
    cfg = load_config(override_env)
    assert cfg["CT_APP_ID"] == "app_id"
    assert cfg["CT_APP_SECRET"] == "secret"
    assert cfg["CT_ACCOUNT_ID"] == 12345
    assert cfg["CT_ACCESS_TOKEN"] == "token"
    assert cfg["CT_HOST_TYPE"] == "live"
    assert cfg["DB_PATH"] == "test_db.db"
    assert cfg["ALPHA_CONFIDENCE_THRESHOLD"] == 0.65
    assert cfg["FILTER_THRESHOLD"] == 0.75
    assert cfg["SL_MULTIPLIER"] == 1.5
    assert cfg["TP_MULTIPLIER"] == 3.5

def test_load_config_missing_vars():
    override_env = {
        "CT_APP_ID": "app_id",
        "CT_APP_SECRET": "secret",
        # Missing CT_ACCOUNT_ID and CT_ACCESS_TOKEN
    }
    with pytest.raises(ConfigError) as exc_info:
        load_config(override_env)
    assert "Missing required environment variables" in str(exc_info.value)

def test_load_config_invalid_account_id():
    override_env = {
        "CT_APP_ID": "app_id",
        "CT_APP_SECRET": "secret",
        "CT_ACCOUNT_ID": "not-an-int",
        "CT_ACCESS_TOKEN": "token",
    }
    with pytest.raises(ConfigError) as exc_info:
        load_config(override_env)
    assert "CT_ACCOUNT_ID must be an integer" in str(exc_info.value)

def test_load_config_invalid_host_type():
    override_env = {
        "CT_APP_ID": "app_id",
        "CT_APP_SECRET": "secret",
        "CT_ACCOUNT_ID": "12345",
        "CT_ACCESS_TOKEN": "token",
        "CT_HOST_TYPE": "invalid_host"
    }
    with pytest.raises(ConfigError) as exc_info:
        load_config(override_env)
    assert "CT_HOST_TYPE must be either 'demo' or 'live'" in str(exc_info.value)

def test_get_thresholds_file_exists(tmp_path):
    optimal_json = tmp_path / "backtest" / "results" / "optimal_thresholds.json"
    optimal_json.parent.mkdir(parents=True, exist_ok=True)
    params = {
        "alpha_confidence_threshold": 0.58,
        "filter_threshold": 0.69,
        "sl_multiplier": 2.5,
        "tp_multiplier": 4.5
    }
    with open(optimal_json, "w") as f:
        json.dump(params, f)

    thresholds = get_thresholds(project_root=tmp_path)
    assert thresholds["alpha_confidence_threshold"] == 0.58
    assert thresholds["filter_threshold"] == 0.69
    assert thresholds["sl_multiplier"] == 2.5
    assert thresholds["tp_multiplier"] == 4.5

def test_get_thresholds_file_missing(tmp_path):
    thresholds = get_thresholds(project_root=tmp_path)
    assert thresholds == DEFAULT_THRESHOLDS

def test_get_thresholds_file_malformed(tmp_path):
    optimal_json = tmp_path / "backtest" / "results" / "optimal_thresholds.json"
    optimal_json.parent.mkdir(parents=True, exist_ok=True)
    with open(optimal_json, "w") as f:
        f.write("{invalid_json}")

    thresholds = get_thresholds(project_root=tmp_path)
    assert thresholds == DEFAULT_THRESHOLDS
