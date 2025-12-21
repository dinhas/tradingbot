import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import lightgbm as lgb
import sys
import os

# Add src to path if needed, though pytest usually handles it if conftest is set up or installed
# For now, we assume TradeGuard package structure
from TradeGuard.src.visualization import ModelVisualizer

@pytest.fixture
def sample_data():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.2, 0.8, 0.6])
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob

@pytest.fixture
def mock_model():
    model = MagicMock(spec=lgb.Booster)
    model.feature_importance.return_value = [10, 5, 2]
    model.feature_name.return_value = ['feature1', 'feature2', 'feature3']
    return model

@pytest.fixture
def output_dir(tmp_path):
    return tmp_path

class TestModelVisualizer:
    
    def test_class_exists(self):
        """Test that ModelVisualizer class exists."""
        if ModelVisualizer is None:
            pytest.fail("ModelVisualizer class could not be imported from TradeGuard.src.visualization")
            
    def test_plot_confusion_matrix(self, sample_data, output_dir):
        """Test confusion matrix generation."""
        y_true, y_pred, _ = sample_data
        viz = ModelVisualizer(output_dir)
        
        # We mock plt.savefig to avoid actual plotting, but we want to verify the function calls it
        # or that it creates a file if we don't mock. 
        # Ideally, we let it run and check file existence if we trust the env.
        # But to be safe and fast, we can check file existence.
        
        output_file = output_dir / "confusion_matrix.png"
        viz.plot_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png")
        
        assert output_file.exists()
        
    def test_plot_feature_importance(self, mock_model, output_dir):
        """Test feature importance plot generation."""
        viz = ModelVisualizer(output_dir)
        output_file = output_dir / "feature_importance.png"
        
        viz.plot_feature_importance(mock_model, filename="feature_importance.png")
        
        assert output_file.exists()
        
    def test_plot_calibration_curve(self, sample_data, output_dir):
        """Test calibration curve generation."""
        y_true, _, y_prob = sample_data
        viz = ModelVisualizer(output_dir)
        output_file = output_dir / "calibration_curve.png"
        
        viz.plot_calibration_curve(y_true, y_prob, filename="calibration_curve.png")
        
        assert output_file.exists()
        
    def test_plot_roc_curve(self, sample_data, output_dir):
        """Test ROC curve generation."""
        y_true, _, y_prob = sample_data
        viz = ModelVisualizer(output_dir)
        output_file = output_dir / "roc_curve.png"
        
        viz.plot_roc_curve(y_true, y_prob, filename="roc_curve.png")
        
        assert output_file.exists()
        
    def test_save_metadata(self, output_dir):
        """Test metadata export."""
        viz = ModelVisualizer(output_dir)
        output_file = output_dir / "model_metadata.json"
        
        metrics = {"auc": 0.75, "precision": 0.65}
        threshold = 0.55
        
        viz.save_metadata(metrics, threshold, filename="model_metadata.json")
        
        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        assert data['metrics']['auc'] == 0.75
        assert data['threshold'] == 0.55

    def test_save_model(self, mock_model, output_dir):
        """Test model saving."""
        viz = ModelVisualizer(output_dir)
        output_file = output_dir / "guard_model.txt"
        
        viz.save_model(mock_model, filename="guard_model.txt")
        
        mock_model.save_model.assert_called_once()
        # Verify call args if needed, or just that it was called.
        # Since mock_model.save_model is a mock, it won't actually create a file unless side_effect does.
        # So we assert called.