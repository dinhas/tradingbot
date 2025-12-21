import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.calibration import calibration_curve
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging

class ModelVisualizer:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_confusion_matrix(self, y_true, y_pred, filename="confusion_matrix.png"):
        """
        Generates and saves a confusion matrix plot.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            filename: Output filename
        """
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap='Blues')
        plt.title("Confusion Matrix")
        
        save_path = self.output_dir / filename
        plt.savefig(save_path)
        plt.close(fig)
        logging.info(f"Saved confusion matrix to {save_path}")

    def plot_feature_importance(self, model, filename="feature_importance.png", max_features=20):
        """
        Generates and saves a feature importance plot for LightGBM.
        
        Args:
            model: Trained LightGBM booster
            filename: Output filename
            max_features: Maximum number of features to display
        """
        # LightGBM specific
        if not hasattr(model, 'feature_importance') or not hasattr(model, 'feature_name'):
            logging.warning("Model does not have feature_importance or feature_name methods.")
            return

        importance = model.feature_importance(importance_type='gain')
        names = model.feature_name()
        
        df_imp = pd.DataFrame({'feature': names, 'importance': importance})
        df_imp = df_imp.sort_values('importance', ascending=False).head(max_features)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(df_imp))
        ax.barh(y_pos, df_imp['importance'], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_imp['feature'])
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Importance (Gain)')
        ax.set_title('Feature Importance')
        
        save_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        logging.info(f"Saved feature importance to {save_path}")

    def plot_calibration_curve(self, y_true, y_prob, filename="calibration_curve.png"):
        """
        Generates and saves a calibration curve.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            filename: Output filename
        """
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Model')
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
        
        ax.set_ylabel("Fraction of positives")
        ax.set_xlabel("Mean predicted value")
        ax.set_title("Calibration Curve")
        ax.legend()
        
        save_path = self.output_dir / filename
        plt.savefig(save_path)
        plt.close(fig)
        logging.info(f"Saved calibration curve to {save_path}")

    def plot_roc_curve(self, y_true, y_prob, filename="roc_curve.png"):
        """
        Generates and saves an ROC curve.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            filename: Output filename
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        
        save_path = self.output_dir / filename
        plt.savefig(save_path)
        plt.close(fig)
        logging.info(f"Saved ROC curve to {save_path}")

    def save_metadata(self, metrics, threshold, filename="model_metadata.json"):
        """
        Saves model metadata including metrics and threshold to JSON.
        
        Args:
            metrics: Dictionary of metrics
            threshold: Optimal threshold
            filename: Output filename
        """
        data = {
            "metrics": metrics,
            "threshold": threshold
        }
        save_path = self.output_dir / filename
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=4)
        logging.info(f"Saved metadata to {save_path}")

    def save_model(self, model, filename="guard_model.txt"):
        """
        Saves the LightGBM model to a text file.
        
        Args:
            model: Trained LightGBM booster
            filename: Output filename
        """
        save_path = self.output_dir / filename
        model.save_model(str(save_path))
        logging.info(f"Saved model to {save_path}")
