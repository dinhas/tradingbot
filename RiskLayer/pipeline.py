"""End-to-end RiskLayer data generation + training pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RiskLayer.generate_dataset import generate_dataset
from RiskLayer.label_generator import generate_labels
from RiskLayer.train import TrainConfig, train

LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run_pipeline(
    data_dir: str,
    output_dir: str,
    alpha_model_path: str,
    seq_len: int,
    adx_threshold: float,
    atr_threshold: float | None,
    alpha_threshold: float,
    lookahead_candles: int,
    batch_size: int,
    epochs: int,
    model_save_path: str,
) -> str:
    sequences_path, labels_path = generate_dataset(
        data_dir=data_dir,
        output_dir=output_dir,
        alpha_model_path=alpha_model_path,
        seq_len=seq_len,
        adx_threshold=adx_threshold,
        atr_threshold=atr_threshold,
        alpha_threshold=alpha_threshold,
        lookahead_candles=lookahead_candles,
    )

    targets_path = f"{output_dir.rstrip('/')}/risk_targets.npz"
    generate_labels(labels_npz_path=labels_path, output_path=targets_path)

    ckpt = train(
        TrainConfig(
            seq_path=sequences_path,
            targets_path=targets_path,
            save_path=model_save_path,
            batch_size=batch_size,
            epochs=epochs,
        )
    )
    LOGGER.info("Pipeline complete. Model saved at %s", ckpt)
    return ckpt


def main() -> None:
    _configure_logging()
    parser = argparse.ArgumentParser(description="Run full RiskLayer pipeline: dataset -> labels -> train")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="RiskLayer/data/training_set")
    parser.add_argument("--alpha-model-path", default="Alpha/models/alpha_model.pth")
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--adx-threshold", type=float, default=20.0)
    parser.add_argument(
        "--atr-threshold",
        type=float,
        default=None,
        help="Deprecated alias for --adx-threshold (forwarded for compatibility).",
    )
    parser.add_argument("--alpha-threshold", type=float, default=0.55)
    parser.add_argument("--lookahead-candles", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--model-save-path", default="RiskLayer/models/risk_lstm_multitask.pth")
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        alpha_model_path=args.alpha_model_path,
        seq_len=args.seq_len,
        adx_threshold=args.adx_threshold,
        atr_threshold=args.atr_threshold,
        alpha_threshold=args.alpha_threshold,
        lookahead_candles=args.lookahead_candles,
        batch_size=args.batch_size,
        epochs=args.epochs,
        model_save_path=args.model_save_path,
    )


if __name__ == "__main__":
    main()
