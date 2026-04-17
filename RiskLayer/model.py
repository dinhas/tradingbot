"""LSTM multi-head regression model for SL/TP/quality prediction."""

from __future__ import annotations

import torch
from torch import nn


class MultiHeadRiskLSTM(nn.Module):
    """Shared LSTM backbone with three regression heads.

    Heads:
      1) SL ATR multiplier
      2) TP ATR multiplier
      3) Trade quality score in [0, 1]
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_size < 1:
            raise ValueError("input_size must be >= 1")

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.backbone = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)

        self.sl_head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1))
        self.tp_head = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1))
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: shape [batch, sequence_length, features]
        Returns:
            Tuple (sl, tp, quality) each with shape [batch].
        """
        output, _ = self.backbone(x)
        final_state = self.norm(output[:, -1, :])

        sl = self.sl_head(final_state).squeeze(-1)
        tp = self.tp_head(final_state).squeeze(-1)
        quality = self.quality_head(final_state).squeeze(-1)
        return sl, tp, quality
