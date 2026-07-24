import torch
import torch.nn as nn
import torch.nn.functional as F


HOLD, SHORT, LONG = 0, 1, 2
NUM_CLASSES = 3


class AlphaSLModel(nn.Module):
    def __init__(self, input_dim: int = 14, lstm_units: int = 128, dense_units: int = 64,
                 dropout: float = 0.3, num_assets: int = 4, asset_embedding_dim: int = 4):
        super(AlphaSLModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_units,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout=dropout,
        )

        self.attention_weights = nn.Linear(lstm_units, 1)

        self.asset_embedding = nn.Embedding(num_assets, asset_embedding_dim)
        self.fc1 = nn.Linear(lstm_units + asset_embedding_dim, dense_units)
        self.dropout = nn.Dropout(dropout)

        self.action_head = nn.Linear(dense_units, NUM_CLASSES)

    def forward(self, x, asset_ids=None, return_dict: bool = False):
        lstm_out, _ = self.lstm(x)

        attn_scores = self.attention_weights(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)

        if asset_ids is None:
            asset_ids = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        asset_context = self.asset_embedding(asset_ids.long())

        x = F.relu(self.fc1(torch.cat([context_vector, asset_context], dim=1)))
        x = self.dropout(x)
        action_logits = self.action_head(x)

        if return_dict:
            return {"action_logits": action_logits}
        return action_logits


def action_opportunity_loss(outputs, action_targets, class_weights=None,
                            label_smoothing: float = 0.05):
    """3-class cross-entropy loss (hold / short / long).

    Args:
        outputs: dict with "action_logits" of shape (B, 3).
        action_targets: class indices of shape (B,) with values in {0, 1, 2}.
        class_weights: optional tensor of shape (3,) for imbalanced classes.
        label_smoothing: soft-label smoothing factor.
    """
    logits = outputs["action_logits"]
    targets = action_targets.long()
    weight = None
    if class_weights is not None:
        weight = torch.as_tensor(class_weights, device=logits.device, dtype=logits.dtype)
    return F.cross_entropy(logits, targets, weight=weight, label_smoothing=label_smoothing)
