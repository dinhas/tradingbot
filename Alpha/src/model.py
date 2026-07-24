import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaSLModel(nn.Module):
    def __init__(self, input_dim: int = 14, lstm_units: int = 64, dense_units: int = 32,
                 dropout: float = 0.3, num_assets: int = 4, asset_embedding_dim: int = 4):
        super(AlphaSLModel, self).__init__()

        # 1. LSTM layer (captures temporal patterns)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # 2. Attention Layer to learn which timesteps matter most
        self.attention_weights = nn.Linear(lstm_units, 1)

        # 3. Small dense layer with dropout
        self.asset_embedding = nn.Embedding(num_assets, asset_embedding_dim)
        self.fc1 = nn.Linear(lstm_units + asset_embedding_dim, dense_units)
        self.dropout = nn.Dropout(dropout)

        # One executable opportunity logit for each action: [short, long].
        self.action_head = nn.Linear(dense_units, 2)

    def forward(self, x, asset_ids=None, return_dict: bool = False):
        # x shape: (batch, seq_len, input_dim)
        
        # LSTM output
        lstm_out, _ = self.lstm(x)
        
        # Attention Mechanism instead of taking just the last output bar
        attn_scores = self.attention_weights(lstm_out) # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1) # (batch, hidden_dim)
        
        if asset_ids is None:
            asset_ids = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        asset_context = self.asset_embedding(asset_ids.long())

        # Dense + Dropout
        x = F.relu(self.fc1(torch.cat([context_vector, asset_context], dim=1)))
        x = self.dropout(x)
        action_logits = self.action_head(x)

        if return_dict:
            return {"action_logits": action_logits}

        return action_logits


def action_opportunity_loss(outputs, action_targets, action_pos_weight=None):
    """Binary loss for independent executable short and long opportunities."""
    logits = outputs["action_logits"]
    targets = action_targets.float()
    pos_weight = None
    if action_pos_weight is not None:
        pos_weight = torch.as_tensor(action_pos_weight, device=logits.device, dtype=logits.dtype)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, targets)
