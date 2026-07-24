import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha 
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AlphaSLModel(nn.Module):
    def __init__(self, input_dim: int = 14, lstm_units: int = 64, dense_units: int = 32, dropout: float = 0.3):
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
        self.fc1 = nn.Linear(lstm_units, dense_units)
        self.dropout = nn.Dropout(dropout)

        # 4. Output heads: tradeability gate + conditional direction
        self.trade_head = nn.Linear(dense_units, 1)
        self.direction_head = nn.Linear(dense_units, 2)

    def forward(self, x, return_dict: bool = False):
        # x shape: (batch, seq_len, input_dim)
        
        # LSTM output
        lstm_out, _ = self.lstm(x)
        
        # Attention Mechanism instead of taking just the last output bar
        attn_scores = self.attention_weights(lstm_out) # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1) # (batch, hidden_dim)
        
        # Dense + Dropout
        x = F.relu(self.fc1(context_vector))
        x = self.dropout(x)
        
        trade_logit = self.trade_head(x).squeeze(-1)
        direction_logits = self.direction_head(x)

        if return_dict:
            return {
                "trade_logit": trade_logit,
                "direction_logits": direction_logits,
            }

        # Backward-compatible default: return direction logits for legacy callers.
        return direction_logits


def trade_direction_loss(outputs, trade_target, direction_target, trade_weight: float = 0.65,
                         direction_weight: float = 0.35, trade_pos_weight=None,
                         direction_class_weights=None):
    """
    Two-head loss:
    - tradeability: binary BCEWithLogitsLoss
    - direction: masked CE over tradeable samples only
    """
    trade_logit = outputs["trade_logit"]
    direction_logits = outputs["direction_logits"]

    trade_target = trade_target.float()
    direction_target = direction_target.long()

    if trade_pos_weight is not None:
        trade_pos_weight = torch.as_tensor(trade_pos_weight, device=trade_logit.device, dtype=torch.float32)
        trade_loss_fn = nn.BCEWithLogitsLoss(pos_weight=trade_pos_weight)
    else:
        trade_loss_fn = nn.BCEWithLogitsLoss()

    trade_loss = trade_loss_fn(trade_logit, trade_target)

    trade_mask = trade_target > 0.5
    if trade_mask.any():
        if direction_class_weights is not None:
            direction_class_weights = torch.as_tensor(direction_class_weights, device=direction_logits.device, dtype=torch.float32)
        direction_loss_fn = nn.CrossEntropyLoss(weight=direction_class_weights)
        direction_loss = direction_loss_fn(direction_logits[trade_mask], direction_target[trade_mask])
    else:
        direction_loss = trade_loss.new_tensor(0.0)

    return (trade_weight * trade_loss) + (direction_weight * direction_loss)
