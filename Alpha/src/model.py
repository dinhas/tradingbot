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
    def __init__(self, input_dim: int = 17, lstm_units: int = 64, dense_units: int = 32, dropout: float = 0.3):
        super(AlphaSLModel, self).__init__()

        # 1. LSTM layer (captures temporal patterns)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # 2. Small dense layer with dropout
        self.fc1 = nn.Linear(lstm_units, dense_units)
        self.dropout = nn.Dropout(dropout)

        # 3. Output head for classification (3 classes: Sell, Neutral, Buy)
        self.fc_out = nn.Linear(dense_units, 3)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        
        # LSTM output
        lstm_out, _ = self.lstm(x)
        
        # Take the output of the last time step (return_sequences=False behavior)
        last_out = lstm_out[:, -1, :]
        
        # Dense + Dropout
        x = F.relu(self.fc1(last_out))
        x = self.dropout(x)
        
        # Output Logits
        logits = self.fc_out(x)
        
        return logits

def direction_loss(logits, target, alpha_dir=None):
    """
    Computes Focal Loss for the directional classification.
    Maps targets [-1, 0, 1] to class indices [0, 1, 2].
    """
    target_mapped = (target + 1).long()
    focal_criterion = FocalLoss(alpha=alpha_dir)
    return focal_criterion(logits, target_mapped)
