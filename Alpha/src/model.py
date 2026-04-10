import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha # Can be a tensor of weights for each class
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [N, C], targets: [N]
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
    def __init__(self, input_dim: int = 40, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.2):
        super(AlphaSLModel, self).__init__()

        # Per-timestep input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 2-layer LSTM backbone
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )

        # Post-LSTM normalization
        self.ln = nn.LayerNorm(hidden_dim)

        # Head A: Direction (3 classes)
        self.direction_head = nn.Linear(hidden_dim, 3)

        # Head B: Quality (regression, 0-1)
        self.quality_head = nn.Linear(hidden_dim, 1)

        # Head C: Meta (binary classification, raw logits)
        self.meta_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = torch.relu(x)

        # LSTM output
        lstm_out, _ = self.lstm(x)
        features = lstm_out[:, -1, :] # Last timestep
        features = self.ln(features)

        direction_logits = self.direction_head(features)
        quality = self.quality_head(features)
        meta_logits = self.meta_head(features)

        return direction_logits, quality, meta_logits

def multi_head_loss(outputs, targets, weights=(2.0, 0.5, 1.0), alpha_dir=None):
    """
    Computes weighted multi-head loss.
    """
    dir_logits, qual_pred, meta_logits = outputs
    dir_target, qual_target, meta_target = targets
    w_dir, w_qual, w_meta = weights

    dir_target_mapped = (dir_target + 1).long()
    focal_criterion = FocalLoss(alpha=alpha_dir)
    loss_dir = focal_criterion(dir_logits, dir_target_mapped)

    loss_qual = F.huber_loss(qual_pred.squeeze(), qual_target.float())
    loss_meta = F.binary_cross_entropy_with_logits(meta_logits.squeeze(), meta_target.float())

    total_loss = (w_dir * loss_dir) + (w_qual * loss_qual) + (w_meta * loss_meta)

    return total_loss, (loss_dir, loss_qual, loss_meta)

if __name__ == "__main__":
    model = AlphaSLModel(input_dim=40, hidden_dim=256, num_layers=2)
    x = torch.randn(16, 50, 40)
    dir_logits, qual, meta = model(x)
    print(f"Direction logits shape: {dir_logits.shape}")
    print(f"Quality shape: {qual.shape}")
    print(f"Meta shape: {meta.shape}")
