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

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x
        out = self.ln1(x)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return residual + out

class AlphaSLModel(nn.Module):
    def __init__(self, input_dim: int = 40, hidden_dim: int = 256, num_res_blocks: int = 4):
        super(AlphaSLModel, self).__init__()

        # Initial Projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim)
        )

        # Residual Backbone
        self.backbone = nn.Sequential(*[
            ResidualBlock(hidden_dim) for _ in range(num_res_blocks)
        ])

        # Head A: Direction (Continuous -1 to 1)
        self.direction_head = nn.Linear(hidden_dim, 1)

        # Head B: Quality (Regression [0, 1])
        self.quality_head = nn.Linear(hidden_dim, 1)

        # Head C: Meta (Binary [0, 1])
        self.meta_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.input_proj(x)
        features = self.backbone(features)

        # Direction: Tanh score (-1 to 1)
        direction_score = torch.tanh(self.direction_head(features))

        # Quality: Linear output
        quality = self.quality_head(features)

        # Meta: Raw Logits (For BCEWithLogitsLoss stability)
        meta_logits = self.meta_head(features)

        return direction_score, quality, meta_logits

def multi_head_loss(outputs, targets, weights=(2.0, 0.5, 1.0), alpha_dir=None):
    """
    Computes weighted multi-head loss.
    outputs: (dir_score, quality_pred, meta_logits)
    targets: (dir_target, quality_target, meta_target)
    """
    dir_score, qual_pred, meta_logits = outputs
    dir_target, qual_target, meta_target = targets
    w_dir, w_qual, w_meta = weights

    # Direction Loss: MSE Loss (for continuous -1 to 1)
    loss_dir = F.mse_loss(dir_score.squeeze(), dir_target.float())

    # Quality Loss: Huber Loss
    loss_qual = F.huber_loss(qual_pred.squeeze(), qual_target.float())

    # Meta Loss: BCE with Logits (Safe for AMP/Autocast)
    loss_meta = F.binary_cross_entropy_with_logits(meta_logits.squeeze(), meta_target.float())

    total_loss = (w_dir * loss_dir) + (w_qual * loss_qual) + (w_meta * loss_meta)

    return total_loss, (loss_dir, loss_qual, loss_meta)

if __name__ == "__main__":
    model = AlphaSLModel(input_dim=40)
    x = torch.randn(16, 40)
    dir_logits, qual, meta = model(x)
    print(f"Direction logits shape: {dir_logits.shape}")
    print(f"Quality shape: {qual.shape}")
    print(f"Meta shape: {meta.shape}")
