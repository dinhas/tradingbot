import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    """
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
    def __init__(self, input_dim: int = 40, hidden_dim: int = 256, num_res_blocks: int = 4, head_a_type: str = 'tanh'):
        super(AlphaSLModel, self).__init__()
        self.head_a_type = head_a_type

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

        # Head A: Direction
        if head_a_type == 'tanh':
            self.direction_head = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Tanh()
            )
        else: # 'classes'
            self.direction_head = nn.Linear(hidden_dim, 3)

        # Head B: Quality (Regression [0, 1])
        self.quality_head = nn.Linear(hidden_dim, 1)

        # Head C: Meta (Probabilistic [0, 1])
        self.meta_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.input_proj(x)
        features = self.backbone(features)

        direction = self.direction_head(features)
        quality = self.quality_head(features)
        meta = self.meta_head(features)

        return direction, quality, meta

def multi_head_loss(outputs, targets, weights=(1.0, 1.0, 1.0), alpha_dir=None, loss_types=('mse', 'huber', 'bce')):
    """
    Computes weighted multi-head loss.
    outputs: (dir_out, quality_pred, meta_prob)
    targets: (dir_target, quality_target, meta_target)
    weights: (w_dir, w_qual, w_meta)
    loss_types: ('mse' or 'focal' or 'ce', 'huber', 'bce')
    """
    dir_out, qual_pred, meta_prob = outputs
    dir_target, qual_target, meta_target = targets
    w_dir, w_qual, w_meta = weights
    dir_loss_type, qual_loss_type, meta_loss_type = loss_types

    # Direction Loss
    if dir_loss_type == 'focal':
        # Map dir_target from {-1, 0, 1} to {0, 1, 2}
        dir_target_mapped = (dir_target + 1).long()
        focal_criterion = FocalLoss(alpha=alpha_dir)
        loss_dir = focal_criterion(dir_out, dir_target_mapped)
    elif dir_loss_type == 'ce':
        dir_target_mapped = (dir_target + 1).long()
        loss_dir = F.cross_entropy(dir_out, dir_target_mapped, weight=alpha_dir)
    else: # Default to MSE for Tanh output
        loss_dir = F.mse_loss(dir_out.squeeze(), dir_target.float())

    # Quality Loss: Huber Loss
    loss_qual = F.huber_loss(qual_pred.squeeze(), qual_target.float())

    # Meta Loss: BCE
    # Using meta_prob directly with binary_cross_entropy since meta_head has Sigmoid
    loss_meta = F.binary_cross_entropy(meta_prob.squeeze(), meta_target.float())

    total_loss = (w_dir * loss_dir) + (w_qual * loss_qual) + (w_meta * loss_meta)

    return total_loss, (loss_dir, loss_qual, loss_meta)

if __name__ == "__main__":
    model = AlphaSLModel(input_dim=40)
    x = torch.randn(16, 40)
    direction, qual, meta = model(x)
    print(f"Direction shape: {direction.shape}")
    print(f"Quality shape: {qual.shape}")
    print(f"Meta shape: {meta.shape}")
