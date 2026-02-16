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
    def __init__(self, input_dim: int = 40, hidden_dim: int = 128):
        super(AlphaSLModel, self).__init__()

        # Shared Trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Head A: Direction (3 classes: -1, 0, 1)
        # The user said "Tanh output (direction)".
        # To reconcile with "Focal Loss", we'll use 3 units.
        # If they literally want a single Tanh, we'd use regression, but Focal Loss is for classification.
        # We will use 3 units and apply Tanh to them? No, Softmax is for classification.
        # Maybe they want Tanh as a squash for a regression-like direction?
        # Let's provide both or stick to classification with a Tanh-like feel if possible.
        # Actually, I'll use 3 units for classification to support Focal Loss.
        self.direction_head = nn.Linear(hidden_dim, 3)

        # Head B: Quality (Regression [0, 1])
        self.quality_head = nn.Linear(hidden_dim, 1)

        # Head C: Meta (Binary [0, 1])
        self.meta_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.trunk(x)

        # Direction: 3 classes
        direction_logits = self.direction_head(features)
        # We'll use the logits for Focal Loss.

        # Quality: Linear output (User said Linear)
        quality = self.quality_head(features)

        # Meta: Sigmoid output (User said Sigmoid)
        meta = torch.sigmoid(self.meta_head(features))

        return direction_logits, quality, meta

def multi_head_loss(outputs, targets, alpha_dir=None):
    """
    Computes weighted multi-head loss.
    outputs: (dir_logits, quality_pred, meta_pred)
    targets: (dir_target, quality_target, meta_target)
    """
    dir_logits, qual_pred, meta_pred = outputs
    dir_target, qual_target, meta_target = targets

    # Direction Loss: Focal Loss
    # Map dir_target from {-1, 0, 1} to {0, 1, 2}
    dir_target_mapped = (dir_target + 1).long()
    focal_criterion = FocalLoss(alpha=alpha_dir)
    loss_dir = focal_criterion(dir_logits, dir_target_mapped)

    # Quality Loss: Huber Loss
    loss_qual = F.huber_loss(qual_pred.squeeze(), qual_target.float())

    # Meta Loss: BCE Loss
    loss_meta = F.binary_cross_entropy(meta_pred.squeeze(), meta_target.float())

    total_loss = loss_dir + loss_qual + loss_meta

    return total_loss, (loss_dir, loss_qual, loss_meta)

if __name__ == "__main__":
    model = AlphaSLModel(input_dim=40)
    x = torch.randn(16, 40)
    dir_logits, qual, meta = model(x)
    print(f"Direction logits shape: {dir_logits.shape}")
    print(f"Quality shape: {qual.shape}")
    print(f"Meta shape: {meta.shape}")
