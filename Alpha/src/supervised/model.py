import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Binary Focal Loss.
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # targets should be in [0, 1]
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
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

class MultiHeadModel(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, num_res_blocks=3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim)
        )

        self.trunk = nn.Sequential(*[
            ResidualBlock(hidden_dim) for _ in range(num_res_blocks)
        ])

        # Head A: Direction (Tanh) - Predicts direction in [-1, 1]
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1),
            nn.Tanh()
        )

        # Head B: Quality (Linear) - Predicts score [0, 1]
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1)
        )

        # Head C: Meta (Sigmoid) - Predicts probability of barrier hit [0, 1]
        self.meta_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.input_proj(x)
        feat = self.trunk(feat)

        direction = self.direction_head(feat)
        quality = self.quality_head(feat)
        meta = self.meta_head(feat)

        return direction, quality, meta

def micro_test_model():
    print("\n--- Phase 5 Micro-test (Part 1: Forward Pass) ---")
    model = MultiHeadModel(input_dim=40)
    dummy_input = torch.randn(5, 40)
    dir_out, qual_out, meta_out = model(dummy_input)

    print(f"Direction output shape: {dir_out.shape}, range: [{dir_out.min().item():.2f}, {dir_out.max().item():.2f}]")
    print(f"Quality output shape: {qual_out.shape}")
    print(f"Meta output shape: {meta_out.shape}, range: [{meta_out.min().item():.2f}, {meta_out.max().item():.2f}]")

    assert dir_out.shape == (5, 1)
    assert qual_out.shape == (5, 1)
    assert meta_out.shape == (5, 1)
    print("Forward pass successful.")

if __name__ == "__main__":
    micro_test_model()
