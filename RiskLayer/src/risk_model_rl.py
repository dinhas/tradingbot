import torch
import torch.nn as nn
import torch.nn.functional as F

SL_CHOICES = [0.5, 0.75, 1.0, 1.5, 1.75, 2.0, 2.5, 2.75, 3.0]
NUM_SL_CHOICES = len(SL_CHOICES)


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
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


class RiskModelRL(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, num_res_blocks=3):
        super(RiskModelRL, self).__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim),
        )

        self.backbone = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_res_blocks)]
        )

        self.sl_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, NUM_SL_CHOICES),
        )

        self.tp_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.size_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.input_proj(x)
        features = self.backbone(features)

        sl_logits = self.sl_head(features)
        tp_raw = self.tp_head(features)
        size_raw = self.size_head(features)

        tp = 1.0 + tp_raw * 4.0
        size = 0.1 + size_raw * 0.9

        return {
            "sl_logits": sl_logits,
            "sl_probs": F.softmax(sl_logits, dim=-1),
            "tp": tp,
            "size": size,
        }

    def get_action(self, x, deterministic=False):
        outputs = self.forward(x)

        if deterministic:
            sl_idx = torch.argmax(outputs["sl_probs"], dim=-1)
        else:
            sl_idx = torch.multinomial(outputs["sl_probs"], num_samples=1).squeeze(-1)

        return {
            "sl_idx": sl_idx,
            "sl_mult": torch.tensor(
                [SL_CHOICES[i] for i in sl_idx.cpu().numpy()], device=x.device
            ),
            "tp_mult": outputs["tp"].squeeze(-1),
            "size": outputs["size"].squeeze(-1),
        }
