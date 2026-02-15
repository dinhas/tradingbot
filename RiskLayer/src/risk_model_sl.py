import torch
import torch.nn as nn
import torch.nn.functional as F

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

class RiskModelSL(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, num_res_blocks=3):
        super(RiskModelSL, self).__init__()
        
        # Initial Projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim)
        )
        
        # Shared Residual Backbone
        self.backbone = nn.Sequential(*[
            ResidualBlock(hidden_dim) for _ in range(num_res_blocks)
        ])
        
        # --- Task Specific Heads ---
        
        # SL Head: Predicts multiplier (e.g., 0.2 to 5.0)
        self.sl_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Softplus() # Ensures positive output
        )
        
        # TP Head: Predicts multiplier (e.g., 0.1 to 10.0)
        self.tp_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Softplus()
        )
        
        # Size Head: Predicts confidence/size (0.0 to 1.0)
        self.size_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid() # Constrains to [0, 1]
        )

    def forward(self, x):
        features = self.input_proj(x)
        features = self.backbone(features)
        
        sl = self.sl_head(features)
        tp = self.tp_head(features)
        size = self.size_head(features)
        
        return {
            'sl': sl,
            'tp': tp,
            'size': size
        }
