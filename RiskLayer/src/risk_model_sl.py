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
    def __init__(self, input_dim=48, hidden_dim=256, num_res_blocks=4):
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
        
        # --- Task Specific Heads (6 TOTAL) ---
        
        # 1. SL Head: Predicts multiplier
        self.sl_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Softplus() 
        )
        
        # 2. TP Head: Predicts multiplier
        self.tp_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Softplus()
        )
        
        # 3. Size Head: Predicts position size factor [0, 1]
        self.size_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid() 
        )
        
        # 4. Prob TP First Head: Binary prediction (logits)
        self.prob_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1)
        )
        
        # 5. Execution Buffer Head: Predicts (spread + slippage) / ATR
        self.exec_buffer_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Softplus()
        )
        
        # 6. Expected Value Head: Predicts net ATR-normalized EV
        self.ev_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.input_proj(x)
        features = self.backbone(features)
        
        return {
            'sl_mult': self.sl_head(features),
            'tp_mult': self.tp_head(features),
            'size': self.size_head(features),
            'prob_tp_first_logits': self.prob_head(features),
            'execution_buffer': self.exec_buffer_head(features),
            'expected_value': self.ev_head(features)
        }
