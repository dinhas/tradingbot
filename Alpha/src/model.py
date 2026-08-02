import torch
import torch.nn as nn
import torch.nn.functional as F


# Binary labels: hold=0, buy=1
HOLD, BUY = 0, 1
NUM_CLASSES = 1


class AlphaSLModel(nn.Module):
    def __init__(self, input_dim: int = 14, lstm_units: int = 128, dense_units: int = 128,
                 dropout: float = 0.3, num_assets: int = 4, asset_embedding_dim: int = 4,
                 num_layers: int = 3, num_heads: int = 4, bidirectional: bool = True):
        super(AlphaSLModel, self).__init__()

        self.num_heads = num_heads
        lstm_output_dim = lstm_units * (2 if bidirectional else 1)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_units,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.lstm_norm = nn.LayerNorm(lstm_output_dim)

        self.attention_heads = nn.ModuleList([
            nn.Linear(lstm_output_dim, 1) for _ in range(num_heads)
        ])
        self.attn_proj = nn.Linear(num_heads * lstm_output_dim, lstm_output_dim)

        self.asset_embedding = nn.Embedding(num_assets, asset_embedding_dim)
        self.fc1 = nn.Linear(lstm_output_dim + asset_embedding_dim, dense_units)
        self.fc2 = nn.Linear(dense_units, dense_units)
        self.dropout = nn.Dropout(dropout)

        self.action_head = nn.Linear(dense_units, 1)  # binary: hold=0, buy=1

    def forward(self, x, asset_ids=None, return_dict: bool = False):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_norm(lstm_out)

        head_outputs = []
        for head_attn in self.attention_heads:
            attn_scores = head_attn(lstm_out)  # (B, T, 1)
            attn_weights = torch.softmax(attn_scores, dim=1)
            context = torch.sum(attn_weights * lstm_out, dim=1)  # (B, D)
            head_outputs.append(context)
        multi_head_context = torch.cat(head_outputs, dim=1)  # (B, num_heads * D)
        context_vector = self.attn_proj(multi_head_context)

        if asset_ids is None:
            asset_ids = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        asset_context = self.asset_embedding(asset_ids.long())

        x = F.relu(self.fc1(torch.cat([context_vector, asset_context], dim=1)))
        x = self.dropout(x)
        residual = x
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = x + residual
        action_logits = self.action_head(x)

        if return_dict:
            return {"action_logits": action_logits}
        return action_logits


def binary_signal_loss(outputs, action_targets, label_smoothing: float = 0.05,
                       class_weights: torch.Tensor | None = None):
    """BCE loss for binary hold=0 / signal=1.

    class_weights: optional tensor. Accepts either:
      - 1-element tensor: used directly as pos_weight
      - 2-element tensor: [weight_hold, weight_signal], converted to pos_weight
    """
    logits = outputs["action_logits"].squeeze(-1)
    targets = action_targets.float()
    if label_smoothing > 0:
        targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing

    pos_weight = None
    if class_weights is not None:
        if class_weights.numel() == 1:
            pos_weight = class_weights
        elif class_weights.numel() == 2:
            pos_weight = torch.tensor([class_weights[1] / max(class_weights[0], 1e-8)],
                                      device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


# =============================================================================
# V7 Architecture: GRN blocks + Scaled Dot-Product Attention + VSN + Aux Head
# =============================================================================

class _GatedLinearUnit(nn.Module):
    """GLU(x) = sigmoid(Linear(x)) * Linear(x)."""
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, x):
        return torch.sigmoid(self.gate(x)) * self.proj(x)


class _GatedResidualNetwork(nn.Module):
    """GRN from Temporal Fusion Transformer.

    GRN(x) = LayerNorm(x + GLU(ELU(Linear(x))))
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.glu = _GatedLinearUnit(output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.skip_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        residual = self.skip_proj(x)
        h = self.fc1(x)
        h = self.elu(h)
        h = self.fc2(h)
        h = self.glu(h)
        h = self.dropout(h)
        return self.layer_norm(h + residual)


class _VariableSelectionNetwork(nn.Module):
    """VSN: learns per-timestep feature gating.

    Vectorized: applies a single shared GRN to weighted-sum of features,
    using per-feature linear projections + softmax gating weights.
    """
    def __init__(self, n_features, hidden_dim, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        # Per-feature projection: (n_feat) -> (n_feat * hidden_dim) via shared weight
        self.feat_proj = nn.Linear(n_features, n_features * hidden_dim)
        self.hidden_dim = hidden_dim
        # Weight GRN: computes softmax importance per feature
        self.weight_grn = _GatedResidualNetwork(n_features, hidden_dim, n_features, dropout)
        # Output GRN: processes the weighted feature sum
        self.out_grn = _GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)

    def forward(self, x):
        # x: (B, T, n_features)
        B, T, n_feat = x.shape
        h = self.hidden_dim

        # Per-feature linear projection: (B, T, n_feat) -> (B, T, n_feat * hidden)
        projected = self.feat_proj(x)                 # (B, T, n_feat * h)
        projected = projected.reshape(B, T, n_feat, h)  # (B, T, n_feat, h)

        # Softmax over features for gating
        weights = self.weight_grn(x)                  # (B, T, n_feat)
        weights = F.softmax(weights, dim=-1)          # (B, T, n_feat)
        weights = weights.unsqueeze(-1)               # (B, T, n_feat, 1)

        # Weighted sum across features: (B, T, n_feat, h) * (B, T, n_feat, 1) -> sum -> (B, T, h)
        selected = (projected * weights).sum(dim=2)   # (B, T, h)
        selected = self.out_grn(selected)             # (B, T, h)

        return selected, weights.squeeze(-1)          # (B, T, h), (B, T, n_feat)


class _ScaledDotProductAttention(nn.Module):
    """Multi-head scaled dot-product attention with Q/K/V projections."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** 0.5

    def forward(self, x):
        B, T, D = x.shape
        H, DH = self.n_heads, self.d_head

        Q = self.q_proj(x).view(B, T, H, DH).transpose(1, 2)  # (B, H, T, DH)
        K = self.k_proj(x).view(B, T, H, DH).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, DH).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, T, T)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)  # (B, H, T, DH)
        context = context.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)
        output = self.out_proj(context)
        return output, attn_weights


class AlphaSLModelV7(nn.Module):
    """V7 Alpha model with:
      - Variable Selection Network at input (learned feature gating)
      - Scaled Dot-Product Multi-Head Attention (Q/K/V + temperature)
      - Gated Residual Network blocks (replacing plain FC)
      - Learnable positional encoding
      - Auxiliary regime head (multi-task loss)
    """
    def __init__(self, input_dim: int = 31, lstm_units: int = 128, dense_units: int = 128,
                 dropout: float = 0.3, num_assets: int = 4, asset_embedding_dim: int = 4,
                 num_layers: int = 3, num_heads: int = 4, bidirectional: bool = True):
        super(AlphaSLModelV7, self).__init__()

        lstm_output_dim = lstm_units * (2 if bidirectional else 1)

        # 1. Variable Selection Network (feature gating)
        self.vsn = _VariableSelectionNetwork(input_dim, hidden_dim=128, dropout=dropout)

        # 2. LSTM backbone
        self.lstm = nn.LSTM(
            input_size=128,  # VSN output dim
            hidden_size=lstm_units,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.lstm_norm = nn.LayerNorm(lstm_output_dim)

        # 3. Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 256, lstm_output_dim) * 0.02)
        self.pos_proj = nn.Linear(lstm_output_dim, lstm_output_dim)

        # 4. Scaled Dot-Product Multi-Head Attention
        self.attention = _ScaledDotProductAttention(lstm_output_dim, num_heads, dropout=dropout)
        self.attn_layer_norm = nn.LayerNorm(lstm_output_dim)

        # 5. GRN blocks (replacing plain FC layers)
        self.grn1 = _GatedResidualNetwork(lstm_output_dim + asset_embedding_dim,
                                           hidden_dim=dense_units,
                                           output_dim=dense_units,
                                           dropout=dropout)
        self.grn2 = _GatedResidualNetwork(dense_units, hidden_dim=dense_units,
                                           output_dim=dense_units,
                                           dropout=dropout)

        # 6. Output head
        self.action_head = nn.Linear(dense_units, 1)  # binary: hold=0, buy=1
        self.asset_embedding = nn.Embedding(num_assets, asset_embedding_dim)

    def forward(self, x, asset_ids=None, return_dict: bool = False):
        # Variable Selection: learn which features matter
        vsn_out, vsn_weights = self.vsn(x)  # (B, T, 64), (B, T, 31)

        # LSTM
        lstm_out, _ = self.lstm(vsn_out)
        lstm_out = self.lstm_norm(lstm_out)

        # Add positional encoding
        seq_len = lstm_out.shape[1]
        pos = self.pos_proj(self.pos_encoding[:, :seq_len, :])
        lstm_out = lstm_out + pos

        # Scaled Dot-Product Attention (skip attn_weights if not needed)
        attn_out, attn_weights = self.attention(lstm_out)
        attn_out = self.attn_layer_norm(attn_out + lstm_out)  # residual
        context = attn_out[:, -1, :]  # last timestep

        # Asset embedding
        if asset_ids is None:
            asset_ids = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        asset_context = self.asset_embedding(asset_ids.long())

        # GRN blocks
        x = self.grn1(torch.cat([context, asset_context], dim=1))
        x = self.grn2(x)

        # Output
        action_logits = self.action_head(x)

        if return_dict:
            return {
                "action_logits": action_logits,
                "vsn_weights": vsn_weights,
            }
        return action_logits



