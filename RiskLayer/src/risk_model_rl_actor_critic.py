import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

SL_CHOICES = [0.5, 0.75, 1.0, 1.5, 1.75, 2.0, 2.5, 2.75, 3.0]
NUM_SL_CHOICES = len(SL_CHOICES)
TP_MIN, TP_MAX = 1.0, 5.0
SIZE_MIN, SIZE_MAX = 0.1, 1.0


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


class RiskModelRLActorCritic(nn.Module):
    """
    True RL Actor-Critic network for risk management.

    Actor: Outputs action distribution (policy)
    Critic: Estimates state value function
    """

    def __init__(self, input_dim=40, hidden_dim=256, num_res_blocks=3):
        super(RiskModelRLActorCritic, self).__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim),
        )

        self.backbone = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_res_blocks)]
        )

        # === ACTOR (Policy) Heads ===
        # SL: Discrete choice -> categorical distribution
        self.sl_policy = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, NUM_SL_CHOICES),
        )

        # TP: Continuous -> Beta distribution params
        self.tp_mu = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # 0 to 1, will scale to [TP_MIN, TP_MAX]
        )
        self.tp_sigma = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Softplus(),  # Positive
        )

        # Size: Continuous -> Beta distribution params
        self.size_mu = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.size_sigma = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Softplus(),
        )

        # === CRITIC (Value) Head ===
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        features = self.input_proj(x)
        features = self.backbone(features)

        # Policy distributions
        sl_logits = self.sl_policy(features)

        tp_mu = self.tp_mu(features)
        tp_sigma = self.tp_sigma(features)

        size_mu = self.size_mu(features)
        size_sigma = self.size_sigma(features)

        # Value estimate
        value = self.value_head(features)

        return {
            "sl_logits": sl_logits,
            "sl_probs": F.softmax(sl_logits, dim=-1),
            "tp_mu": tp_mu,
            "tp_sigma": tp_sigma,
            "size_mu": size_mu,
            "size_sigma": size_sigma,
            "value": value,
        }

    def get_policy(self, x):
        """Get policy distribution for actions."""
        outputs = self.forward(x)

        return {
            "sl_logits": outputs["sl_logits"],
            "sl_probs": outputs["sl_probs"],
            "tp_mu": outputs["tp_mu"],
            "tp_sigma": outputs["tp_sigma"],
            "size_mu": outputs["size_mu"],
            "size_sigma": outputs["size_sigma"],
        }

    def get_value(self, x):
        """Get state value estimate."""
        outputs = self.forward(x)
        return outputs["value"]

    def get_action(self, x, deterministic=False):
        """
        Sample action from policy.
        Returns action dict and log probabilities for PPO loss.
        """
        outputs = self.forward(x)

        batch_size = x.shape[0]

        # SL: Categorical sampling
        if deterministic:
            sl_idx = torch.argmax(outputs["sl_probs"], dim=-1)
        else:
            sl_idx = torch.multinomial(outputs["sl_probs"], num_samples=1).squeeze(-1)

        # Get log prob for SL
        sl_log_probs = F.log_softmax(outputs["sl_logits"], dim=-1)
        sl_log_prob = sl_log_probs.gather(1, sl_idx.unsqueeze(1)).squeeze(1)

        # TP: Beta distribution sampling
        tp_mu = outputs["tp_mu"].squeeze(-1)
        tp_sigma = outputs["tp_sigma"].squeeze(-1)

        if deterministic:
            tp_mult = tp_mu
            tp_log_prob = torch.zeros(batch_size, device=x.device)
        else:
            tp_alpha = tp_mu / (tp_sigma + 1e-6)
            tp_beta = (1 - tp_mu) / (tp_sigma + 1e-6)
            tp_beta_dist = torch.distributions.Beta(tp_alpha, tp_beta)
            tp_sample = tp_beta_dist.sample()
            tp_mult = TP_MIN + tp_sample * (TP_MAX - TP_MIN)
            tp_log_prob = tp_beta_dist.log_prob(tp_sample)

        # Size: Beta distribution sampling
        size_mu = outputs["size_mu"].squeeze(-1)
        size_sigma = outputs["size_sigma"].squeeze(-1)

        if deterministic:
            size = size_mu
            size_log_prob = torch.zeros(batch_size, device=x.device)
        else:
            size_alpha = size_mu / (size_sigma + 1e-6)
            size_beta = (1 - size_mu) / (size_sigma + 1e-6)
            size_beta_dist = torch.distributions.Beta(size_alpha, size_beta)
            size_sample = size_beta_dist.sample()
            size = SIZE_MIN + size_sample * (SIZE_MAX - SIZE_MIN)
            size_log_prob = size_beta_dist.log_prob(size_sample)

        # Total log prob (sum of all action components)
        total_log_prob = (
            sl_log_prob + tp_log_prob.squeeze(-1) + size_log_prob.squeeze(-1)
        )

        # Get SL multiplier value
        sl_mult = torch.tensor(
            [SL_CHOICES[i] for i in sl_idx.cpu().numpy()],
            device=x.device,
            dtype=torch.float32,
        )

        return {
            "sl_idx": sl_idx,
            "sl_mult": sl_mult,
            "tp_mult": tp_mult,
            "size": size,
            "sl_log_prob": sl_log_prob,
            "tp_log_prob": tp_log_prob.squeeze(-1),
            "size_log_prob": size_log_prob.squeeze(-1),
            "total_log_prob": total_log_prob,
            "value": outputs["value"].squeeze(-1),
        }
