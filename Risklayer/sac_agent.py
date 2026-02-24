import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.amp import autocast, GradScaler
import numpy as np
from typing import Tuple
import logging

from Risklayer.config import config

logger = logging.getLogger(__name__)


class SharedEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, encoder: SharedEncoder, hidden_dim: int, action_dim: int):
        super().__init__()
        self.encoder = encoder
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, state) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, state) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


class Critic(nn.Module):
    def __init__(self, encoder: SharedEncoder, hidden_dim: int, action_dim: int):
        super().__init__()
        self.encoder = encoder
        self.q_linear = nn.Linear(hidden_dim + action_dim, 1)

    def forward(self, state, action):
        x = self.encoder(state)
        xu = torch.cat([x, action], 1)
        return self.q_linear(xu)


class SACAgent:
    def __init__(self):
        if config.USE_GPU and torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU")

        self.shared_encoder = SharedEncoder(config.STATE_DIM, config.HIDDEN_DIM).to(
            self.device
        )

        self.actor = Actor(
            self.shared_encoder, config.HIDDEN_DIM, config.ACTION_DIM
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.LR)

        self.critic1 = Critic(
            self.shared_encoder, config.HIDDEN_DIM, config.ACTION_DIM
        ).to(self.device)
        self.critic2 = Critic(
            self.shared_encoder, config.HIDDEN_DIM, config.ACTION_DIM
        ).to(self.device)
        self.critic1_target = Critic(
            self.shared_encoder, config.HIDDEN_DIM, config.ACTION_DIM
        ).to(self.device)
        self.critic2_target = Critic(
            self.shared_encoder, config.HIDDEN_DIM, config.ACTION_DIM
        ).to(self.device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        critic_params = list(self.critic1.parameters())
        for p in self.critic2.parameters():
            if id(p) not in [id(param) for param in critic_params]:
                critic_params.append(p)

        self.critic_optimizer = torch.optim.Adam(
            critic_params,
            lr=config.LR,
        )

        self.target_entropy = -torch.prod(
            torch.Tensor([config.ACTION_DIM]).to(self.device)
        ).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.LR)

        self.scaler = GradScaler("cuda", enabled=self.device.type == "cuda")

    def select_action(self, state, evaluate=False):
        if isinstance(state, np.ndarray) and len(state.shape) > 1:
            state = torch.FloatTensor(state).to(self.device)
            if evaluate is False:
                action, _, _ = self.actor.sample(state)
            else:
                _, _, action = self.actor.sample(state)
            return action.detach().cpu().numpy()
        else:
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            if evaluate is False:
                action, _, _ = self.actor.sample(state)
            else:
                _, _, action = self.actor.sample(state)
            return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = (
            memory.sample(batch_size=batch_size)
        )

        state_batch = torch.as_tensor(
            state_batch, device=self.device, dtype=torch.float32
        )
        next_state_batch = torch.as_tensor(
            next_state_batch, device=self.device, dtype=torch.float32
        )
        action_batch = torch.as_tensor(
            action_batch, device=self.device, dtype=torch.float32
        )
        reward_batch = torch.as_tensor(
            reward_batch, device=self.device, dtype=torch.float32
        )
        mask_batch = torch.as_tensor(
            mask_batch, device=self.device, dtype=torch.float32
        )

        with autocast("cuda", enabled=self.device.type == "cuda"):
            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = self.actor.sample(
                    next_state_batch
                )
                qf1_next_target = self.critic1_target(
                    next_state_batch, next_state_action
                )
                qf2_next_target = self.critic2_target(
                    next_state_batch, next_state_action
                )
                min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target)
                    - self.alpha * next_state_log_pi
                )
                next_q_value = reward_batch + mask_batch * config.GAMMA * (
                    min_qf_next_target
                )

            qf1 = self.critic1(state_batch, action_batch)
            qf2 = self.critic2(state_batch, action_batch)
            qf1_loss = F.mse_loss(qf1, next_q_value)
            qf2_loss = F.mse_loss(qf2, next_q_value)
            q_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        self.scaler.scale(q_loss).backward()
        self.scaler.unscale_(self.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.critic_optimizer.param_groups[0]["params"], 1.0
        )
        self.scaler.step(self.critic_optimizer)

        with autocast("cuda", enabled=self.device.type == "cuda"):
            pi, log_pi, _ = self.actor.sample(state_batch)
            qf1_pi = self.critic1(state_batch, pi)
            qf2_pi = self.critic2(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        self.scaler.scale(policy_loss).backward()
        self.scaler.unscale_(self.actor_optimizer)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.scaler.step(self.actor_optimizer)

        with autocast("cuda", enabled=self.device.type == "cuda"):
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()

        self.alpha_optimizer.zero_grad()
        self.scaler.scale(alpha_loss).backward()
        self.scaler.step(self.alpha_optimizer)

        self.scaler.update()

        self.alpha = self.log_alpha.exp()

        if updates % 1 == 0:
            self._soft_update(self.critic1_target, self.critic1, config.TAU)
            self._soft_update(self.critic2_target, self.critic2, config.TAU)

        with torch.no_grad():
            explained_var = 1.0 - torch.var(next_q_value - qf1) / (
                torch.var(next_q_value) + 1e-8
            )
            explained_var = explained_var.item() if torch.var(next_q_value) > 0 else 0.0

        metrics = {
            "q1_loss": qf1_loss.item(),
            "q2_loss": qf2_loss.item(),
            "q1_mean": qf1.mean().item(),
            "q2_mean": qf2.mean().item(),
            "target_q_mean": next_q_value.mean().item(),
            "actor_loss": policy_loss.item(),
            "policy_entropy": -log_pi.mean().item(),
            "mean_action": pi.mean().item(),
            "action_std": pi.std().item(),
            "explained_variance": explained_var,
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "actor_lr": self.actor_optimizer.param_groups[0]["lr"],
            "critic_lr": self.critic_optimizer.param_groups[0]["lr"],
            "alpha_lr": self.alpha_optimizer.param_groups[0]["lr"],
        }

        return metrics

    def _soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @alpha.setter
    def alpha(self, value):
        pass  # Handle if needed
