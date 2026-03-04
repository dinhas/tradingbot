import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple, Dict
import logging

from Risklayer.config import config

logger = logging.getLogger(__name__)

class SharedEncoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.output_dim = 64

    def forward(self, x):
        return self.net(x)

class Actor(nn.Module):
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.encoder = SharedEncoder(input_dim)
        hidden_dim = self.encoder.output_dim
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
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

class Critic(nn.Module):
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.encoder = SharedEncoder(input_dim)
        hidden_dim = self.encoder.output_dim
        self.q_linear = nn.Linear(hidden_dim + action_dim, 1)

    def forward(self, state, action):
        x = self.encoder(state)
        xu = torch.cat([x, action], 1)
        return self.q_linear(xu)

class SACAgent:
    def __init__(self):
        self.device = torch.device("cpu")
        logger.info("Using optimized CPU agent")

        self.actor = Actor(config.STATE_DIM, config.ACTION_DIM).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

        self.critic1 = Critic(config.STATE_DIM, config.ACTION_DIM).to(self.device)
        self.critic2 = Critic(config.STATE_DIM, config.ACTION_DIM).to(self.device)
        self.critic1_target = Critic(config.STATE_DIM, config.ACTION_DIM).to(self.device)
        self.critic2_target = Critic(config.STATE_DIM, config.ACTION_DIM).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=config.LR, weight_decay=config.WEIGHT_DECAY)

        self.target_entropy = -config.ACTION_DIM
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.LR)
        self._alpha = self.log_alpha.exp().detach()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state):
        with torch.no_grad():
            state_t = torch.as_tensor(state, device=self.device, dtype=torch.float32)
            if len(state_t.shape) == 1: state_t = state_t.unsqueeze(0)
            action, _, _ = self.actor.sample(state_t)
            return action.cpu().numpy()

    def update_parameters(self, memory, batch_size, updates) -> Dict:
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size)

        state_batch = torch.as_tensor(state_batch, device=self.device, dtype=torch.float32)
        next_state_batch = torch.as_tensor(next_state_batch, device=self.device, dtype=torch.float32)
        action_batch = torch.as_tensor(action_batch, device=self.device, dtype=torch.float32)
        reward_batch = torch.as_tensor(reward_batch, device=self.device, dtype=torch.float32)
        mask_batch = torch.as_tensor(mask_batch, device=self.device, dtype=torch.float32)

        alpha = self.alpha

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state_batch)
            qf1_next_target = self.critic1_target(next_state_batch, next_state_action)
            qf2_next_target = self.critic2_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
            target_q_value = reward_batch + mask_batch * config.GAMMA * min_qf_next_target

        qf1 = self.critic1(state_batch, action_batch)
        qf2 = self.critic2(state_batch, action_batch)
        qf_loss = F.mse_loss(qf1, target_q_value) + F.mse_loss(qf2, target_q_value)

        self.critic_optimizer.zero_grad(set_to_none=True)
        q_loss_val = qf_loss
        q_loss_val.backward()
        self.critic_optimizer.step()

        pi, log_pi, _ = self.actor.sample(state_batch)
        qf1_pi = self.critic1(state_batch, pi)
        qf2_pi = self.critic2(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = (alpha * log_pi - min_qf_pi).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        if updates % 1 == 0:
            self._soft_update(self.critic1_target, self.critic1, config.TAU)
            self._soft_update(self.critic2_target, self.critic2, config.TAU)

        return {
            "q1_loss": qf_loss.item() / 2,
            "q2_loss": qf_loss.item() / 2,
            "actor_loss": policy_loss.item(),
            "alpha": alpha.item()
        }

    def _soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
