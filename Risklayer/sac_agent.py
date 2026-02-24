import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple

from Risklayer.config import config

class SharedEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.shared_encoder = SharedEncoder(config.STATE_DIM, config.HIDDEN_DIM).to(self.device)

        self.actor = Actor(self.shared_encoder, config.HIDDEN_DIM, config.ACTION_DIM).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.LR)

        self.critic1 = Critic(self.shared_encoder, config.HIDDEN_DIM, config.ACTION_DIM).to(self.device)
        self.critic2 = Critic(self.shared_encoder, config.HIDDEN_DIM, config.ACTION_DIM).to(self.device)
        self.critic1_target = Critic(self.shared_encoder, config.HIDDEN_DIM, config.ACTION_DIM).to(self.device)
        self.critic2_target = Critic(self.shared_encoder, config.HIDDEN_DIM, config.ACTION_DIM).to(self.device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=config.LR
        )

        self.target_entropy = -torch.prod(torch.Tensor([config.ACTION_DIM]).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.LR)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.actor.sample(state)
        else:
            _, _, action = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state_batch)
            qf1_next_target = self.critic1_target(next_state_batch, next_state_action)
            qf2_next_target = self.critic2_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * config.GAMMA * (min_qf_next_target)

        qf1 = self.critic1(state_batch, action_batch)
        qf2 = self.critic2(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        q_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        pi, log_pi, _ = self.actor.sample(state_batch)
        qf1_pi = self.critic1(state_batch, pi)
        qf2_pi = self.critic2(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()

        if updates % 1 == 0:
            self._soft_update(self.critic1_target, self.critic1, config.TAU)
            self._soft_update(self.critic2_target, self.critic2, config.TAU)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item()

    def _soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @alpha.setter
    def alpha(self, value):
        pass # Handle if needed
