import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Tuple, List
from Risklayer.config import config

class SharedEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(SharedEncoder, self).__init__()
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
        super(Actor, self).__init__()
        self.encoder = encoder
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, x):
        features = self.encoder(x)
        mu = self.mu(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, x):
        mu, log_std = self.forward(x)
        std = log_std.exp()
        normal = Normal(mu, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, encoder: SharedEncoder, hidden_dim: int, action_dim: int):
        super(Critic, self).__init__()
        self.encoder = encoder
        self.q1 = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, a):
        features = self.encoder(x)
        xu = torch.cat([features, a], 1)
        return self.q1(xu), self.q2(xu)

class SACAgent:
    def __init__(self, state_dim: int, action_dim: int):
        self.device = torch.device(config.DEVICE)
        self.hidden_dim = config.HIDDEN_DIMS[0]

        self.encoder = SharedEncoder(state_dim, self.hidden_dim).to(self.device)
        self.actor = Actor(self.encoder, self.hidden_dim, action_dim).to(self.device)
        self.critic = Critic(self.encoder, self.hidden_dim, action_dim).to(self.device)
        self.critic_target = Critic(self.encoder, self.hidden_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.LR)

        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.LR)

        self.gamma = config.GAMMA
        self.tau = config.TAU

    def select_action(self, state: np.ndarray, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate:
            with torch.no_grad():
                mu, _ = self.actor(state)
                action = torch.tanh(mu)
        else:
            with torch.no_grad():
                action, _ = self.actor.sample(state)

        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi = self.actor.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.log_alpha.exp() * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target

        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        critic_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        pi, log_pi = self.actor.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        actor_loss = ((self.log_alpha.exp() * log_pi) - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.soft_update(self.critic_target, self.critic, self.tau)

        return critic_loss.item(), actor_loss.item(), alpha_loss.item()

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in batch:
            s, a, r, ns, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)
