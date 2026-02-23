import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import random
from collections import deque
from .config import RiskConfig

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (np.stack(state), np.stack(action), np.stack(reward), np.stack(next_state), np.stack(done))

    def __len__(self):
        return len(self.buffer)

class SharedEncoder(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super(SharedEncoder, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = self.layer_norm(F.relu(self.l2(x)))
        return x

class Actor(nn.Module):
    def __init__(self, encoder: nn.Module, action_dim: int, hidden_dim: int):
        super(Actor, self).__init__()
        self.encoder = encoder
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.encoder(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean

class Critic(nn.Module):
    def __init__(self, encoder: nn.Module, action_dim: int, hidden_dim: int):
        super(Critic, self).__init__()
        self.encoder = encoder
        # Q1 architecture
        self.l1 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.l3 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = self.encoder(state)
        xa = torch.cat([x, action], 1)

        q1 = F.relu(self.l1(xa))
        q1 = self.l2(q1)

        q2 = F.relu(self.l3(xa))
        q2 = self.l4(q2)
        return q1, q2

class SACAgent:
    def __init__(self, state_dim: int, action_dim: int, config: RiskConfig):
        self.config = config
        self.device = torch.device(config.DEVICE)

        self.encoder = SharedEncoder(state_dim, config.HIDDEN_DIM).to(self.device)
        self.actor = Actor(self.encoder, action_dim, config.HIDDEN_DIM).to(self.device)
        self.critic = Critic(self.encoder, action_dim, config.HIDDEN_DIM).to(self.device)

        # We need a separate encoder for the target critic to avoid leakage during updates
        self.encoder_target = SharedEncoder(state_dim, config.HIDDEN_DIM).to(self.device)
        self.critic_target = Critic(self.encoder_target, action_dim, config.HIDDEN_DIM).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.LR)

        self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.LR)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.actor.sample(state)
        else:
            _, _, action = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, replay_buffer, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = replay_buffer.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.log_alpha.exp() * next_state_log_pi
            next_q_value = reward_batch + (1 - mask_batch) * self.config.GAMMA * min_qf_next_target

        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        pi, log_pi, _ = self.actor.sample(state_batch)
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

        self.soft_update(self.critic, self.critic_target, self.config.TAU)

        return qf_loss.item(), actor_loss.item(), alpha_loss.item()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
