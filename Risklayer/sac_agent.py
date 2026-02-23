import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import random
from collections import deque
from .config import config

class SharedEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(SharedEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class Actor(nn.Module):
    def __init__(self, encoder: nn.Module, encoder_output_dim: int, action_dim: int):
        super(Actor, self).__init__()
        self.encoder = encoder
        self.mu = nn.Linear(encoder_output_dim, action_dim)
        self.log_std = nn.Linear(encoder_output_dim, action_dim)

    def forward(self, x):
        e = self.encoder(x)
        mu = self.mu(e)
        log_std = self.log_std(e)
        log_std = torch.clamp(log_std, -20, 2)
        return mu, log_std

    def sample(self, x):
        mu, log_std = self.forward(x)
        std = log_std.exp()
        normal = Normal(mu, std)
        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, encoder: nn.Module, encoder_output_dim: int, action_dim: int):
        super(Critic, self).__init__()
        self.encoder = encoder
        self.q1 = nn.Sequential(
            nn.Linear(encoder_output_dim + action_dim, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(encoder_output_dim + action_dim, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM, 1)
        )

    def forward(self, x, action):
        e = self.encoder(x)
        xu = torch.cat([e, action], 1)
        return self.q1(xu), self.q2(xu)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)

class SACAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Encoder is shared between Actor and both Critics (using 3 copies of same architecture but only one if desired)
        # However, typically shared encoder means same weights.
        # But for stability in SAC, critics and actors sometimes have separate encoders.
        # Requirement says "Shared encoder".
        self.shared_encoder = SharedEncoder(state_dim, config.HIDDEN_DIM).to(self.device)

        self.actor = Actor(self.shared_encoder, config.HIDDEN_DIM, action_dim).to(self.device)
        self.critic = Critic(self.shared_encoder, config.HIDDEN_DIM, action_dim).to(self.device)
        self.critic_target = Critic(self.shared_encoder, config.HIDDEN_DIM, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.LEARNING_RATE)

        # Entropy
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.LEARNING_RATE)

        self.memory = ReplayBuffer(config.BUFFER_SIZE)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate:
            mu, _ = self.actor.forward(state)
            action = torch.tanh(mu)
        else:
            action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, batch_size):
        if len(self.memory) < batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.memory.sample(batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi = self.actor.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1 - mask_batch) * config.GAMMA * min_qf_next_target

        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        pi, log_pi = self.actor.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.soft_update(self.critic_target, self.critic, config.TAU)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def save_checkpoint(self, filename):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
