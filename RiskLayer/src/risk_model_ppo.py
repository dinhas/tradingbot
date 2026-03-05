import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Dict, List, Optional, Tuple, Type, Union

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super(ResidualBlock, self).__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.ln1(x)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return residual + out

class RiskFeatureExtractor(nn.Module):
    """
    Custom feature extractor for Risk Layer using Residual Blocks.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_res_blocks: int = 3):
        super(RiskFeatureExtractor, self).__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim)
        )
        
        self.backbone = nn.Sequential(*[
            ResidualBlock(hidden_dim) for _ in range(num_res_blocks)
        ])
        
        self.latent_dim_pi = hidden_dim
        self.latent_dim_vf = hidden_dim

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (latent_policy, latent_value)
        """
        shared_latent = self.input_proj(features)
        shared_latent = self.backbone(shared_latent)
        return shared_latent, shared_latent

class RiskActorCriticPolicy(ActorCriticPolicy):
    """
    Custom PPO Policy for the Risk Layer.
    """
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):
        # We handle the architecture internally using ResidualBlocks
        super(RiskActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        """
        Constructs the MLP extractor using the custom Residual backbone.
        """
        # input_dim comes from observation_space.shape[0]
        input_dim = self.observation_space.shape[0]
        self.mlp_extractor = RiskFeatureExtractor(input_dim=input_dim)
