"""
policy.py — Custom PyTorch MLP registered with SB3 as a features extractor.
No pygame. No gymnasium internals beyond BaseFeaturesExtractor.
"""

from __future__ import annotations

import numpy as np
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SnakeMLP(BaseFeaturesExtractor):
    """
    Multi-layer perceptron that flattens the (3, H, W) observation and passes
    it through configurable hidden layers before producing a fixed-size feature
    vector that SB3 feeds into its policy and value heads.
    """

    def __init__(
        self,
        observation_space,
        features_dim: int = 64,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__(observation_space, features_dim)

        if hidden_dims is None:
            hidden_dims = [256, 128]

        n_input = int(np.prod(observation_space.shape))
        layer_sizes = [n_input] + hidden_dims + [features_dim]

        layers: list[nn.Module] = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs.flatten(start_dim=1))
