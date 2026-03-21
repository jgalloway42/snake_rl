"""Tests for snake_rl.policy — PyTorch MLP features extractor."""

import numpy as np
import pytest
import torch
import gymnasium as gym
from gymnasium import spaces

from snake_rl.policy import SnakeMLP


def make_obs_space(grid_w=24, grid_h=24):
    return spaces.Box(low=0.0, high=1.0, shape=(3, grid_h, grid_w), dtype=np.float32)


def random_obs_batch(obs_space, batch_size=4):
    raw = np.stack([obs_space.sample() for _ in range(batch_size)])
    return torch.tensor(raw, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


class TestForwardPass:
    def test_output_shape_default(self):
        obs_space = make_obs_space()
        model = SnakeMLP(obs_space)
        batch = random_obs_batch(obs_space, batch_size=4)
        out = model(batch)
        assert out.shape == (4, 64)

    def test_output_shape_custom_features_dim(self):
        obs_space = make_obs_space()
        model = SnakeMLP(obs_space, features_dim=32)
        batch = random_obs_batch(obs_space, batch_size=2)
        out = model(batch)
        assert out.shape == (2, 32)

    def test_no_nan_in_output(self):
        obs_space = make_obs_space()
        model = SnakeMLP(obs_space)
        batch = random_obs_batch(obs_space, batch_size=8)
        out = model(batch)
        assert not torch.isnan(out).any()

    def test_no_inf_in_output(self):
        obs_space = make_obs_space()
        model = SnakeMLP(obs_space)
        batch = random_obs_batch(obs_space, batch_size=8)
        out = model(batch)
        assert not torch.isinf(out).any()

    def test_output_varies_across_inputs(self):
        """Different inputs should produce different outputs (not a dead network)."""
        obs_space = make_obs_space()
        model = SnakeMLP(obs_space)
        batch = random_obs_batch(obs_space, batch_size=2)
        out = model(batch)
        assert not torch.allclose(out[0], out[1])


# ---------------------------------------------------------------------------
# Configurability
# ---------------------------------------------------------------------------


class TestConfigurability:
    def test_custom_hidden_dims(self):
        obs_space = make_obs_space()
        model = SnakeMLP(obs_space, features_dim=32, hidden_dims=[128, 64, 32])
        batch = random_obs_batch(obs_space, batch_size=3)
        out = model(batch)
        assert out.shape == (3, 32)

    def test_single_hidden_layer(self):
        obs_space = make_obs_space()
        model = SnakeMLP(obs_space, features_dim=16, hidden_dims=[64])
        batch = random_obs_batch(obs_space, batch_size=1)
        out = model(batch)
        assert out.shape == (1, 16)

    def test_small_grid_obs_space(self):
        obs_space = make_obs_space(grid_w=8, grid_h=8)
        model = SnakeMLP(obs_space, features_dim=64)
        batch = random_obs_batch(obs_space, batch_size=4)
        out = model(batch)
        assert out.shape == (4, 64)
