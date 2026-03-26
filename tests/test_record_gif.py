"""Tests for scripts/record_gif.py."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scripts.record_gif import run_episode, save_gif


class _FakeModel:
    def predict(self, obs, deterministic=True):
        return 0, None  # always go straight


class _FakeEnv:
    def reset(self):
        return np.zeros(11, dtype=np.float32), {}

    def render(self):
        return np.zeros((360, 360, 3), dtype=np.uint8)

    def step(self, action):
        obs = np.zeros(11, dtype=np.float32)
        return obs, 0.0, True, False, {"score": 3}

    def close(self):
        pass


def test_run_episode_returns_frames_and_score():
    frames, score = run_episode(_FakeModel(), _FakeEnv())
    assert score == 3
    assert len(frames) >= 2  # at least reset frame + terminal frame
    assert frames[0].shape == (360, 360, 3)


def test_save_gif_creates_file(tmp_path):
    out = tmp_path / "test.gif"
    frames = [np.zeros((20, 20, 3), dtype=np.uint8) for _ in range(3)]
    save_gif(frames, out, fps=10)
    assert out.exists()
    assert out.stat().st_size > 0
