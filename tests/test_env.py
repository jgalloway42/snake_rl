"""Tests for snake_rl.env — Gymnasium wrapper."""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from snake_rl.core import Action, DOWN, LEFT, RIGHT, UP
from snake_rl.env import SnakeEnv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_env(grid_w=8, grid_h=8, max_steps=50):
    return SnakeEnv(grid_w=grid_w, grid_h=grid_h, max_steps=max_steps)


# ---------------------------------------------------------------------------
# Observation shape and dtype
# ---------------------------------------------------------------------------


class TestObservationShape:
    def test_reset_obs_shape(self):
        env = make_env()
        obs, _ = env.reset()
        # grid_w=8 playable → 10×10 total (+ 1-cell border each side)
        # flat: 3 * 10 * 10 grid + 4 direction one-hot = 304
        assert obs.shape == (304,)

    def test_reset_obs_dtype(self):
        env = make_env()
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_step_obs_shape(self):
        env = make_env()
        env.reset()
        obs, *_ = env.step(Action.STRAIGHT)
        assert obs.shape == (
            304,
        )  # grid_w=8 playable → 10×10 total, flat + 4 direction

    def test_step_obs_dtype(self):
        env = make_env()
        env.reset()
        obs, *_ = env.step(Action.STRAIGHT)
        assert obs.dtype == np.float32


# ---------------------------------------------------------------------------
# Observation correctness
# ---------------------------------------------------------------------------


class TestObservationCorrectness:
    def test_channel_0_has_one_at_head(self):
        env = make_env()
        obs, _ = env.reset()
        grid = obs[:300].reshape(3, 10, 10)
        hx, hy = env._snake.head
        assert grid[0, hy, hx] == 1.0
        assert grid[0].sum() == 1.0

    def test_channel_1_body_cells(self):
        env = make_env()
        obs, _ = env.reset()
        grid = obs[:300].reshape(3, 10, 10)
        # Single-cell snake — body channel should be all zeros
        assert grid[1].sum() == 0.0

    def test_channel_1_no_head(self):
        """After growing, body channel has 1s only at body cells, not head."""
        env = make_env()
        env.reset()
        # Manually grow and extend positions to get a 3-cell snake
        env._snake.positions = [(4, 4), (3, 4), (2, 4)]
        env._snake.length = 3
        obs = env._get_obs()
        grid = obs[:300].reshape(3, 10, 10)
        hx, hy = env._snake.head
        assert grid[1, hy, hx] == 0.0
        assert grid[1].sum() == 2.0

    def test_channel_2_food_position(self):
        env = make_env()
        obs, _ = env.reset()
        grid = obs[:300].reshape(3, 10, 10)
        fx, fy = env._food.position
        assert grid[2, fy, fx] == 1.0
        assert grid[2].sum() == 1.0


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------


class TestReward:
    def _place_food_adjacent(self, env, direction):
        """Place food one step ahead of the snake's head."""
        hx, hy = env._snake.head
        env._food.position = (hx + direction.dx, hy + direction.dy)

    def test_eating_food_reward(self):
        env = make_env()
        env.reset()
        env._snake.direction = RIGHT
        hx, hy = env._snake.head
        env._food.position = (hx + 1, hy)
        _, reward, terminated, _, _ = env.step(Action.STRAIGHT)
        assert not terminated
        assert reward == pytest.approx(10.0)

    def test_collision_reward(self):
        env = make_env()
        env.reset()
        # Put snake at left wall facing left
        env._snake.positions = [(0, 4)]
        env._snake.direction = LEFT
        env._food.position = (7, 7)
        _, reward, terminated, _, _ = env.step(Action.STRAIGHT)
        assert terminated
        assert reward == pytest.approx(-10.0)

    def test_moving_toward_food_reward(self):
        env = make_env()
        env.reset()
        env._snake.positions = [(3, 3)]
        env._snake.direction = RIGHT
        env._food.position = (6, 3)  # food to the right — moving closer
        _, reward, _, _, _ = env.step(Action.STRAIGHT)
        assert reward == pytest.approx(0.1)

    def test_moving_away_from_food_reward(self):
        env = make_env()
        env.reset()
        env._snake.positions = [(3, 3)]
        env._snake.direction = LEFT
        env._food.position = (6, 3)  # food to the right — moving away
        _, reward, terminated, _, _ = env.step(Action.STRAIGHT)
        if not terminated:
            assert reward == pytest.approx(env.away_penalty)


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------


class TestTermination:
    def test_terminated_on_collision(self):
        env = make_env()
        env.reset()
        env._snake.positions = [(0, 4)]
        env._snake.direction = LEFT
        env._food.position = (7, 7)
        _, _, terminated, truncated, _ = env.step(Action.STRAIGHT)
        assert terminated is True
        assert truncated is False

    def test_truncated_after_max_steps(self):
        # Snake at (3,3) facing RIGHT: steps to (4,3),(5,3),(6,3) — all interior.
        # max_steps=3 triggers truncation before any wall is reached.
        env = SnakeEnv(grid_w=8, grid_h=8, max_steps=3)
        env.reset()
        env._snake.positions = [(3, 3)]
        env._snake.direction = RIGHT
        env._food.position = (1, 6)
        for _ in range(3):
            _, _, terminated, truncated, _ = env.step(Action.STRAIGHT)
            if terminated or truncated:
                break
        assert truncated is True
        assert terminated is False  # one must trigger

    def test_neither_flag_on_normal_step(self):
        env = make_env()
        env.reset()
        env._snake.positions = [(4, 4)]
        env._snake.direction = RIGHT
        env._food.position = (0, 0)
        _, _, terminated, truncated, _ = env.step(Action.STRAIGHT)
        assert terminated is False
        assert truncated is False


# ---------------------------------------------------------------------------
# Info dict
# ---------------------------------------------------------------------------


class TestInfoDict:
    def test_reset_info_keys(self):
        env = make_env()
        _, info = env.reset()
        assert "score" in info
        assert "length" in info
        assert "steps" in info

    def test_step_info_keys(self):
        env = make_env()
        env.reset()
        _, _, _, _, info = env.step(Action.STRAIGHT)
        assert "score" in info
        assert "length" in info
        assert "steps" in info

    def test_steps_increments(self):
        env = make_env()
        env.reset()
        _, _, _, _, info = env.step(Action.STRAIGHT)
        assert info["steps"] == 1


# ---------------------------------------------------------------------------
# Gymnasium compliance
# ---------------------------------------------------------------------------


class TestGymnasiumCompliance:
    def test_obs_in_observation_space_after_reset(self):
        env = make_env()
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    def test_obs_in_observation_space_after_step(self):
        env = make_env()
        env.reset()
        obs, *_ = env.step(Action.STRAIGHT)
        assert env.observation_space.contains(obs)

    def test_check_env(self):
        from stable_baselines3.common.env_checker import check_env

        env = SnakeEnv(grid_w=8, grid_h=8, max_steps=50)
        check_env(env, warn=True)


# ---------------------------------------------------------------------------
# Rendering paths (mocked — no pygame required)
# ---------------------------------------------------------------------------


class TestRenderingPaths:
    def _make_mock_renderer(self):
        renderer = MagicMock()
        renderer.get_rgb_array.return_value = MagicMock()
        return renderer

    def test_render_human_mode_calls_renderer(self):
        with patch("snake_rl.rendering.PygameRenderer") as MockRenderer:
            mock_instance = self._make_mock_renderer()
            MockRenderer.return_value = mock_instance

            env = SnakeEnv(grid_w=8, grid_h=8, render_mode="human")
            env.reset()
            assert mock_instance.draw.called

    def test_render_rgb_array_returns_array(self):
        import numpy as np

        fake_frame = np.zeros((160, 160, 3), dtype=np.uint8)
        with patch("snake_rl.rendering.PygameRenderer") as MockRenderer:
            mock_instance = self._make_mock_renderer()
            mock_instance.get_rgb_array.return_value = fake_frame
            MockRenderer.return_value = mock_instance

            env = SnakeEnv(grid_w=8, grid_h=8, render_mode="rgb_array")
            env.reset()
            result = env.render()
            assert result is fake_frame

    def test_render_human_explicit_call(self):
        with patch("snake_rl.rendering.PygameRenderer") as MockRenderer:
            mock_instance = self._make_mock_renderer()
            MockRenderer.return_value = mock_instance

            env = SnakeEnv(grid_w=8, grid_h=8, render_mode="human")
            env.reset()
            result = env.render()
            assert result is None

    def test_render_none_mode_returns_none(self):
        env = make_env()
        env.reset()
        result = env.render()
        assert result is None

    def test_close_shuts_down_renderer(self):
        with patch("snake_rl.rendering.PygameRenderer") as MockRenderer:
            mock_instance = self._make_mock_renderer()
            MockRenderer.return_value = mock_instance

            env = SnakeEnv(grid_w=8, grid_h=8, render_mode="human")
            env.reset()
            env.close()
            assert mock_instance.close.called

    def test_step_with_human_render(self):
        with patch("snake_rl.rendering.PygameRenderer") as MockRenderer:
            mock_instance = self._make_mock_renderer()
            MockRenderer.return_value = mock_instance

            env = SnakeEnv(grid_w=8, grid_h=8, render_mode="human")
            env.reset()
            env.step(Action.STRAIGHT)
            assert mock_instance.draw.call_count >= 2  # reset + step

    def test_handle_events_with_renderer(self):
        with patch("snake_rl.rendering.PygameRenderer") as MockRenderer:
            mock_instance = self._make_mock_renderer()
            mock_instance.handle_events.return_value = True
            MockRenderer.return_value = mock_instance

            env = SnakeEnv(grid_w=8, grid_h=8, render_mode="human")
            env.reset()
            result = env.handle_events()
            assert result is True
            assert mock_instance.handle_events.called

    def test_handle_events_no_renderer_returns_true(self):
        env = make_env()
        env.reset()
        assert env.handle_events() is True


# ---------------------------------------------------------------------------
# Heuristic action via env
# ---------------------------------------------------------------------------


class TestGetHeuristicAction:
    def test_returns_valid_action(self):
        env = make_env()
        env.reset()
        result = env.get_heuristic_action()
        assert result in (Action.STRAIGHT, Action.TURN_LEFT, Action.TURN_RIGHT)

    def test_raises_before_reset(self):
        env = make_env()
        with pytest.raises(AssertionError):
            env.get_heuristic_action()
