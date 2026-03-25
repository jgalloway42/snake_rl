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
        assert obs.shape == (11,)

    def test_reset_obs_dtype(self):
        env = make_env()
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_step_obs_shape(self):
        env = make_env()
        env.reset()
        obs, *_ = env.step(Action.STRAIGHT)
        assert obs.shape == (11,)

    def test_step_obs_dtype(self):
        env = make_env()
        env.reset()
        obs, *_ = env.step(Action.STRAIGHT)
        assert obs.dtype == np.float32


# ---------------------------------------------------------------------------
# Observation correctness
# ---------------------------------------------------------------------------


class TestObservationCorrectness:
    def test_danger_straight_detects_wall(self):
        # Snake at (1,4) facing LEFT — next cell is (0,4), a wall
        env = make_env()
        env.reset()
        env._snake.positions = [(1, 4)]
        env._snake.direction = LEFT
        obs = env._get_obs()
        assert obs[0] == 1.0  # danger_straight

    def test_danger_left_detects_wall(self):
        # Snake at (4,1) facing RIGHT — turning left gives UP, next cell (4,0) is a wall
        env = make_env()
        env.reset()
        env._snake.positions = [(4, 1)]
        env._snake.direction = RIGHT
        obs = env._get_obs()
        assert obs[1] == 1.0  # danger_left

    def test_no_danger_open_space(self):
        # Snake in center facing RIGHT — all three directions are open
        env = make_env()
        env.reset()
        env._snake.positions = [(5, 5)]
        env._snake.direction = RIGHT
        env._food.position = (7, 7)
        obs = env._get_obs()
        assert obs[0] == 0.0  # danger_straight
        assert obs[1] == 0.0  # danger_left
        assert obs[2] == 0.0  # danger_right

    def test_food_left_flag(self):
        env = make_env()
        env.reset()
        env._snake.positions = [(5, 5)]
        env._food.position = (3, 5)  # food to the left
        obs = env._get_obs()
        assert obs[3] == 1.0  # food_left
        assert obs[4] == 0.0  # food_right

    def test_food_up_flag(self):
        env = make_env()
        env.reset()
        env._snake.positions = [(5, 5)]
        env._food.position = (5, 3)  # food above (lower Y)
        obs = env._get_obs()
        assert obs[5] == 1.0  # food_up
        assert obs[6] == 0.0  # food_down

    def test_direction_one_hot(self):
        # Heading RIGHT → obs[8] = 1.0, others 0.0
        env = make_env()
        env.reset()
        env._snake.direction = RIGHT
        obs = env._get_obs()
        assert obs[7] == 0.0  # dir_up
        assert obs[8] == 1.0  # dir_right
        assert obs[9] == 0.0  # dir_down
        assert obs[10] == 0.0  # dir_left


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
