"""
env.py — Gymnasium wrapper for the Snake game.
No pygame imports at module level; rendering is lazy.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from snake_rl.core import Action, Food, Snake, apply_action, DOWN, LEFT, RIGHT, UP


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        grid_w: int = 24,
        grid_h: int = 24,
        max_steps: int = 500,
        render_mode: str | None = None,
        food_reward: float = 10.0,
        collision_penalty: float = -10.0,
        toward_reward: float = 0.1,
        away_penalty: float = -0.3,
        step_penalty: float = 0.0,
        snake_color: tuple = (51, 255, 0),
        snake_head_color: tuple = (15, 150, 15),
    ) -> None:
        super().__init__()
        # grid_w/grid_h from config are the *playable* interior dimensions.
        # Internally we add a 1-cell border on each side, so the total grid
        # (including walls) is (grid_w + 2) × (grid_h + 2).
        self.grid_w = grid_w + 2
        self.grid_h = grid_h + 2
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.food_reward = food_reward
        self.collision_penalty = collision_penalty
        self.toward_reward = toward_reward
        self.away_penalty = away_penalty
        self.step_penalty = step_penalty
        self.snake_color = snake_color
        self.snake_head_color = snake_head_color

        obs_size = 11
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        self._snake: Snake | None = None
        self._food: Food | None = None
        self._steps: int = 0
        self._collision: bool = False
        self._timeout: bool = False
        self._renderer = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._snake = Snake(grid_w=self.grid_w, grid_h=self.grid_h)
        occupied = set(self._snake.positions)
        self._food = Food(grid_w=self.grid_w, grid_h=self.grid_h)
        self._food.randomize(occupied)
        self._steps = 0
        self._collision = False
        self._timeout = False

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_human()

        return obs, info

    def step(self, action: int):
        assert self._snake is not None, "Call reset() before step()"

        new_dir = apply_action(self._snake.direction, Action(action))
        self._snake.turn(new_dir)

        hx, hy = self._snake.head
        fx, fy = self._food.position
        dist_before = abs(hx - fx) + abs(hy - fy)

        collision = self._snake.step()
        self._steps += 1

        terminated = False
        truncated = False
        reward = 0.0

        if collision:
            reward = self.collision_penalty
            terminated = True
        elif self._snake.head == self._food.position:
            reward = self.food_reward + self.step_penalty
            self._snake.grow()
            occupied = set(self._snake.positions)
            self._food.randomize(occupied)
        else:
            hx2, hy2 = self._snake.head
            dist_after = abs(hx2 - fx) + abs(hy2 - fy)
            reward = (
                self.toward_reward if dist_after < dist_before else self.away_penalty
            )
            reward += self.step_penalty

        if not terminated and self._steps >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = self._get_info()

        self._collision = terminated
        self._timeout = truncated

        if self.render_mode == "human":
            self._render_human()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_rgb_array()
        if self.render_mode == "human":
            self._render_human()
            return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def handle_events(self) -> bool:
        """Pump the pygame event queue. Returns False if quit was requested."""
        if self._renderer is None:
            return True
        return self._renderer.handle_events()

    def get_heuristic_action(self) -> int:
        """Return a rule-based action for the current game state.

        Uses the collision-avoiding, food-seeking heuristic from
        ``snake_rl.heuristic``.  Raises ``AssertionError`` if called before
        ``reset()``.
        """
        assert self._snake is not None, "Call reset() before get_heuristic_action()"
        from snake_rl.heuristic import heuristic_action  # lazy import

        return heuristic_action(self._snake, self._food)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _DIRECTION_INDEX = {UP: 0, RIGHT: 1, DOWN: 2, LEFT: 3}

    def _get_obs(self) -> np.ndarray:
        hx, hy = self._snake.head
        fx, fy = self._food.position
        d = self._snake.direction

        left_dir = apply_action(d, Action.TURN_LEFT)
        right_dir = apply_action(d, Action.TURN_RIGHT)

        body = self._snake.positions[1:]

        def danger(direction) -> float:
            cx, cy = hx + direction.dx, hy + direction.dy
            if cx < 1 or cx >= self.grid_w - 1 or cy < 1 or cy >= self.grid_h - 1:
                return 1.0
            return 1.0 if (cx, cy) in body else 0.0

        one_hot = [0.0, 0.0, 0.0, 0.0]
        one_hot[self._DIRECTION_INDEX[d]] = 1.0

        return np.array(
            [
                danger(d),  # danger straight
                danger(left_dir),  # danger left
                danger(right_dir),  # danger right
                1.0 if fx < hx else 0.0,  # food left
                1.0 if fx > hx else 0.0,  # food right
                1.0 if fy < hy else 0.0,  # food up (Y increases downward)
                1.0 if fy > hy else 0.0,  # food down
                *one_hot,  # heading one-hot: up, right, down, left
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict:
        return {
            "score": self._snake.score,
            "length": self._snake.length,
            "steps": self._steps,
        }

    def _get_renderer(self):
        if self._renderer is None:
            from snake_rl.rendering import PygameRenderer  # lazy import

            self._renderer = PygameRenderer(
                self.grid_w,
                self.grid_h,
                headless=(self.render_mode == "rgb_array"),
                snake_color=self.snake_color,
                snake_head_color=self.snake_head_color,
            )
        return self._renderer

    def _render_human(self):
        renderer = self._get_renderer()
        renderer.draw(
            self._snake,
            self._food,
            self._snake.score,
            0,
            self._steps,
            self._collision,
            self._timeout,
        )

    def _get_rgb_array(self) -> np.ndarray:
        renderer = self._get_renderer()
        return renderer.get_rgb_array(
            self._snake, self._food, self._collision, self._timeout, self._snake.score
        )
