"""
core.py — Headless Snake game logic. Zero pygame, zero torch.
All coordinates are integer grid units; pixels are the renderer's concern.
"""

import random
from enum import IntEnum
from typing import NamedTuple


class Direction(NamedTuple):
    dx: int
    dy: int


UP = Direction(0, -1)
DOWN = Direction(0, 1)
LEFT = Direction(-1, 0)
RIGHT = Direction(1, 0)

OPPOSITES: dict[Direction, Direction] = {
    UP: DOWN,
    DOWN: UP,
    LEFT: RIGHT,
    RIGHT: LEFT,
}

_TURN_LEFT: dict[Direction, Direction] = {
    UP: LEFT,
    LEFT: DOWN,
    DOWN: RIGHT,
    RIGHT: UP,
}

_TURN_RIGHT: dict[Direction, Direction] = {
    UP: RIGHT,
    RIGHT: DOWN,
    DOWN: LEFT,
    LEFT: UP,
}


class Action(IntEnum):
    STRAIGHT = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2


def apply_action(direction: Direction, action: Action) -> Direction:
    """Return the new heading after applying a relative action."""
    if action == Action.STRAIGHT:
        return direction
    if action == Action.TURN_LEFT:
        return _TURN_LEFT[direction]
    return _TURN_RIGHT[direction]


class Snake:
    def __init__(self, grid_w: int = 24, grid_h: int = 24) -> None:
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.high_score: int = 0
        self.length: int = 1
        self.score: int = 0
        self.positions: list[tuple[int, int]] = []
        self.direction: Direction = UP
        self.reset()

    @property
    def head(self) -> tuple[int, int]:
        return self.positions[0]

    def turn(self, direction: Direction) -> None:
        """Change heading, refusing 180° reversal when length > 1."""
        if self.length > 1 and direction == OPPOSITES[self.direction]:
            return
        self.direction = direction

    def step(self) -> bool:
        """Move one cell. Returns True on collision (game over), False otherwise."""
        cur = self.head
        new = (cur[0] + self.direction.dx, cur[1] + self.direction.dy)
        if self._is_collision(new):
            return True
        self.positions.insert(0, new)
        if len(self.positions) > self.length:
            self.positions.pop()
        return False

    def grow(self) -> None:
        """Extend snake by one, increment score, update high score if needed."""
        self.length += 1
        self.score += 1
        if self.score > self.high_score:
            self.high_score = self.score

    def reset(self, start: tuple[int, int] | None = None) -> None:
        """Reset to a single cell (default: grid center). Preserves high_score."""
        if start is None:
            start = (self.grid_w // 2, self.grid_h // 2)
        self.positions = [start]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.length = 1
        self.score = 0

    def body_cells(self) -> list[tuple[int, int]]:
        return self.positions[1:]

    def _is_collision(self, point: tuple[int, int]) -> bool:
        x, y = point
        # Border cells are walls — playable area is the interior
        if x < 1 or x >= self.grid_w - 1 or y < 1 or y >= self.grid_h - 1:
            return True
        if point in self.positions[1:]:
            return True
        return False


class Food:
    def __init__(self, grid_w: int = 24, grid_h: int = 24) -> None:
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.position: tuple[int, int] = (0, 0)
        self.randomize(set())

    def randomize(self, occupied: set[tuple[int, int]]) -> None:
        """Place food on a random free interior cell, excluding occupied positions."""
        free = [
            (x, y)
            for x in range(1, self.grid_w - 1)
            for y in range(1, self.grid_h - 1)
            if (x, y) not in occupied
        ]
        if not free:
            self.position = (0, 0)
        else:
            self.position = random.choice(free)
