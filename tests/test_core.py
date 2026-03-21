"""Tests for snake_rl.core — game logic only, no pygame."""

import random
from unittest.mock import patch

import pytest

from snake_rl.core import (
    Action,
    Direction,
    Food,
    Snake,
    apply_action,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    OPPOSITES,
)

# ---------------------------------------------------------------------------
# Snake movement
# ---------------------------------------------------------------------------


def make_snake(direction, start=(12, 12), grid_w=24, grid_h=24):
    s = Snake(grid_w=grid_w, grid_h=grid_h)
    s.reset(start=start)
    s.direction = direction
    return s


class TestMovement:
    def test_step_up(self):
        s = make_snake(UP, start=(12, 12))
        s.step()
        assert s.head == (12, 11)

    def test_step_down(self):
        s = make_snake(DOWN, start=(12, 12))
        s.step()
        assert s.head == (12, 13)

    def test_step_left(self):
        s = make_snake(LEFT, start=(12, 12))
        s.step()
        assert s.head == (11, 12)

    def test_step_right(self):
        s = make_snake(RIGHT, start=(12, 12))
        s.step()
        assert s.head == (13, 12)

    def test_step_into_left_wall_returns_true(self):
        # Border cell x=0 is a wall; stepping from x=1 onto it is a collision
        s = make_snake(LEFT, start=(1, 12))
        assert s.step() is True

    def test_step_into_right_wall_returns_true(self):
        s = make_snake(RIGHT, start=(22, 12))
        assert s.step() is True

    def test_step_into_top_wall_returns_true(self):
        s = make_snake(UP, start=(12, 1))
        assert s.step() is True

    def test_step_into_bottom_wall_returns_true(self):
        s = make_snake(DOWN, start=(12, 22))
        assert s.step() is True

    def test_valid_step_returns_false(self):
        s = make_snake(RIGHT, start=(5, 5))
        assert s.step() is False

    def test_no_wrapping(self):
        """Confirm position stops at border rather than wrapping."""
        s = make_snake(LEFT, start=(1, 12))
        result = s.step()
        assert result is True
        assert s.head == (1, 12)  # position unchanged on collision

    def test_step_into_body_returns_true(self):
        # 4-cell snake curled so the next step lands on a body cell.
        # Head=(5,5), body=[(4,5),(4,6),(5,6)]. Stepping DOWN sends head to (5,6),
        # which is still in the body list before it gets a chance to pop.
        s = make_snake(DOWN, start=(5, 5))
        s.positions = [(5, 5), (4, 5), (4, 6), (5, 6)]
        s.length = 4
        result = s.step()
        assert result is True


# ---------------------------------------------------------------------------
# Turn logic
# ---------------------------------------------------------------------------


class TestTurn:
    def test_turn_to_new_direction_accepted(self):
        s = make_snake(UP)
        s.turn(LEFT)
        assert s.direction == LEFT

    def test_180_reversal_rejected_when_length_gt_1(self):
        s = make_snake(UP)
        s.length = 2
        s.turn(DOWN)
        assert s.direction == UP

    def test_180_reversal_accepted_when_length_1(self):
        s = make_snake(UP)
        assert s.length == 1
        s.turn(DOWN)
        assert s.direction == DOWN


# ---------------------------------------------------------------------------
# grow()
# ---------------------------------------------------------------------------


class TestGrow:
    def test_grow_increments_length(self):
        s = make_snake(UP)
        s.grow()
        assert s.length == 2

    def test_grow_increments_score(self):
        s = make_snake(UP)
        s.grow()
        assert s.score == 1

    def test_grow_updates_high_score_when_exceeded(self):
        s = make_snake(UP)
        s.high_score = 0
        s.grow()
        assert s.high_score == 1

    def test_grow_does_not_update_high_score_when_below(self):
        s = make_snake(UP)
        s.high_score = 10
        s.grow()
        assert s.high_score == 10


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_length_is_1(self):
        s = make_snake(UP)
        s.length = 5
        s.reset()
        assert s.length == 1

    def test_reset_score_is_0(self):
        s = make_snake(UP)
        s.score = 7
        s.reset()
        assert s.score == 0

    def test_reset_preserves_high_score(self):
        s = make_snake(UP)
        s.high_score = 42
        s.reset()
        assert s.high_score == 42

    def test_reset_positions_has_one_element(self):
        s = make_snake(UP)
        s.positions = [(1, 1), (2, 1), (3, 1)]
        s.reset()
        assert len(s.positions) == 1

    def test_reset_custom_start(self):
        s = make_snake(UP)
        s.reset(start=(3, 7))
        assert s.head == (3, 7)


# ---------------------------------------------------------------------------
# apply_action()
# ---------------------------------------------------------------------------


class TestApplyAction:
    @pytest.mark.parametrize(
        "heading, action, expected",
        [
            # STRAIGHT — all four headings unchanged
            (UP, Action.STRAIGHT, UP),
            (DOWN, Action.STRAIGHT, DOWN),
            (LEFT, Action.STRAIGHT, LEFT),
            (RIGHT, Action.STRAIGHT, RIGHT),
            # TURN_LEFT
            (UP, Action.TURN_LEFT, LEFT),
            (LEFT, Action.TURN_LEFT, DOWN),
            (DOWN, Action.TURN_LEFT, RIGHT),
            (RIGHT, Action.TURN_LEFT, UP),
            # TURN_RIGHT
            (UP, Action.TURN_RIGHT, RIGHT),
            (RIGHT, Action.TURN_RIGHT, DOWN),
            (DOWN, Action.TURN_RIGHT, LEFT),
            (LEFT, Action.TURN_RIGHT, UP),
        ],
    )
    def test_apply_action(self, heading, action, expected):
        assert apply_action(heading, action) == expected


# ---------------------------------------------------------------------------
# Food
# ---------------------------------------------------------------------------


class TestFood:
    def test_randomize_never_places_on_occupied(self):
        food = Food(grid_w=5, grid_h=5)
        occupied = {
            (x, y) for x in range(5) for y in range(5) if not (x == 2 and y == 2)
        }
        food.randomize(occupied)
        assert food.position == (2, 2)

    def test_randomize_nearly_full_board(self):
        """Only one free cell — food must land there."""
        food = Food(grid_w=3, grid_h=3)
        free_cell = (1, 1)
        occupied = {(x, y) for x in range(3) for y in range(3)} - {free_cell}
        food.randomize(occupied)
        assert food.position == free_cell

    def test_randomize_full_board_places_at_origin(self):
        """Completely full board — food falls back to (0, 0)."""
        food = Food(grid_w=3, grid_h=3)
        occupied = {(x, y) for x in range(3) for y in range(3)}
        food.randomize(occupied)
        assert food.position == (0, 0)

    def test_randomize_does_not_place_on_snake(self):
        """Randomize 100 times, confirm food never lands on snake body."""
        snake_cells = {(5, 5), (5, 6), (5, 7)}
        food = Food(grid_w=24, grid_h=24)
        for _ in range(100):
            food.randomize(snake_cells)
            assert food.position not in snake_cells
