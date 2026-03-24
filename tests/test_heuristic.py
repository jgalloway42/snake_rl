"""Tests for snake_rl.heuristic — rule-based action selector."""

from snake_rl.core import Action, Food, Snake, RIGHT, LEFT, UP, DOWN
from snake_rl.heuristic import heuristic_action


def make_snake(pos, direction, body=None, grid_w=10, grid_h=10):
    snake = Snake(grid_w=grid_w, grid_h=grid_h)
    snake.positions = [pos] + (body or [])
    snake.length = len(snake.positions)
    snake.direction = direction
    return snake


def make_food(pos, grid_w=10, grid_h=10):
    food = Food(grid_w=grid_w, grid_h=grid_h)
    food.position = pos
    return food


class TestHeuristicAction:
    def test_returns_valid_action(self):
        snake = make_snake((5, 5), RIGHT)
        food = make_food((7, 5))
        result = heuristic_action(snake, food)
        assert result in (Action.STRAIGHT, Action.TURN_LEFT, Action.TURN_RIGHT)

    def test_returns_int(self):
        snake = make_snake((5, 5), RIGHT)
        food = make_food((7, 5))
        assert isinstance(heuristic_action(snake, food), int)

    def test_moves_straight_toward_food(self):
        # Snake at (5,5) facing RIGHT, food at (7,5) — straight is closest
        # STRAIGHT → (6,5) dist=1; LEFT(UP) → (5,4) dist=3; RIGHT(DOWN) → (5,6) dist=3
        snake = make_snake((5, 5), RIGHT)
        food = make_food((7, 5))
        assert heuristic_action(snake, food) == Action.STRAIGHT

    def test_turns_left_toward_food(self):
        # Snake at (5,5) facing RIGHT, food at (5,3) — turning left (UP) is closest
        # STRAIGHT → (6,5) dist=3; TURN_LEFT(UP) → (5,4) dist=1; TURN_RIGHT(DOWN) → (5,6) dist=3
        snake = make_snake((5, 5), RIGHT)
        food = make_food((5, 3))
        assert heuristic_action(snake, food) == Action.TURN_LEFT

    def test_turns_right_toward_food(self):
        # Snake at (5,5) facing RIGHT, food at (5,7) — turning right (DOWN) is closest
        # STRAIGHT → (6,5) dist=3; TURN_LEFT(UP) → (5,4) dist=3; TURN_RIGHT(DOWN) → (5,6) dist=1
        snake = make_snake((5, 5), RIGHT)
        food = make_food((5, 7))
        assert heuristic_action(snake, food) == Action.TURN_RIGHT

    def test_avoids_wall_collision(self):
        # Snake at (1,5) facing LEFT — STRAIGHT would hit the left wall
        snake = make_snake((1, 5), LEFT)
        food = make_food((5, 5))
        result = heuristic_action(snake, food)
        assert result != Action.STRAIGHT

    def test_avoids_top_wall(self):
        # Snake at (5,1) facing UP — STRAIGHT would hit the top wall
        snake = make_snake((5, 1), UP)
        food = make_food((5, 5))
        result = heuristic_action(snake, food)
        assert result != Action.STRAIGHT

    def test_avoids_body_collision(self):
        # Snake at (5,5) facing UP with body segment directly above at (5,4)
        # STRAIGHT → (5,4) = body; must turn
        snake = make_snake((5, 5), UP, body=[(5, 4)])
        food = make_food((3, 5))
        result = heuristic_action(snake, food)
        assert result != Action.STRAIGHT

    def test_body_avoidance_picks_toward_food(self):
        # Snake at (5,5) facing UP, body at (5,4) blocks STRAIGHT
        # Food at (3,5): TURN_LEFT(LEFT)→(4,5) dist=1; TURN_RIGHT(RIGHT)→(6,5) dist=3
        snake = make_snake((5, 5), UP, body=[(5, 4)])
        food = make_food((3, 5))
        assert heuristic_action(snake, food) == Action.TURN_LEFT

    def test_all_moves_blocked_returns_straight(self):
        # Snake at (1,1) facing UP:
        #   STRAIGHT(UP) → (1,0) wall; TURN_LEFT(LEFT) → (0,1) wall; TURN_RIGHT(RIGHT) → (2,1) body
        snake = make_snake((1, 1), UP, body=[(2, 1)])
        food = make_food((5, 5))
        assert heuristic_action(snake, food) == Action.STRAIGHT
