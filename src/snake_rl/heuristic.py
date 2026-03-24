"""
heuristic.py — Rule-based action selector for Snake.

Used to pre-fill the DQN replay buffer with decent transitions before
RL training begins, replacing purely random exploration with a
food-seeking, collision-avoiding policy.
"""

from __future__ import annotations

from snake_rl.core import Action, Food, Snake, apply_action


def heuristic_action(snake: Snake, food: Food) -> int:
    """Return the best relative action (0=STRAIGHT, 1=TURN_LEFT, 2=TURN_RIGHT).

    Evaluates all three relative moves and eliminates those that result in an
    immediate wall or self-collision. Among the safe moves, picks the one that
    minimises Manhattan distance to food. Falls back to STRAIGHT if every move
    is blocked (unavoidable death).

    Args:
        snake: Current snake state (head position, direction, body positions).
        food:  Current food state.

    Returns:
        An integer action value (0, 1, or 2).
    """
    head = snake.head
    fx, fy = food.position

    candidates: list[tuple[int, int]] = []  # (distance_to_food, action_value)
    for action in (Action.STRAIGHT, Action.TURN_LEFT, Action.TURN_RIGHT):
        new_dir = apply_action(snake.direction, action)
        nx = head[0] + new_dir.dx
        ny = head[1] + new_dir.dy

        # Wall check (border cells are walls; playable interior is 1..grid-2)
        if nx < 1 or nx >= snake.grid_w - 1 or ny < 1 or ny >= snake.grid_h - 1:
            continue
        # Self-collision check (exclude head; tail vacates on each step)
        if (nx, ny) in snake.positions[1:]:
            continue

        dist = abs(nx - fx) + abs(ny - fy)
        candidates.append((dist, action.value))

    if not candidates:
        return Action.STRAIGHT.value  # all moves blocked; accept death

    candidates.sort()
    return candidates[0][1]
