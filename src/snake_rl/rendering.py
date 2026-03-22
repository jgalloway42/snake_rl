"""
rendering.py — All pygame code lives here and only here.
Excluded from test coverage. No game logic resides in this module.
"""

import numpy as np
import pygame

CELL_SIZE_DEFAULT = 20

BG_COLOR_DARK = (40, 40, 40)
BG_COLOR_LIGHT = (52, 52, 52)
SNAKE_COLOR = (51, 255, 0)
FOOD_COLOR = (255, 204, 0)
BORDER_COLOR = (255, 255, 255)
BORDER_COLLISION_COLOR = (220, 30, 30)
HUD_COLOR = (220, 220, 220)


class PygameRenderer:
    def __init__(
        self, grid_w: int, grid_h: int, cell_size: int = CELL_SIZE_DEFAULT
    ) -> None:
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.cell_size = cell_size
        self.screen_w = grid_w * cell_size
        self.screen_h = grid_h * cell_size

        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("snake-rl")
        self.surface = pygame.Surface((self.screen_w, self.screen_h))
        self.font = pygame.font.SysFont("terminal", 18)

    def draw(
        self, snake, food, score: int, episode: int, step: int, collision: bool = False
    ) -> None:
        self._draw_grid(collision)
        self._draw_snake(snake)
        self._draw_food(food)
        self._draw_hud(score, episode, step)
        self.screen.blit(self.surface, (0, 0))
        pygame.display.update()

    def get_rgb_array(self, snake, food, collision: bool = False) -> np.ndarray:
        self._draw_grid(collision)
        self._draw_snake(snake)
        self._draw_food(food)
        self.screen.blit(self.surface, (0, 0))
        return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)

    def close(self) -> None:
        pygame.quit()

    def handle_events(self) -> bool:
        """Pump event queue. Returns False if quit was requested."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    # ------------------------------------------------------------------
    # Private drawing helpers
    # ------------------------------------------------------------------

    def _draw_grid(self, collision: bool = False) -> None:
        cs = self.cell_size
        border_color = BORDER_COLLISION_COLOR if collision else BORDER_COLOR
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                rect = pygame.Rect(x * cs, y * cs, cs, cs)
                if x == 0 or x == self.grid_w - 1 or y == 0 or y == self.grid_h - 1:
                    pygame.draw.rect(self.surface, border_color, rect)
                elif (x + y) % 2 == 0:
                    pygame.draw.rect(self.surface, BG_COLOR_DARK, rect)
                else:
                    pygame.draw.rect(self.surface, BG_COLOR_LIGHT, rect)

    def _draw_snake(self, snake) -> None:
        cs = self.cell_size
        for gx, gy in snake.positions:
            rect = pygame.Rect(gx * cs, gy * cs, cs, cs)
            pygame.draw.rect(self.surface, SNAKE_COLOR, rect)
            pygame.draw.rect(self.surface, BG_COLOR_DARK, rect, 1)

    def _draw_food(self, food) -> None:
        cs = self.cell_size
        fx, fy = food.position
        rect = pygame.Rect(fx * cs, fy * cs, cs, cs)
        pygame.draw.rect(self.surface, FOOD_COLOR, rect)
        pygame.draw.rect(self.surface, BG_COLOR_DARK, rect, 1)

    def _draw_hud(self, score: int, episode: int, step: int) -> None:
        cs = self.cell_size
        text = self.font.render(
            f"SCORE: {score}  EP: {episode}  STEP: {step}", True, HUD_COLOR
        )
        self.screen.blit(text, (cs, 4))
