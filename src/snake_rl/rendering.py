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
SNAKE_HEAD_COLOR = (255, 255, 0)
FOOD_COLOR = (255, 204, 0)
BORDER_COLOR = (255, 255, 255)
BORDER_COLLISION_COLOR = (220, 30, 30)
BORDER_TIMEOUT_COLOR = (220, 200, 0)
HUD_COLOR = (220, 220, 220)


class PygameRenderer:
    def __init__(
        self,
        grid_w: int,
        grid_h: int,
        cell_size: int = CELL_SIZE_DEFAULT,
        headless: bool = False,
        snake_color: tuple = SNAKE_COLOR,
        snake_head_color: tuple = SNAKE_HEAD_COLOR,
    ) -> None:
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.cell_size = cell_size
        self.screen_w = grid_w * cell_size
        self.screen_h = grid_h * cell_size
        self.snake_color = snake_color
        self.snake_head_color = snake_head_color

        pygame.init()
        if headless:
            self.screen = pygame.Surface((self.screen_w, self.screen_h))
        else:
            self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
            pygame.display.set_caption("snake-rl")
        self.surface = pygame.Surface((self.screen_w, self.screen_h))
        self.font = pygame.font.SysFont("terminal", 18)
        self.score_font = pygame.font.SysFont("monospace", 14, bold=True)

    def draw(
        self,
        snake,
        food,
        score: int,
        episode: int,
        step: int,
        collision: bool = False,
        timeout: bool = False,
    ) -> None:
        self._draw_grid(collision, timeout)
        self._draw_score_on_border(score)
        self._draw_snake(snake)
        self._draw_food(food)
        self._draw_hud(score, episode, step)
        self.screen.blit(self.surface, (0, 0))
        pygame.display.update()

    def get_rgb_array(
        self,
        snake,
        food,
        collision: bool = False,
        timeout: bool = False,
        score: int = 0,
    ) -> np.ndarray:
        self._draw_grid(collision, timeout)
        self._draw_score_on_border(score)
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

    def _draw_grid(self, collision: bool = False, timeout: bool = False) -> None:
        cs = self.cell_size
        if collision:
            border_color = BORDER_COLLISION_COLOR
        elif timeout:
            border_color = BORDER_TIMEOUT_COLOR
        else:
            border_color = BORDER_COLOR
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
        for i, (gx, gy) in enumerate(snake.positions):
            color = self.snake_head_color if i == 0 else self.snake_color
            rect = pygame.Rect(gx * cs, gy * cs, cs, cs)
            pygame.draw.rect(self.surface, color, rect)
            pygame.draw.rect(self.surface, BG_COLOR_DARK, rect, 1)

    def _draw_food(self, food) -> None:
        cs = self.cell_size
        fx, fy = food.position
        rect = pygame.Rect(fx * cs, fy * cs, cs, cs)
        pygame.draw.rect(self.surface, FOOD_COLOR, rect)
        pygame.draw.rect(self.surface, BG_COLOR_DARK, rect, 1)

    def _draw_score_on_border(self, score: int) -> None:
        """Render the score centered in the top border row."""
        text = self.score_font.render(f"Score: {score}", True, (20, 20, 20))
        x = (self.screen_w - text.get_width()) // 2
        y = (self.cell_size - text.get_height()) // 2
        self.surface.blit(text, (x, y))

    def _draw_hud(self, score: int, episode: int, step: int) -> None:
        cs = self.cell_size
        text = self.font.render(
            f"SCORE: {score}  EP: {episode}  STEP: {step}", True, HUD_COLOR
        )
        self.screen.blit(text, (cs, 4))
