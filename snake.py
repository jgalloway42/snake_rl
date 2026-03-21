# -*- coding: utf-8 -*-
"""
Snake Game using pygame
"""
import pygame
import sys
import random

SCREEN_WIDTH = 480
SCREEN_HEIGHT = 480

GRIDSIZE = 20
GRID_WIDTH = SCREEN_WIDTH/GRIDSIZE
GRID_HEIGHT = SCREEN_HEIGHT/GRIDSIZE

BG_COLOR_DARK = (40, 40, 40)
BG_COLOR_LIGHT = (52, 52, 52)
SNAKE_COLOR = (51, 255, 0)
FOOD_COLOR = (255, 204, 0)

UP = (0,-1)
DOWN = (0,1)
LEFT = (-1,0)
RIGHT = (1,0)

class Snake():
    def __init__(self):
        self.color = SNAKE_COLOR
        self.high_score = 0
        self.reset()
        
    def get_head_position(self):
        return self.positions[0]
    
    def turn(self, point):
        if self.length > 1 and (point[0]*(-1), point[1]*(-1)) == self.direction:
            return
        else:
            self.direction = point
            
            
    def is_collision(self, point):
        #hit self
        if point in self.positions[1:]:
            return True
        # hit boundary
        if point[0] > SCREEN_WIDTH - GRID_WIDTH or point[0] < GRID_WIDTH/2 or \
            point[1] > SCREEN_HEIGHT - GRID_HEIGHT or point[1] < GRID_WIDTH/2:
                return True
        return False
    
    def move(self):
        cur = self.get_head_position()
        x, y = self.direction
        new = ((cur[0] + x*GRIDSIZE) % SCREEN_WIDTH, (cur[1] + y*GRIDSIZE) % SCREEN_HEIGHT)
        if self.is_collision(new):
            self.reset() #game over...snake hit itself
        else:
            self.positions.insert(0,new)
            if len(self.positions) > self.length:
                self.positions.pop()
    
    def reset(self):
        self.length = 1
        self.positions = [(SCREEN_WIDTH/2, SCREEN_HEIGHT/2)]
        self.direction = random.choice([UP,DOWN,LEFT,RIGHT])
        self.score = 0
    
    def draw(self, surface):
        for p in self.positions:
            r = pygame.Rect((p[0],p[1]),(GRIDSIZE,GRIDSIZE))
            pygame.draw.rect(surface, self.color, r)
            pygame.draw.rect(surface, BG_COLOR_DARK, r, 1)
    
    def handle_keys(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.turn(UP)
                elif event.key == pygame.K_DOWN:
                    self.turn(DOWN)
                elif event.key == pygame.K_LEFT:
                    self.turn(LEFT)
                elif event.key == pygame.K_RIGHT:
                    self.turn(RIGHT)     
    
    def update_high_score(self):
        if self.score > self.high_score:
            self.high_score = self.score

class Food():
    
    def __init__(self):
        self.position = (0,0)
        self.color = FOOD_COLOR
        self.randomize_position()
    
    def randomize_position(self):
        self.position = (random.randint(1, GRID_WIDTH-2)*GRIDSIZE,
                         random.randint(1, GRID_HEIGHT -2)*GRIDSIZE)
    
    def draw(self, surface):
        r = pygame.Rect((self.position[0],self.position[1]),(GRIDSIZE,GRIDSIZE))
        pygame.draw.rect(surface, self.color, r)
        pygame.draw.rect(surface, BG_COLOR_DARK, r, 1)
    
# =========================================
# Main Loop 

def drawGrid(surface):
    for y in range(0, int(GRID_HEIGHT)):
        for x in range(0, int(GRID_WIDTH)):
            if x == 0 or x == GRID_WIDTH - 1 or y == 0 or y == GRID_HEIGHT -1:
                r = pygame.Rect((x*GRIDSIZE, y*GRIDSIZE), (GRIDSIZE, GRIDSIZE))
                pygame.draw.rect(surface, (255, 255, 255), r)
            elif (x+y) % 2 == 0:
                r = pygame.Rect((x*GRIDSIZE, y*GRIDSIZE), (GRIDSIZE, GRIDSIZE))
                pygame.draw.rect(surface, BG_COLOR_DARK, r)
            else:
                r = pygame.Rect((x*GRIDSIZE, y*GRIDSIZE), (GRIDSIZE, GRIDSIZE))
                pygame.draw.rect(surface, BG_COLOR_LIGHT, r)

    
def play_game():
    pygame.init()
    
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT), 0, 32)
    
    surface = pygame.Surface(screen.get_size())
    surface = surface.convert()
    drawGrid(surface)
    
    # begin game
    snake = Snake()
    food = Food()
    myfont = pygame.font.SysFont("terminal",20)
    
    while(True):
        clock.tick(10)
        # handle events
        snake.handle_keys()
        drawGrid(surface)
        snake.move()
        
        # check for food being eaten
        if snake.get_head_position() == food.position:
            #ate the food
            snake.length += 1
            snake.score += 1
            snake.update_high_score()
            food.randomize_position()
            
        
        # redraw screen
        snake.draw(surface)
        food.draw(surface)
        screen.blit(surface, (0,0))
        text = myfont.render(f'SCORE: {snake.score:0.0f}', 1, (0, 0, 0))
        screen.blit(text, (GRIDSIZE, 5))
        text = myfont.render(f'HIGH SCORE: {snake.high_score:0.0f}', 1, (0, 0, 0))
        screen.blit(text, (SCREEN_WIDTH/2, 5))
        pygame.display.update()
    

# play game
if __name__ == '__main__':
    play_game()
    
    