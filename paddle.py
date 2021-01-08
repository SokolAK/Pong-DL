import pygame
from random import randint

BLACK = (0,0,0)
 
class Paddle(pygame.sprite.Sprite):

    def __init__(self, color, width, height, court_height, step):
        super().__init__()

        self.mode = "player"
        self.step = step
        self.prediction = 0
        self.width = width
        self.height = height
        self.image = pygame.Surface([width, height])
        self.image.fill(BLACK)
        self.image.set_colorkey(BLACK)
        self.court_height = court_height
 
        pygame.draw.rect(self.image, color, [0, 0, width, height])
        self.rect = self.image.get_rect()
        

    def move_up(self):
        self.rect.y -= self.step
        if self.rect.y < 0:
          self.rect.y = 0
          

    def move_down(self):
        self.rect.y += self.step 
        maxY = self.court_height - self.height
        if self.rect.y > maxY:
          self.rect.y = maxY