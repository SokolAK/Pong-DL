import pygame
from random import randint
from numpy import sign, ceil
 
BLACK = (0, 0, 0)
 
class Ball(pygame.sprite.Sprite):
    def __init__(self, color, diameter, speed):
        super().__init__()
        
        self.bounceParams = []
        self.speedX = speed
        self.speedY = speed / 2
        #self.velocity = [self.speedX, self.speedY * randint(-4, 4)]
        self.velocity = [self.speedX, 0]
        self.diameter = diameter
        self.image = pygame.Surface([diameter, diameter])
        self.image.fill(BLACK)
        self.image.set_colorkey(BLACK)
 
        pygame.draw.rect(self.image, color, [0, 0, diameter, diameter])
        self.rect = self.image.get_rect()
        

    def update(self, courtSize):
        self.rect.x += self.velocity[0]
        self.rect.y += self.velocity[1]
        # if self.rect.x < 0:
        #     self.rect.x = 0 
        # if self.rect.x > courtSize[0] - self.diameter:
        #     self.rect.x = courtSize[0] - self.diameter
        if self.rect.y < 0:
            self.rect.y = 0 
        if self.rect.y > courtSize[1] - self.diameter:
            self.rect.y = courtSize[1] - self.diameter


    def calc_y_velocity(self, paddle):
        ballY = self.get_y_position()
        paddleY = paddle.get_y_position()
        paddleH = paddle.height

        diff = ceil((ballY - paddleY) / (self.diameter)) * self.speedY

        if self.velocity[1] != 0 :
            return sign(self.velocity[1]) * abs(diff)
        elif diff != 0:
            return diff
        else :
           return self.speedY * randint(-1, 1)


    def bounce_horizontal(self) :
        self.velocity[1] = -self.velocity[1]   


    def bounce_goal(self) :
        self.velocity[0] = -self.velocity[0]
        if self.velocity[1] == 0 :
            self.velocity[1] = self.speedY * randint(-1, 1)


    def bounce_paddle(self, direction, paddle) :
        self.rect.x -= self.velocity[0]
        self.rect.y -= self.velocity[1]
        self.velocity[0] = direction * abs(self.velocity[0])
        self.velocity[1] = self.calc_y_velocity(paddle)


    #def updateBounceParams(self) :
    #    self.bounceParams = [self.getYPosition(), self.velocity[1]]


    def get_x_position(self):
        return self.rect.x + int(self.diameter/2)


    def get_y_position(self):
        return self.rect.y + self.diameter/2


    def get_position(self):
        return [get_x_position(), get_y_position()]