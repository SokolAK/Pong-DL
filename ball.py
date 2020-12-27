import pygame
from random import randint
from numpy import sign, ceil
 
BLACK = (0, 0, 0)
 
class Ball(pygame.sprite.Sprite):
    def __init__(self, color, diameter, maxSpeed):
        super().__init__()
        
        self.bounceParams = []
        self.maxXSpeed = maxSpeed
        self.maxYSpeed = maxSpeed
        self.diameter = diameter
        self.image = pygame.Surface([diameter, diameter])
        self.image.fill(BLACK)
        self.image.set_colorkey(BLACK)
 
        pygame.draw.rect(self.image, color, [0, 0, diameter, diameter])
        
        self.velocity = [self.maxXSpeed, self.maxYSpeed/2*randint(-4, 4)]
        
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


    def calcYvelocity(self, paddle):
        ballY = self.getYPosition()
        paddleY = paddle.getYPosition()
        paddleH = paddle.height

        diff = int((ballY - paddleY) / (self.diameter)) * 5

        if self.velocity[1] != 0 :
            return sign(self.velocity[1]) * abs(diff)
        elif diff != 0:
            return diff
        else :
           return self.maxYSpeed/2*randint(-4, 4)


    def bouncePaddle(self, direction, paddle) :
        self.velocity[0] = direction * abs(self.velocity[0])
        self.velocity[1] = self.calcYvelocity(paddle)
        self.updateBounceParams()


    def updateBounceParams(self) :
        self.bounceParams = [self.getYPosition(), self.velocity[1]]


    def getXPosition(self):
        return self.rect.x + int(self.diameter/2)


    def getYPosition(self):
        return self.rect.y + self.diameter/2


    def getPosition(self):
        return [getXPosition(), getYPosition()]