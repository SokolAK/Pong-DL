import pygame
from random import randint
from numpy import sign, ceil
 
BLACK = (0, 0, 0)
 
class Ball(pygame.sprite.Sprite):
    def __init__(self, color, diameter, maxSpeed):
        super().__init__()
        
        self.bounceParams = []
        self.maxXSpeed = maxSpeed
        self.maxYSpeed = 1
        self.diameter = diameter
        self.image = pygame.Surface([diameter, diameter])
        self.image.fill(BLACK)
        self.image.set_colorkey(BLACK)
 
        pygame.draw.rect(self.image, color, [0, 0, diameter, diameter])
        
        self.velocity = [self.maxXSpeed, randint(-self.maxYSpeed, self.maxYSpeed)]
        
        self.rect = self.image.get_rect()
        

    def update(self):
        self.rect.x += self.velocity[0]
        self.rect.y += self.velocity[1]
      

    def calcYvelocity(self, ball, paddle):
        return -self.velocity[1]
        ballY = ball.getYPosition()
        paddleY = paddle.getYPosition()
        paddleH = paddle.height
        diff = ceil(self.maxYSpeed * 2 * (ballY - paddleY) / paddleH)

        if self.velocity[1] != 0 :
            return sign(self.velocity[1]) * abs(diff)
        elif diff != 0:
            return diff
        else :
           return randint(-self.maxYSpeed, self.maxYSpeed)


    def bouncePaddle(self, direction, ball, paddle) :
        self.velocity[0] = direction * abs(self.velocity[0])
        self.velocity[1] = self.calcYvelocity(ball, paddle)
        self.updateBounceParams()


    def updateBounceParams(self) :
        self.bounceParams = [self.getYPosition(), self.velocity[1]]


    def getXPosition(self):
        return self.rect.x + int(self.diameter/2)


    def getYPosition(self):
        return self.rect.y + self.diameter/2


    def getPosition(self):
        return [getXPosition(), getYPosition()]