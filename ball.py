import pygame
import random
from numpy import sign, ceil
 
BLACK = (0, 0, 0)
 
class Ball(pygame.sprite.Sprite):
    def __init__(self, color, pixel, diameter_factor, speed_factor):
        super().__init__()
        
        self.bounceParams = []
        self.pixel = pixel
        self.diameter = diameter_factor * pixel

        self.speed = speed_factor * pixel
        self.speed_x = self.speed * random.choice([-1, 1])
        self.max_speed_y = int(self.speed_x * 1.5)
        self.velocity = [self.speed_x, 0]

        self.image = pygame.Surface([self.diameter, self.diameter])
        self.image.fill(BLACK)
        self.image.set_colorkey(BLACK)
        pygame.draw.rect(self.image, color, [0, 0, self.diameter, self.diameter])
        self.rect = self.image.get_rect()

        self.is_served = False
        self.serving_factor = 1 / pixel
        

    def serve(self):
        self.multiply_speeds(self.serving_factor, False)
        self.is_served = True     


    def play(self):
        if self.is_served == True:
            self.multiply_speeds(1 / self.serving_factor, True)
            self.is_served = False


    def multiply_speed(self, speed, factor, rounding):
        speed = int(speed * factor)
        if rounding:
            speed = round(speed / self.pixel) * self.pixel
        return speed


    def multiply_speeds(self, factor, rounding):
        self.speed = self.multiply_speed(self.speed, factor, rounding)
        self.speed_x = self.multiply_speed(self.speed_x, factor, rounding)
        self.max_speed_y = self.multiply_speed(self.max_speed_y, factor, rounding)
        self.velocity[0] = self.multiply_speed(self.velocity[0], factor, rounding)
        self.velocity[1] = self.multiply_speed(self.velocity[1], factor, rounding)


    def update(self, court_width, court_height):
        self.rect.x += self.velocity[0]
        self.rect.y += self.velocity[1]
        # if self.rect.left < 0:
        #     self.rect.left = 0 
        # if self.rect.right > court_width:
        #     self.rect.right = court_width
        if self.rect.top < 0:
            self.rect.top = 0 
        if self.rect.bottom > court_height:
            self.rect.bottom = court_height


    def _rand_y_velocity(self):
        return int(self.max_speed_y/self.pixel * random.uniform(-1, 1)) * self.pixel


    def _calc_y_velocity(self, paddle):
        ballY = self.rect.centery
        paddleY = paddle.rect.centery
        paddleH = paddle.height

        #diff = ceil((ballY - paddleY) / (self.diameter)) * self.speedY
        diff = int((ballY - paddleY) * 2 / (paddleH + self.diameter) * self.max_speed_y/self.pixel) * self.pixel

        if self.velocity[1] != 0 :
            return sign(self.velocity[1]) * abs(diff)
        elif diff != 0:
            return diff
        else :
           return self._rand_y_velocity()


    def bounce_horizontal(self) :
        self.velocity[1] = -self.velocity[1]   


    def bounce_goal(self) :
        self.velocity[0] = -self.velocity[0]
        if self.velocity[1] == 0 :
            self.velocity[1] = self._rand_y_velocity()
        self.play()


    def bounce_paddle(self, direction, paddle) :
        self.rect.x -= self.velocity[0]
        self.rect.y -= self.velocity[1]
        self.velocity[0] = direction * abs(self.velocity[0])
        self.velocity[1] = self._calc_y_velocity(paddle)
        self.play()
