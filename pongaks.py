# Import the pygame library and initialise the game engine
import pygame
from paddle import Paddle
from ball import Ball
import utils
import numpy as np
import matplotlib.pyplot as plt
import time
import random


class Pong():
    def __init__(self, pixel, court_size, ball_size, ball_base_speed, paddle_size, paddle_step, max_score, game_base_speed, player_A, player_B):
        self.pixel = pixel
        self.court_size = court_size
        self.court_width = court_size[0] * pixel
        self.court_height = court_size[1] * pixel
        self.ball_size = ball_size
        self.ball_base_speed = ball_base_speed
        self.paddle_size = paddle_size
        self.paddle_width = paddle_size[0] * pixel
        self.paddle_height = paddle_size[1] * pixel
        self.paddle_step = paddle_step * pixel
        self.max_score = max_score
        self.tick_freq = game_base_speed
        
        self.BLACK = (0,0,0)
        self.WHITE = (255,255,255)
        self.RED = (255, 0, 0)

        self.ball = Ball(self.WHITE, self.pixel, ball_size, ball_base_speed)
        #self.court_size = (self.court_width, self.court_height)

        self.score_board_height = int(self.court_height * 0.2)
        self.display_size = (self.court_width, self.court_height + self.score_board_height)

        self.screen = pygame.display.set_mode(self.display_size, vsync=1)
        self.court_line_width = 2

        self.paddle_A = Paddle(self.WHITE, self.paddle_width, self.paddle_height, self.court_height, self.paddle_step)
        self.paddle_A.rect.left = 0
        self.paddle_A.rect.y = (self.court_height - self.paddle_A.height)/2
        self.paddle_A.mode = player_A

        self.paddle_B = Paddle(self.WHITE, self.paddle_width, self.paddle_height, self.court_height, self.paddle_step)
        self.paddle_B.rect.right = self.court_width
        self.paddle_B.rect.y = (self.court_height - self.paddle_B.height)/2
        self.paddle_B.mode = player_B

        self.all_sprites_list = pygame.sprite.Group()
        self.all_sprites_list.add(self.paddle_A)
        self.all_sprites_list.add(self.paddle_B)
        self.all_sprites_list.add(self.ball)

        self.clock = pygame.time.Clock()

        pygame.init()
        pygame.display.set_caption("Pong-DL")


    def display_scores(self):
        interline_size = self.pixel
        font_size = int((self.score_board_height - interline_size * 3) / 2)
        font = pygame.font.Font(None, font_size)

        #Recalculate font size to adjust the text to score board height
        f_width, f_height = font.size(self.paddle_A.mode + self.paddle_B.mode)
        font_size *= font_size / f_height
        font_size = int(font_size)
        font = pygame.font.Font(None, font_size)

        label_A = self.paddle_A.mode
        label_A_width, label_A_height = font.size(label_A)
        label = font.render(label_A, 1, self.WHITE)
        self.screen.blit(label, (self.court_width / 4 - label_A_width / 2, self.court_height + interline_size))
        score_A_width, score_A_height = font.size(str(self.score_A))
        score = font.render(str(self.score_A), 1, self.WHITE)
        self.screen.blit(score, (self.court_width / 4 - score_A_width / 2, self.court_height + interline_size * 2 + label_A_height))

        label_B = self.paddle_B.mode
        label_B_width, label_B_height = font.size(label_B)
        label = font.render(label_B, 1, self.WHITE)
        self.screen.blit(label, (self.court_width / 4 * 3 - label_B_width / 2, self.court_height + interline_size))
        score_B_width, score_B_height = font.size(str(self.score_B))
        score = font.render(str(self.score_B), 1, self.WHITE)
        self.screen.blit(score, (self.court_width / 4 * 3 - score_B_width / 2, self.court_height + interline_size * 2 + label_B_height))


    def update_sprites(self) :
        #all_sprites_list.update()
        self.paddle_A.update()
        self.paddle_B.update()
        self.ball.update(self.court_width, self.court_height)


    def detect_horizontal_bounce(self):
        if self.ball.rect.bottom >= self.court_height or self.ball.rect.top <= 0:
            self.ball.bounce_horizontal()   


    def detect_vertical_bounce_A(self):
        #print(self.ball.rect.left)
        if self.ball.rect.left < self.paddle_width - 1:
            if pygame.sprite.collide_rect(self.ball, self.paddle_A) and self.ball.velocity[0] < 0:
                self.ball.bounce_paddle(+1, self.paddle_A)
                self.ball.update(self.court_width, self.court_height)
                return 1
            elif self.ball.rect.left < 1:
                self.ball.bounce_goal()
                self.score_B += 1
                return -1
        return 0


    def detect_vertical_bounce_B(self):
        if self.ball.rect.right > self.court_width - self.paddle_width - 1:
            if pygame.sprite.collide_rect(self.ball, self.paddle_B) and self.ball.velocity[0] > 0:
                self.ball.bounce_paddle(-1, self.paddle_B)
                self.ball.update(self.court_width, self.court_height)
                return 1
            elif self.ball.rect.right > self.court_width - 1:
                self.ball.bounce_goal()
                self.score_A += 1
                return -1
        return 0


    def move_paddle(self, paddle, action):
        if paddle.mode == "AI":
            if action == 2:
                paddle.move_up()
            if action == 3:
                paddle.move_down() 

        if paddle.mode == "trainer":
            if self.ball.rect.centery + self.ball.velocity[1] < paddle.rect.top  :
                paddle.move_up()
            if self.ball.rect.centery + self.ball.velocity[1] > paddle.rect.bottom :
                paddle.move_down() 

        if paddle.mode == "rand":
            paddle.rect.centery = int(random.uniform(self.paddle_height / 2, self.court_height - self.paddle_height / 2))


    def move_AI_paddles(self, action_A, action_B):
        self.move_paddle(self.paddle_A, action_A)
        self.move_paddle(self.paddle_B, action_B)


    def check_user_action(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_KP_MINUS:
                    self.tick_freq = max(int(self.tick_freq / 1.2), 1)
                    print(f"Game speed: {self.tick_freq}")
                if event.key == pygame.K_KP_PLUS:
                    self.tick_freq = min(int(np.ceil(self.tick_freq * 1.2)), 8192)
                    print(f"Game speed: {self.tick_freq}")

        keys = pygame.key.get_pressed()
        if self.paddle_A.mode == "player" :
            if keys[pygame.K_w]:
                self.paddle_A.move_up()
            if keys[pygame.K_s]:
                self.paddle_A.move_down()
        if self.paddle_B.mode == "player" :
            if keys[pygame.K_UP]:
                self.paddle_B.move_up()
            if keys[pygame.K_DOWN]:
                self.paddle_B.move_down() 


    def get_screen_frame(self) :
        frame = pygame.surfarray.pixels_red(self.screen)[:,0:self.court_height]/255
        frameReducedX = utils.sum_chunk(frame, self.pixel) / self.pixel
        frame = utils.sum_chunk(frameReducedX, self.pixel, axis=0) / self.pixel
        return frame


    def refresh_screen(self):
        self.screen.fill(self.BLACK)
        pygame.draw.line(self.screen, self.WHITE, [0, self.court_height + self.court_line_width / 2 - 1], [self.court_width, self.court_height + self.court_line_width / 2 - 1], self.court_line_width)
        self.all_sprites_list.draw(self.screen) 
        self.display_scores()
        pygame.display.flip()
        self.clock.tick(self.tick_freq)


    def reset(self):
        self.ball.rect.center = (self.court_width / 2 , self.court_height / 2)
        self.ball.velocity = [self.ball.speed_x, 0]
        self.score_A = 0
        self.score_B = 0
        self.ball.serve()
        self.refresh_screen()


    def step(self, action_A, action_B):
        self.check_user_action()
        self.move_AI_paddles(action_A, action_B)

        self.update_sprites()
        # self.paddle_A.update()
        # self.paddle_B.update()
        # self.ball.update(self.court_width, self.court_height)

        bounced_A = self.detect_vertical_bounce_A()
        bounced_B = self.detect_vertical_bounce_B()

        self.detect_horizontal_bounce()

        self.refresh_screen()

        finished = False
        if self.score_A == self.max_score or self.score_B == self.max_score:
            finished = True

        return self.score_A, self.score_B, bounced_A, bounced_B, finished