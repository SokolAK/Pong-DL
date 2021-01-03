# Import the pygame library and initialise the game engine
import pygame
from network import Network
from paddle import Paddle
from ball import Ball
import utils
import numpy as np
import matplotlib.pyplot as plt
import time
from gym_test import test
np.set_printoptions(threshold=np.inf)

pygame.init()
pygame.display.set_caption("Pong-DL")

playerA = "AI"
playerB = "AI"
 
maxScore = 21

BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (255, 0, 0)

pixel = 5
ball = Ball(RED, pixel * 2, pixel * 1)

courtSize = (15 * ball.diameter, 17 * ball.diameter)
scoreBoardHeight = pixel * 10
displaySize = (courtSize[0], courtSize[1] + scoreBoardHeight)
screen = pygame.display.set_mode(displaySize)
courtLineWidth = 2
tickFreq = 0

paddleWidth = pixel
paddleHeight = ball.diameter * 3
paddleStep = pixel * 3
 
paddleA = Paddle(WHITE, paddleWidth, paddleHeight, paddleStep)
paddleA.rect.x = 0
paddleA.rect.y = (courtSize[1] - paddleA.height)/2
paddleA.mode = playerA

paddleB = Paddle(WHITE, paddleWidth, paddleHeight, paddleStep)
paddleB.rect.x = courtSize[0] - paddleB.width
paddleB.rect.y = (courtSize[1] - paddleB.height)/2
paddleB.mode = playerB
 
if paddleA.mode == "AI":
    networkA = Network(name='A200', inputSize=int(courtSize[0]/pixel)*int(courtSize[1]/pixel), hiddenSize=200, gamma=1, mode='defense', batch='half', cont=True)
if paddleB.mode == "AI":
    networkB = Network(name='B200', inputSize=int(courtSize[0]/pixel)*int(courtSize[1]/pixel), hiddenSize=200, gamma=1, mode='defense', batch='half', cont=True)

all_sprites_list = pygame.sprite.Group()
all_sprites_list.add(paddleA)
all_sprites_list.add(paddleB)
all_sprites_list.add(ball)

 
clock = pygame.time.Clock()
 

def display_scores():
    fontSize = int(scoreBoardHeight / 2)
    font = pygame.font.Font(None, fontSize)

    labelA = paddleA.mode
    labelAWidth, labelAHeight = font.size(labelA)
    label = font.render(labelA, 1, WHITE)
    screen.blit(label, (courtSize[0] / 4 - labelAWidth / 2, courtSize[1] + pixel))
    scoreAWidth, scoreAHeight = font.size(str(scoreA))
    score = font.render(str(scoreA), 1, WHITE)
    screen.blit(score, (courtSize[0] / 4 - scoreAWidth / 2, courtSize[1] + pixel * 2 + labelAHeight))

    labelB = paddleB.mode
    labelBWidth, labelBHeight = font.size(labelB)
    label = font.render(labelB, 1, WHITE)
    screen.blit(label, (courtSize[0] / 4 * 3 - labelBWidth / 2, courtSize[1] + pixel))
    scoreBWidth, scoreBHeight = font.size(str(scoreB))
    score = font.render(str(scoreB), 1, WHITE)
    screen.blit(score, (courtSize[0] / 4 * 3 - scoreBWidth / 2, courtSize[1] + pixel * 2 + labelBHeight))


def update_sprites() :
    #all_sprites_list.update()
    paddleA.update()
    paddleB.update()
    ball.update(courtSize)


def get_screen_frame() :
    frame = pygame.surfarray.pixels_red(screen)[:,0:courtSize[1]]/255
    frameReducedX = utils.sum_chunk(frame, pixel) / pixel
    frame = utils.sum_chunk(frameReducedX, pixel, axis=0) / pixel
    
    currFrame = frame.ravel()
    global prevFrame
    x = currFrame - prevFrame if prevFrame is not None else np.ones(len(currFrame))
    prevFrame = currFrame
    
    #xPlot = x.reshape(-1, int(courtSize[1]/pixel))
    # xPlot = pygame.surfarray.pixels_red(screen)[:,0:courtSize[1]]/255
    # ax3.imshow(xPlot.T, cmap='hot', interpolation='none')
    # plt.draw()

    return x


def proceed_NN(paddle, network, direction, scoreMine, scoreEnemy, reward) :
    #if ball.velocity[0] * direction > 0  or reward != 0:
    #if (network.mode == 'defense' and (ball.velocity[0] * direction > 0  or reward != 0)) or network.mode == 'defense-attack':
    ballApproaching = ball.velocity[0] * direction > 0  or reward != 0
    if network.batch == 'full' or (network.batch == 'half' and ballApproaching) :
        f = get_screen_frame()
        paddle.prediction = network.push(f, scoreMine, scoreEnemy, reward)
    else:
        paddle.prediction = -1


def refresh_screen():
    screen.fill(BLACK)
    #pygame.draw.line(screen, WHITE, [courtSize[0]/2-courtLineWidth/2, 0], [courtSize[0]/2-courtLineWidth/2, displaySize[1]], courtLineWidth)
    pygame.draw.line(screen, WHITE, [0, courtSize[1] + courtLineWidth/2 - 1], [courtSize[0], courtSize[1] + courtLineWidth/2 - 1], courtLineWidth)
    all_sprites_list.draw(screen) 
    display_scores()
    pygame.display.flip()
    #pygame.display.update()
    clock.tick(tickFreq)


def check_user_action():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    keys = pygame.key.get_pressed()
    if paddleA.mode == "player" :
        if keys[pygame.K_w]:
            paddleA.move_up()
        if keys[pygame.K_s]:
            paddleA.move_down(courtSize[1])
    if paddleB.mode == "player" :
        if keys[pygame.K_UP]:
            paddleB.move_up()
        if keys[pygame.K_DOWN]:
            paddleB.move_down(courtSize[1]) 


def detect_horizontal_bounce():
    if ball.rect.y >= courtSize[1] - ball.diameter or ball.rect.y <= 0:
        ball.bounce_horizontal()   


def detect_vertical_bounce_A():
    global scoreB
    if ball.rect.x < paddleWidth:
        yCondition1 = ball.rect.y > paddleA.rect.y and ball.rect.y < paddleA.rect.y + paddleHeight
        yCondition2 = ball.rect.y + ball.diameter > paddleA.rect.y and ball.rect.y + ball.diameter < paddleA.rect.y + paddleHeight
        if yCondition1 or yCondition2:
            ball.bounce_paddle(+1, paddleA)
            ball.update(courtSize)
            return 1
        else:
            ball.bounce_goal()
            scoreB += 1
            return -1
    return 0


def detect_vertical_bounce_B():
    global scoreA
    if ball.rect.x + ball.diameter > courtSize[0] - paddleWidth:
        yCondition1 = ball.rect.y > paddleB.rect.y and ball.rect.y < paddleB.rect.y + paddleHeight
        yCondition2 = ball.rect.y + ball.diameter > paddleB.rect.y and ball.rect.y + ball.diameter < paddleB.rect.y + paddleHeight
        if yCondition1 or yCondition2:
            ball.bounce_paddle(-1, paddleB)
            ball.update(courtSize)
            return 1
        else:
            ball.bounce_goal()
            scoreA += 1
            return -1
    return 0


def move_paddle(paddle):
    if paddle.mode == "AI":
        if paddle.prediction == 1 :
            paddle.move_up()
        if paddle.prediction == 0 :
            paddle.move_down(courtSize[1]) 
    if paddle.mode == "trainer":
        if ball.get_y_position() + ball.velocity[1] < paddle.rect.y  :
            paddle.move_up()
        if ball.get_y_position() + ball.velocity[1] > paddle.rect.y + paddleHeight :
            paddle.move_down(courtSize[1]) 


def move_AI_paddles():
    move_paddle(paddleA)
    move_paddle(paddleB)


# -------- Main Program Loop -----------
plt.style.use('dark_background')
f, (ax1, ax2) = plt.subplots(2, 1)
plt.show(block=False)

while True:

    ball.rect.x = (courtSize[0] - ball.diameter)/2
    ball.rect.y = (courtSize[1] - ball.diameter)/2
    scoreA = 0
    scoreB = 0
    currFrame = None
    prevFrame = None

    if paddleA.mode == "AI" :
        networkA.prepare()
    if paddleB.mode == "AI" :
        networkB.prepare()

    refresh_screen()

    while scoreA < maxScore and scoreB < maxScore :
        check_user_action()
        move_AI_paddles()
        #update_sprites()
        paddleA.update()
        paddleB.update()

        ball.update(courtSize)
        bouncedA = detect_vertical_bounce_A()
        bouncedB = detect_vertical_bounce_B()

        detect_horizontal_bounce()

        refresh_screen()

        if paddleA.mode == "AI" :
            if networkA.mode == "defense":
                reward = bouncedA
            else:
                reward = scoreA - networkA.scoreMine - (scoreB - networkA.scoreEnemy)
            proceed_NN(paddleA, networkA, -1, scoreA, scoreB, reward)

        if paddleB.mode == "AI" :
            if networkB.mode == "defense":
                reward = bouncedB
            else:
                reward = scoreB - networkB.scoreMine - (scoreA - networkB.scoreEnemy)
            proceed_NN(paddleB, networkB, +1, scoreB, scoreA, reward)

    if paddleA.mode == "AI" :
        networkA.learn()
        ax1.bar(networkA.gameNo, networkA.rewardBalance, 1, color='tab:blue')
        plt.draw()
    if paddleB.mode == "AI" :
        networkB.learn()
        ax2.bar(networkB.gameNo, networkB.rewardBalance, 1, color='tab:orange')
        plt.draw()


pygame.quit()