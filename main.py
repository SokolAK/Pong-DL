# Import the pygame library and initialise the game engine
import pygame
from network import Network
from paddle import Paddle
from ball import Ball
from utils import *
import numpy as np
import NN3 as NN
np.set_printoptions(threshold=np.inf)

pygame.init()
 
BLACK = (0,0,0)
WHITE = (255,255,255)
 
courtSize = (200, 200)
displaySize = (200, 250)
screen = pygame.display.set_mode(displaySize, vsync=1)
pygame.display.set_caption("Pong-DL")

ball = Ball(WHITE, courtSize[1]/20, 10)
#ball.rect.x = 190
#ball.rect.y = 190
# ball.rect.x = ((courtSize[0] / (ball.diameter/2)) / 2 - 1) * (ball.diameter/2)
# ball.rect.y = ((courtSize[1] / (ball.diameter/2)) / 2 - 1) * (ball.diameter/2)

paddleWidth = ball.diameter/2
paddleHeight = ball.diameter*3
paddleStep = (courtSize[1]-paddleHeight)/2/10
 
paddleA = Paddle(WHITE, paddleWidth, paddleHeight, paddleStep)
paddleA.rect.x = 0
paddleA.rect.y = (courtSize[1] - paddleA.height)/2

paddleB = Paddle(WHITE, paddleWidth, paddleHeight, paddleStep)
paddleB.rect.x = courtSize[0] - paddleB.width
paddleB.rect.y = (courtSize[1] - paddleB.height)/2
 

courtLineWidth = 2
tickFreq = 600
 
all_sprites_list = pygame.sprite.Group()
 
all_sprites_list.add(paddleA)
all_sprites_list.add(paddleB)
all_sprites_list.add(ball)
 
carryOn = True
 
clock = pygame.time.Clock()
 
scoreA = 0
scoreB = 0

paddleA.mode = "AI"
paddleB.mode = "AI"
NN.init(courtSize, paddleHeight, ball)


def displayScores():
    font = pygame.font.Font(None, 50)
    text = font.render(str(scoreA), 1, WHITE)
    screen.blit(text, (courtSize[0] / 4 - 50/2, courtSize[1] + 15))
    text = font.render(str(scoreB), 1, WHITE)
    screen.blit(text, (courtSize[0] / 4 * 3, courtSize[1] + 15))


def updateSprites() :
    #all_sprites_list.update()
    paddleA.update()
    paddleB.update()
    ball.update(courtSize)


def NNaction() :
    frame = pygame.surfarray.pixels_red(screen)[:,0:courtSize[1]]/255
    frameReducedX = sum_chunk(frame, 5)/5
    frame = sum_chunk(frameReducedX, 5, axis=0)/5
    #frame = reversed(frame.T)
    currFrame = frame.ravel()
    global prevFrame
    x = currFrame - prevFrame if prevFrame is not None else np.zeros(len(currFrame))
    prevFrame = currFrame

    if paddleA.mode == "AI" :
        paddleA.prediction = networkA.push(x, scoreA, scoreB)
    if paddleB.mode == "AI" :
        paddleB.prediction = networkB.push(x, scoreB, scoreA)


def refreshScreen():
    screen.fill(BLACK)
    #pygame.draw.line(screen, WHITE, [courtSize[0]/2-courtLineWidth/2, 0], [courtSize[0]/2-courtLineWidth/2, displaySize[1]], courtLineWidth)
    pygame.draw.line(screen, WHITE, [0, courtSize[1] + courtLineWidth/2 - 1], [courtSize[0], courtSize[1] + courtLineWidth/2 - 1], courtLineWidth)
    all_sprites_list.draw(screen) 
    displayScores()
    #pygame.display.flip()
    pygame.display.update()

    if paddleA.mode == "AI" or paddleB.mode == "AI" :
        NNaction()

    clock.tick(tickFreq)


def checkUserAction():
    global carryOn
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            carryOn = False

    keys = pygame.key.get_pressed()
    if paddleA.mode == "player" :
        if keys[pygame.K_w]:
            paddleA.moveUp()
        if keys[pygame.K_s]:
            paddleA.moveDown(courtSize[1])
    if paddleB.mode == "player" :
        if keys[pygame.K_UP]:
            paddleB.moveUp()
        if keys[pygame.K_DOWN]:
            paddleB.moveDown(courtSize[1]) 


def performGoalBounce(paddle) :
    ball.velocity[0] = -ball.velocity[0]
    ball.updateBounceParams()


def detectWallBounce():
    global scoreA, scoreB

    if ball.rect.x <= 0:
        performGoalBounce(paddleA)
        scoreB += 1

    if ball.rect.x >= courtSize[0] - ball.diameter:
        performGoalBounce(paddleB)
        scoreA += 1

    if ball.rect.y >= courtSize[1] - ball.diameter or ball.rect.y <= 0:
        ball.velocity[1] = -ball.velocity[1]     


def detectPaddleBounce():
    #if pygame.sprite.collide_mask(ball, paddleA):
    if ball.rect.x == paddleWidth and ball.rect.y + ball.diameter >= paddleA.rect.y and ball.rect.y <= paddleA.rect.y + paddleHeight :
        ball.bouncePaddle(+1, paddleA)
    #if pygame.sprite.collide_mask(ball, paddleB):
    if ball.rect.x + ball.diameter == courtSize[0] - paddleWidth and ball.rect.y + ball.diameter >= paddleB.rect.y and ball.rect.y <= paddleB.rect.y + paddleHeight :
        ball.bouncePaddle(-1, paddleB)


def movePaddle(paddle):
    if paddle.mode == "AI":
        if paddle.prediction == 1 :
            paddle.moveUp()
        if paddle.prediction == 0 :
            paddle.moveDown(courtSize[1]) 
    if paddle.mode == "rand":
        if np.random.uniform() < 1 :
            if paddle.getYPosition() > ball.getYPosition() :
                paddle.moveUp()
                # print("UP")
            else :
                paddle.moveDown(courtSize[1]) 
                # print("DOWN")  


def moveAIpaddles():
    movePaddle(paddleA)
    movePaddle(paddleB)


# -------- Main Program Loop -----------
networkA = Network('A', [int(courtSize[0]/5), int(courtSize[1]/5)])
networkB = Network('B', [int(courtSize[0]/5), int(courtSize[1]/5)])

while carryOn:

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

    refreshScreen()

    while scoreA < 21 and scoreB < 21 and carryOn:
        updateSprites()

        checkUserAction()
        detectWallBounce()
        detectPaddleBounce()

        moveAIpaddles()

        refreshScreen()

    # updateSprites()
    # refreshScreen()

    if paddleA.mode == "AI" :
        networkA.learn()
    if paddleB.mode == "AI" :
        networkB.learn()


pygame.quit()