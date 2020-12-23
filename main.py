# Import the pygame library and initialise the game engine
import pygame
from paddle import Paddle
from ball import Ball
from numpy import ceil
import NN3 as NN

mode = 1

pygame.init()
 
BLACK = (0,0,0)
WHITE = (255,255,255)
 
size = (700, 500)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Pong DL")

paddleWidth = 10
paddleHeight = size[1]/5
paddleStep = 10
 
paddleA = Paddle(WHITE, paddleWidth, paddleHeight, paddleStep)
paddleA.rect.x = 0
paddleA.rect.y = (size[1] - paddleA.height)/2

paddleB = Paddle(WHITE, paddleWidth, paddleHeight, paddleStep)
paddleB.rect.x = size[0] - paddleB.width
paddleB.rect.y = (size[1] - paddleB.height)/2
 
ball = Ball(WHITE, size[1]/50, 10)
ball.rect.x = (size[0] - ball.diameter)/2
ball.rect.y = (size[1] - ball.diameter)/2
 
all_sprites_list = pygame.sprite.Group()
 
all_sprites_list.add(paddleA)
all_sprites_list.add(paddleB)
all_sprites_list.add(ball)
 
carryOn = True
 
clock = pygame.time.Clock()
 
scoreA = 0
scoreB = 0

if mode == 0 :
    paddleA.mode = "AI"
    paddleB.mode = "AI"
if mode == 1 :
    paddleA.mode = "AI"
    paddleB.mode = "player"
if mode == 2 :
    paddleA.mode = "player"
    paddleB.mode = "player"
if mode > 0 :
    NN.init(size, paddleHeight, ball)

def displayScores():
    font = pygame.font.Font(None, 74)
    text = font.render(str(scoreA), 1, WHITE)
    screen.blit(text, (size[0] / 4 - 74/2.6, 10))
    text = font.render(str(scoreB), 1, WHITE)
    screen.blit(text, (size[0] / 4 * 3, 10))


def refresh():
    all_sprites_list.update()
    screen.fill(BLACK)
    pygame.draw.line(screen, WHITE, [size[0]/2-2, 0], [size[0]/2-2, size[1]], 4)
    all_sprites_list.draw(screen) 

    displayScores()
    pygame.display.flip()
    clock.tick(60 * 10)	


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
            paddleA.moveDown(size[1])
    if paddleB.mode == "player" :
        if keys[pygame.K_UP]:
            paddleB.moveUp()
        if keys[pygame.K_DOWN]:
            paddleB.moveDown(size[1]) 


def performGoalBounce(paddle) :
    AIlearn(ball, paddleA)
    ball.velocity[0] = -ball.velocity[0]
    ball.updateBounceParams()
    AIpredict(ball, paddleA)


def detectWallBounce():
    global scoreA, scoreB

    if ball.rect.x <= 0:
        performGoalBounce(paddleA)
        scoreB += 1

    if ball.rect.x >= size[0] - ball.diameter:
        performGoalBounce(paddleB)
        scoreA += 1

    if ball.rect.y > size[1] - ball.diameter or ball.rect.y < 0:
        ball.velocity[1] = -ball.velocity[1]     


def AIlearn(ball, paddle):
    global paddleA, paddleB
    if(paddleA.mode == "AI" or paddleB.mode == "AI"):
        NN.learn(paddle, ball)

def AIpredict(ball, paddle):
    if(paddleA.mode == "AI"):
        paddleA.prediction = NN.predict(paddle, ball) 
    if(paddleB.mode == "AI"):
        paddleB.prediction = NN.predict(paddle, ball) 


def detectPaddleBounce():
    if pygame.sprite.collide_mask(ball, paddleA):
        AIlearn(ball, paddleA)
        ball.bouncePaddle(+1, ball, paddleA)
        AIpredict(ball, paddleB)
    if pygame.sprite.collide_mask(ball, paddleB):
        AIlearn(ball, paddleB)
        ball.bouncePaddle(-1, ball, paddleB)
        AIpredict(ball, paddleA)


def movePaddle(paddle):
    # if paddle.prediction > 0:
    #     paddle.moveDown(size[1])
    #     paddle.prediction -= 1
    # if paddle.prediction < 0:
    #     paddle.moveUp()
    #     paddle.prediction += 1

    if paddle.prediction > paddle.getYPosition():
        paddle.moveDown(size[1])
    if paddle.prediction < paddle.getYPosition():
        paddle.moveUp()



def moveAIpaddles():
    if paddleA.mode == "AI":
        movePaddle(paddleA)
    if paddleB.mode == "AI":
        movePaddle(paddleB)



# -------- Main Program Loop -----------
while carryOn:
    checkUserAction()
    detectWallBounce()
    detectPaddleBounce()

    moveAIpaddles()

    refresh()

pygame.quit()