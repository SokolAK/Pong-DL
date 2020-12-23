import numpy as np


size = [700,500]

alpha0 = 1e-3
alpha1 = alpha0/700
weights = [670, 1]


def init(numberOfPositions) :
    pass

def predict(paddle, ball) :
    if len(ball.bounceParams) > 0 :

        print(ball.getYPosition())

        goal = goal_pred = ball.bounceParams[1] * 700 + ball.bounceParams[0]
        layer_0 = [ball.bounceParams[1], ball.bounceParams[0]]
        pred = weights[0] * layer_0[0] + weights[1] * layer_0[1]
        
        pos = int(abs(int(pred)) % size[1])
        if pred < 0 :
            pos = size[1] - pos

        moves = (pos - paddle.getYPosition())/paddle.step
        print (ball.bounceParams[1], ball.bounceParams[0], " -> ", pos)
        return int(moves)


    return 0


def learn(paddle, ball) :
    if len(ball.bounceParams) > 0 :

        global weights


        #goal_pred = ball.bounceParams[1] * 700 + ball.bounceParams[0]
        goal_pred = ball.getYPosition()

        layer_0 = [ball.bounceParams[1], ball.bounceParams[0]]
        pred = weights[0] * layer_0[0] + weights[1] * layer_0[1]

        error = (pred - goal_pred) ** 2
        delta = pred - goal_pred
        weight_deltas = [delta*layer_0[0], delta*layer_0[1]]
        weights[0] -= alpha0 * weight_deltas[0]
        weights[1] -= alpha1 * weight_deltas[1]

        #print(weights, pred, goal_pred, error)
