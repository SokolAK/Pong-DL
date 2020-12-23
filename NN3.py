import numpy as np

#hiddenLayersNodes = [17*500]
#output = []

itera = 0
hiddenSize = 0
outputSize = 0
alpha = 1e-3
weights_0_1 = []
weights_1_2 = []



def tanh(x):
    return np.tanh(x)
def tanh2deriv(output):
    return 1 - (output ** 2)

def relu(x):
    return (x >= 0) * x
def relu2deriv(output):
    return output >= 0

def act(x):
    return tanh(x)
def act2deriv(output):
    return tanh2deriv(output)



def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)


def init(size, paddleHeight, ball) :
    global hiddenSize, outputSize, weights_0_1, weights_1_2

    #hiddenLayerSize = (2*ball.maxSpeed+1) * size[1]
    hiddenLayerSize = 128
    outputSize = int(size[1] / paddleHeight)


    weights_0_1 = 2 * alpha * np.random.random((2, hiddenLayerSize)) - alpha
    weights_1_2 = 2 * alpha * np.random.random((hiddenLayerSize, outputSize)) - alpha


def predict(paddle, ball) :
    # if len(ball.bounceParams) > 0 :
    #     layer_0 = [ball.bounceParams[0], ball.bounceParams[1]]
    #     layer_1 = tanh(np.dot(layer_0,weights_0_1))
    #     layer_2 = np.dot(layer_1,weights_1_2)
    #     return layer_2
    return 0


def learn(paddle, ball) :
    if len(ball.bounceParams) > 0 :
        global itera
        itera += 1

        global weights_0_1, weights_1_2, outputSize

        #goal_pred = int(np.ceil((ball.getYPosition() - paddle.getYPosition())/paddle.step))
        goal_pred = np.zeros(outputSize)
        range = int(ball.getYPosition() / paddle.height)
        goal_pred[range] = 1

        layer_0 = np.array([[ball.bounceParams[0], ball.bounceParams[1]]])
        layer_1 = act(np.dot(layer_0,weights_0_1))
        dropout_mask = np.random.randint(2, size=layer_1.shape)
        #layer_1 *= dropout_mask * 2
        layer_2 = softmax(np.dot(layer_1,weights_1_2))


        maxpos = np.argmax(layer_2)
        cor = False
        if goal_pred[maxpos] > 0:
            cor = True

        #print(weights_0_1)
        #print(weights_1_2)
        print(layer_0[0], layer_2[0], goal_pred, cor)

        layer_2_delta = (layer_2 - goal_pred)/outputSize
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * act2deriv(layer_1)
        #layer_1_delta *= dropout_mask

        weights_1_2 -= alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 -= alpha * layer_0.T.dot(layer_1_delta)
