import numpy as np

def sum_chunk(x, chunk_size, axis=-1):
    shape = x.shape
    if axis < 0:
        axis += x.ndim
    shape = shape[:axis] + (-1, chunk_size) + shape[axis+1:]
    x = x.reshape(shape)
    return x.sum(axis=axis+1)

def sigmoid(x):
    return 1/(1 + np.exp(-x))
def sigmoid2deriv(output):
    return output*(1-output)

def tanh(x):
    return np.tanh(x)
def tanh2deriv(output):
    return 1 - (output ** 2)

def relu(x):
    return (x >= 0) * x
def relu2deriv(output):
    return output >= 0

def act(x):
    return relu(x)
def act2deriv(output):
    return relu2deriv(output)

def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)


def discount_rewards(r, gamma):
    r = np.array(r)
    discounted_r = np.zeros_like(r, dtype=float)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: 
            running_add = 0 
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    discounted_r = discounted_r - np.mean(discounted_r)
    discounted_r /= np.std(discounted_r)
    return discounted_r


def discount_rewards_aks(r, gamma):
    r = np.array(r)
    discounted_r = np.zeros_like(r, dtype=float)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: 
            running_add = 0 
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()