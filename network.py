from random import randint
from utils import *
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

class Network():

    def __init__(self, name, courtSize):
        self.gamesNo = 0
        self.name = name
        self.model = Sequential()
        self.model.add(Dense(units=100,input_dim=courtSize[0]*courtSize[1], activation='relu', kernel_initializer='glorot_uniform'))
        self.model.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.gamma = 0.99


    def prepare(self) :
        self.xTrain = []
        self.yTrain = []
        self.rewards = []
        self.rewardsWin = 0
        self.rewardsLoss = 0
        self.scoreMine = 0
        self.scoreEnemy = 0


    def predict(self,x) :
        return self.model.predict(np.expand_dims(x, axis=1).T)


    def push(self, x, scoreMine, scoreEnemy) :
        #proba = model.predict(np.expand_dims(x, axis=1).T)
        proba = self.predict(x)
        y = 1 if np.random.uniform() < proba else 0 # 1 - up, 0 - down

        # log the input and label to train later
        self.xTrain.append(x)
        self.yTrain.append(y)


        reward = scoreMine - self.scoreMine + self.scoreEnemy - scoreEnemy
        self.rewards.append(reward)
        if reward > 0 :
            self.rewardsWin += 1
        if reward < 0 :
            self.rewardsLoss += 1

        self.scoreMine = scoreMine
        self.scoreEnemy = scoreEnemy
        # with open("frame.txt", 'w') as file:
        #     for row in frame:
        #         for item in row:
        #             file.write(f"{item} ")
        #         file.write("\n")
        return y


    def discount_rewards(self, r, gamma):
        """ take 1D float array of rewards and compute discounted reward """
        r = np.array(r)
        discounted_r = np.zeros_like(r)
        running_add = 0
        # we go from last reward to first one so we don't have to do exponentiations
        for t in reversed(range(0, r.size)):
            if r[t] != 0: 
                running_add = 0 # if the game ended (in Pong), reset the reward sum
            running_add = running_add * gamma + r[t] # the point here is to use Horner's method to compute those rewards efficiently
            discounted_r[t] = running_add
        discounted_r = discounted_r - np.mean(discounted_r) #normalizing the result
        discounted_r /= np.std(discounted_r) #idem
        return discounted_r


    def learn(self):
        self.gamesNo += self.scoreMine + self.scoreEnemy
        print("------------------------------------------------------\nGames no.: ", self.gamesNo)
        print(self.name + ': The total reward was:', self.rewardsWin - self.rewardsLoss, end='')
        #episode_nb += 1
        self.model.fit(x=np.vstack(self.xTrain), y=np.vstack(self.yTrain), verbose=1, sample_weight=self.discount_rewards(self.rewards, self.gamma))