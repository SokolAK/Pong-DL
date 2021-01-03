from random import randint
import utils
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from keras.initializers import RandomNormal

class Network():

    def __init__(self, name, inputSize, hiddenSize, gamma, mode, batch, cont):
        self.gameNo = 0
        self.name = name
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.gamma = gamma
        self.mode = mode
        self.batch = batch
        self.model = None
        self.rewardBalance = 0
        self.totalRewards = 0
        self.fileName = f"keras_n{name}_i{inputSize}_h{hiddenSize}_g{gamma}_m{mode}_b{batch}"

        if cont:
            try:
                self.model = load_model(self.fileName)
            except:
                print(f"File '{self.fileName}' does not exist! Preparing new model...")

        if self.model == None:
            self.model = Sequential()
            self.model.add(Dense(units=hiddenSize, input_dim=inputSize, activation='relu', kernel_initializer='glorot_uniform'))
            self.model.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal'))
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    def prepare(self) :
        self.xTrain = []
        self.yTrain = []
        self.rewards = []
        self.rewardsWin = 0
        self.rewardsLoss = 0
        self.scoreMine = 0
        self.scoreEnemy = 0


    def predict(self,x) :
        return self.model(np.expand_dims(x, axis=1).T)
        #return self.model.predict(np.expand_dims(x, axis=1).T)


    def push(self, x, scoreMine, scoreEnemy, reward) :
        proba = self.predict(x)
        y = 1 if np.random.uniform() < proba else 0 # 1 - up, 0 - down

        self.xTrain.append(x)
        self.yTrain.append(y)
        
        self.rewards.append(reward)
        if reward > 0 :
            self.rewardsWin += 1
        if reward < 0 :
            self.rewardsLoss += 1

        self.scoreMine = scoreMine
        self.scoreEnemy = scoreEnemy

        return y


    def log(self):
        try:
            lastLine = ""
            with open(f"log_{self.name}.txt", 'r') as f:
                for line in f:
                    pass
                lastLine = line.split()
                gameNo = lastLine[0]
                totalRewards = lastLine[1]
        except:
            gameNo = self.gameNo
            totalRewards = self.totalRewards

        self.gameNo = int(gameNo) + 1
        self.totalRewards = int(totalRewards) + self.rewardsWin + self.rewardsLoss
        self.rewardBalance = self.rewardsWin - self.rewardsLoss
        
        #with open(f"log_{self.name}.txt", 'a') as f:
        #    f.write(f"{self.gameNo} {self.totalRewards} {self.rewardBalance}\n")

    def learn(self):
        self.log()

        print("-----------------------------------------------------------------------------------------")
        print(f"Network: {self.name} -> Game no.: {self.gameNo}")
        print(f"Total number of rewards: {self.totalRewards} -> Reward balance: {self.rewardBalance}", end = '')
        self.model.fit(x=np.vstack(self.xTrain), y=np.vstack(self.yTrain), verbose=1, sample_weight=utils.discount_rewards(self.rewards, self.gamma))
        self.model.save(self.fileName)

