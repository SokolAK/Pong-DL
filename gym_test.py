# import necessary modules from keras
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import gym
import utils
import matplotlib.pyplot as plt


def test():
    # creates a generic neural network architecture
    model = Sequential()

    # hidden layer takes a pre-processed frame as input, and has 200 units
    model.add(Dense(units=200,input_dim=80*80, activation='relu', kernel_initializer='glorot_uniform'))

    # output layer
    model.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal'))

    # compile the model using traditional Machine Learning losses and optimizers
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    env = gym.make("Pong-v0")
    observation = env.reset()
    prev_input = None

    # Macros
    UP_ACTION = 2
    DOWN_ACTION = 3

    # Hyperparameters
    gamma = 0.99

    # initialization of variables used in the main loop
    x_train, y_train, rewards = [],[],[]
    reward_sum = 0
    episode_nb = 0

    while True:
        # preprocess the observation, set input as difference between images
        cur_input = utils.prepro(observation)
        x = cur_input - prev_input if prev_input is not None else np.zeros(80 * 80)
        prev_input = cur_input
        
        # forward the policy network and sample action according to the proba distribution
        proba = model(np.expand_dims(x, axis=1).T)
        action = UP_ACTION if np.random.uniform() < proba else DOWN_ACTION
        y = 1 if action == 2 else 0 # 0 and 1 are our labels

        # log the input and label to train later
        x_train.append(x)
        y_train.append(y)

        # do one step in our environment
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        reward_sum += reward

        # end of an episode
        if done:
            print('\nAt the end of episode', episode_nb, 'the total reward was :', reward_sum, end='')
            
            # increment episode number
            episode_nb += 1
            
            # training
            model.fit(x=np.vstack(x_train), y=np.vstack(y_train), verbose=1, sample_weight=utils.discount_rewards(rewards, gamma))
                                                                 
            # Reinitialization
            x_train, y_train, rewards = [],[],[]
            observation = env.reset()
            reward_sum = 0
            prev_input = None

            with open(f"log_gym_test.txt", 'a') as f:
                f.write(f"{episode_nb} {reward_sum}\n")