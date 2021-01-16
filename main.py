from network import Network
from pongaks import Pong
from plot import Plot
import plot
import numpy as np
import utils
import time


# Players: 'player', 'AI', 'trainer', 'rand'
player_A ='AI'
player_B ='AI'

pixel = 10
ball_size = 2
ball_base_speed = 2
court_size = (23 * ball_size, 21 * ball_size)
paddle_size = (ball_base_speed, ball_size * 5)
paddle_step = 3
game_base_speed = 20
max_score = 21


# Game & Networks
# ----------------------------------------------------------------------------------------------------------------------
game = Pong(pixel=pixel,
    court_size=court_size,
    ball_size=ball_size,
    ball_base_speed=ball_base_speed,
    paddle_size=paddle_size,
    paddle_step=paddle_step,
    game_base_speed=game_base_speed,
    max_score=max_score,
    player_A=player_A,
    player_B=player_B)

net_A = Network(name="A", 
    input_size=game.court_size[0]*game.court_size[1],
    hidden_size=400, 
    gamma=0.98,
    decay_rate=0.99,
    batch=10,
    learn_rate=1e-3,
    strategy='defense', 
    resume=True)
    
net_B = Network(name="B", 
    input_size=game.court_size[0]*game.court_size[1],
    hidden_size=200, 
    gamma=0.98,
    decay_rate=0.99,
    batch=10,
    learn_rate=1e-3,
    strategy='defense', 
    resume=True)   

plotting = True


# Auto configuration
# ----------------------------------------------------------------------------------------------------------------------
net_A.is_active = True if game.paddle_A.mode == 'AI' else False
net_B.is_active = True if game.paddle_B.mode == 'AI' else False
if game.paddle_A.mode != 'player' and game.paddle_B.mode != 'player':
    game.tick_freq = 4096

pause_time = 0
if game.paddle_A.mode == 'player' or game.paddle_B.mode == 'player':
    pause_time = 1


# Functions
# ----------------------------------------------------------------------------------------------------------------------
def determine_reward(net, bounced_me, bounced_enemy):
    reward = 0
    if net.strategy == "defense":
        reward = bounced_me
    if net.strategy == "defense-attack":
        reward = -1 if bounced_me == -1 else reward
        reward = 1 if bounced_enemy == -1 else reward     
    return reward
    

def process_frame(net, direction):
    x = None

    # if direction < 0:
    #     id0 = game.paddle_size[0]
    #     idE = game.court_size[0]
    #     net.curr_frame[idE-id0:idE, :] = 0

    # if direction > 0:
    #     idE = game.paddle_size[0]
    #     net.curr_frame[0:idE, :] = 0

    net.curr_frame = net.curr_frame.ravel()
    if net.prev_frame is not None:
        x = net.curr_frame - net.prev_frame
        #prev_frame = curr_frame

        # xPlot = x.reshape(-1, int(game.court_size[1]))
        # axs[1,1].imshow(xPlot.T, cmap='gray', interpolation='none')
        # plt.draw()

    net.prev_frame = net.curr_frame

    return x    


def push_net(net, direction, reward) :
    should_be_pushed = False
    is_ball_approaching = game.ball.velocity[0] * direction > 0 or reward != 0
    if net.strategy == 'defense-attack' or (net.strategy == 'defense' and is_ball_approaching):
        should_be_pushed = True

    action = 0
    if should_be_pushed:
        net.curr_frame = game.get_screen_frame()
        observation = process_frame(net, direction)

        if observation is not None:
            prob, hidden_state = net.policy_forward(observation)
            action = 2 if np.random.uniform() < prob else 3
            net.train(observation, hidden_state, prob, action, reward)

    else:
        net.prev_frame = None

    return action


# Main loop
# ----------------------------------------------------------------------------------------------------------------------
plot = Plot()
if plotting:
    figure = plot.prepare_plots(net_A, net_B)

while True:

    net_A.prepare()
    net_B.prepare()
    action_A = 0
    action_B = 0
    game.reset()
    time.sleep(pause_time)

    finished = False
    while not finished:

        score_A, score_B, bounced_A, bounced_B, finished = game.step(action_A, action_B)

        if bounced_A < 0 or bounced_B < 0:
            time.sleep(pause_time)

        if net_A.is_active:
            reward_A = determine_reward(net_A, bounced_me=bounced_A, bounced_enemy=bounced_B)
            action_A = push_net(net_A, -1, reward_A)

        if net_B.is_active:
            reward_B = determine_reward(net_B, bounced_me=bounced_B, bounced_enemy=bounced_A)
            action_B = push_net(net_B, +1, reward_B)

        if finished:
            if net_A.is_active:
                net_A.learn()
            if net_B.is_active:
                net_B.learn()

            if plotting:
                plot.update_plots(figure, net_A, net_B)

            time.sleep(pause_time)