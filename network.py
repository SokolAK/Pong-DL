import utils
import pickle
import numpy as np

class Network():
    def __init__(self, name, input_size, hidden_size, gamma, decay_rate, batch, learn_rate, strategy, resume):
        self.file_name = f"nets/net_{name}_h{hidden_size}"

        loaded = False
        if resume:
            loaded = self.load_net()

        if not loaded:
            self.name = name
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.gamma = gamma
            self.decay_rate = decay_rate
            self.batch = batch
            self.learn_rate = learn_rate
            self.strategy = strategy
            self.resume = resume
            self.is_active = True
            self.episode_no = 0
            self.running_reward = None
            self.grad_buffer = None
            self.rmsprop_cache = None
            self.history = {'episode_no':[], 'reward_sum':[], 'running_reward':[]}
            self.prepare_model()
            self.prepare()

        self.batch = batch

    def load_net(self):
        try:
            self.__dict__  = pickle.load(open(self.file_name, 'rb'))
            print(f"Loading neural network from file: {self.file_name}")
            return True
        except Exception as e:
            print(e)
            print("Creating new neural network...")
            return False


    def prepare_model(self):
        self.model = {}
        self.model['W1'] = np.random.randn(self.hidden_size, self.input_size) / np.sqrt(self.input_size) # "Xavier" initialization
        self.model['W2'] = np.random.randn(self.hidden_size) / np.sqrt(self.hidden_size)


    def prepare(self):
        self.reset()
        if self.episode_no % self.batch == 0:
            self.reward_sum = 0

        if self.grad_buffer is None:
            self.grad_buffer = { k : np.zeros_like(v) for k,v in self.model.items() } # update buffers that add up gradients over a batch
        if self.rmsprop_cache is None:
            self.rmsprop_cache = { k : np.zeros_like(v) for k,v in self.model.items() } # rmsprop memory


    def reset(self):
        self.x_train = []
        self.h_train = [] # hidden state
        self.dlogps_train = [] # grad that encourages the action that was taken to be taken
        self.r_train = []
        self.curr_frame = None
        self.prev_frame = None


    def train(self, x, h, prob, action, reward):
        self.x_train.append(x) # observation
        self.h_train.append(h) # hidden state
        y = 1 if action == 2 else 0 # a "fake label"
        self.dlogps_train.append(y - prob) # grad that encourages the action that was taken to be taken
        self.r_train.append(reward) # record reward
        self.reward_sum += reward


    def policy_forward(self, x):
        h = np.dot(self.model['W1'], x)
        h[h < 0] = 0 # ReLU nonlinearity
        logp = np.dot(self.model['W2'], h)
        p = utils.sigmoid(logp)
        return p, h # return probability of taking action 2, and hidden state


    def policy_backward(self, epx, eph, epdlogp):
        """ backward pass. (eph is array of intermediate hidden states) """
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, self.model['W2'])
        dh[eph <= 0] = 0 # backpro prelu
        dW1 = np.dot(dh.T, epx)
        return {'W1':dW1, 'W2':dW2}


    def learn(self):
        self.episode_no += 1
        epx = np.vstack(self.x_train)
        eph = np.vstack(self.h_train)
        epdlogp = np.vstack(self.dlogps_train)
        epr = np.vstack(self.r_train)

        # compute the discounted reward backwards through time
        discounted_epr = utils.discount_rewards(epr, self.gamma)

        epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
        grad = self.policy_backward(epx, eph, epdlogp)
        for k in self.model:
            self.grad_buffer[k] += grad[k] # accumulate grad over batch

        #self.running_reward = self.reward_sum if self.running_reward is None else self.running_reward * 0.99 + self.reward_sum * 0.01
        print(f"Network {self.name}: episode {self.episode_no} -> reward sum: {self.reward_sum}")

        if self.episode_no % self.batch == 0:
            self.update_model()
            return True
        return False


    def update_model(self):
        for k,v in self.model.items():
            g = self.grad_buffer[k] # gradient
            self.rmsprop_cache[k] = self.decay_rate * self.rmsprop_cache[k] + (1 - self.decay_rate) * g**2
            self.model[k] += self.learn_rate * g / (np.sqrt(self.rmsprop_cache[k]) + 1e-5)
            #self.grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

        #curr_factor = 1 / (9 + 1)
        #prev_factor = 1 - curr_factor
        #self.running_reward = self.reward_sum if self.running_reward is None else self.running_reward * prev_factor + self.reward_sum * curr_factor
        self.history['episode_no'].append(self.episode_no)
        self.history['reward_sum'].append(self.reward_sum)
        
        history_length = len(self.history['reward_sum'])
        self.running_reward = np.average(self.history['reward_sum'][-min(history_length,10):]) / self.batch
        self.history['running_reward'].append(self.running_reward)

        print(f"Network {self.name}: update -> running reward: {self.running_reward}")

        self.reset()
        self.grad_buffer = None
        self.rmsprop_cache = None
        pickle.dump(self.__dict__, open(self.file_name, 'wb'))
