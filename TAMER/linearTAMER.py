"""
Some code adapted from:
https://github.com/dennybritz/reinforcement-learning/blob/master/FA/Q-Learning%20with%20Value%20Function%20Approximation%20Solution.ipynb
"""

from itertools import count
from sys import stdout
from time import sleep
import numpy as np
import gym
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn import pipeline, preprocessing

matplotlib.use("Agg")  # stops python crashing


class LinearH:
    def __init__(self, env):

        # Feature Preprocessing: Normalize to zero mean and unit variance
        # We use a few samples from the observation space to do this
        observation_examples = np.array([env.observation_space.sample() for _ in range(10000)])
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        # Used to convert a state to a featurized represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        self.featurizer = pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
        self.featurizer.fit(self.scaler.transform(observation_examples))

        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate='constant')
            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)

    def predict(self, state, action=None):
        features = self.featurize_state(state)
        if not action:
            return [m.predict([features])[0] for m in self.models]
        else:
            return self.models[action].predict([features])[0]

    def update(self, state, action, td_target):
        features = self.featurize_state(state)
        self.models[action].partial_fit([features], [td_target])

    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]


class TAMERAgent:
    def __init__(self, env, discount_factor, epsilon, min_eps, num_episodes, ignore_terminal_states=False):

        self.Q = LinearH(env)  # init Q function
        self.env = env

        # Hyperparameters
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.num_episodes = num_episodes

        # Funnily enough, ignoring terminal states may make it train better
        # This seems counter intuitive but the real reward signal here seems to be how quickly you can
        # reset to the start state
        # I've given an option to ignore terminal states for hyperparamater search
        self.ignore_terminal_states = ignore_terminal_states

        # Calculate episodic reduction in epsilon
        self.epsilon_step = (epsilon - min_eps) / num_episodes

        # Rewards
        self.reward_list = []

    def act(self, state):
        """ Epsilon-greedy Policy? """
        if np.random.random() < 1 - epsilon:
            return np.argmax(self.Q.predict(state))
        else:
            return np.random.randint(0, env.action_space.n)

    def train(self):

        # Run Q learning algorithm
        for i in range(self.num_episodes):
            print(f'Episode: {i + 1}  Timestep:', end='')
            tot_reward = 0
            state = self.env.reset()

            for ts in count():
                print(f' {ts}', end='')
                # Render environment
                self.env.render()
                # sleep(0.1)

                # Determine next action
                action = self.act(state)

                # Get next state and reward
                next_state, reward, done, info = self.env.step(action)

                if done and next_state[0] >= 0.5 and self.ignore_terminal_states:
                    td_target = reward
                else:
                    td_target = reward + discount_factor * np.max(self.Q.predict(next_state))
                # print(td_target)

                self.Q.update(state, action, td_target)

                # Update variables
                tot_reward += reward
                if done:
                    print(f'  Reward: {tot_reward}')
                    break
                else:
                    stdout.write('\b' * (len(str(ts)) + 1))
                    state = next_state

            # Decay epsilon
            if self.epsilon > min_eps:
                self.epsilon -= self.epsilon_step

        self.env.close()

    def play(self):
        self.epsilon = 0
        state = self.env.reset()
        done = False
        while not done:
            action = self.act(state)
            next_state, reward, done, info = self.env.step(action)
            self.env.render()
            state = next_state
        self.env.close()


if __name__ == "__main__":

    env = gym.make("MountainCar-v0")

    # hyperparameters
    discount_factor = 1
    epsilon = 0  # actually works well with no random exploration
    min_eps = 0
    num_episodes = 2
    ignore_terminal_states = False

    agent = TAMERAgent(
        env, discount_factor, epsilon, min_eps, num_episodes, ignore_terminal_states
    )
    agent.train()
    agent.play()
