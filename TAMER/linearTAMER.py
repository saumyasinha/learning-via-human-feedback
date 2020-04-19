import os
from itertools import count
from sys import stdout
import time
import numpy as np
import gym
import matplotlib
import matplotlib.pyplot as plt
import pygame
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn import pipeline, preprocessing

matplotlib.use("Agg")  # stops python crashing
pygame.init()

FONT = pygame.font.Font('freesansbold.ttf', 32)
ACTION_MAP = {0: 'left', 1: 'none', 2: 'right'}

# set position of pygame window (so it doesn't overlap with gym)
os.environ['SDL_VIDEO_WINDOW_POS'] = '1000,100'


def get_scalar_feedback(screen):
    """
    Get human input. 'W' key for positive, 'A' key for negative.
    Args:
        screen: pygame screen object

    Returns: scalar reward (1 for positive, -1 for negative)
    """
    reward = 0
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                screen.fill((0, 255, 0))
                reward = 1
                break
            elif event.key == pygame.K_a:
                screen.fill((255, 0, 0))
                reward = -1
                break
    pygame.display.flip()
    # print(reward)
    return reward


def show_action(screen, action):
    """
    Show agent's action on pygame screen
    Args:
        screen: pygame screen object
        action: numerical action (for MountainCar environment only currently)
    """
    screen.fill((0, 0, 0))
    pygame.display.flip()
    text = FONT.render(ACTION_MAP[action], True, (0, 255, 0))
    text_rect = text.get_rect()
    text_rect.center = (100, 50)
    screen.blit(text, text_rect)
    pygame.display.flip()


class LinearFunctionApproximator:
    def __init__(self, env):
        """
        SGD function approximator, with preprocessing steps from:
        https://github.com/dennybritz/reinforcement-learning/blob/master/FA/Q-Learning%20with%20Value%20Function%20Approximation%20Solution.ipynb
        """
        # Feature preprocessing: Normalize to zero mean and unit variance
        # We use a few samples from the observation space to do this
        observation_examples = np.array([env.observation_space.sample() for _ in range(10000)], dtype='float64')
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
        """ Returns the featurized representation for a state. """
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]


class TAMERAgent:
    """
    QLearning Agent adapted to TAMER using steps from:
    http://www.cs.utexas.edu/users/bradknox/kcap09/Knox_and_Stone,_K-CAP_2009.html
    """
    def __init__(self, env, discount_factor, epsilon, min_eps, num_episodes, tame=True, ts_len=0.2):

        if tame:
            self.H = LinearFunctionApproximator(env)  # init H function
        else:  # optionally run as standard Q Learning
            self.Q = LinearFunctionApproximator(env)  # init Q function

        self.tame = tame
        self.ts_len = ts_len  # length of timestep for training TAMER
        self.env = env

        # Hyperparameters
        self.discount_factor = discount_factor  # not used for TAMER
        self.epsilon = epsilon if not tame else 0  # no epsilon for TAMER
        self.num_episodes = num_episodes

        # Calculate episodic reduction in epsilon
        self.epsilon_step = (epsilon - min_eps) / num_episodes

        # Rewards
        self.reward_list = []

    def act(self, state):
        """ Epsilon-greedy Policy """
        if np.random.random() < 1 - epsilon:
            preds = self.H.predict(state) if self.tame else self.Q.predict(state)
            return np.argmax(preds)
        else:
            return np.random.randint(0, env.action_space.n)

    def train(self):
        # pygame display init
        screen = pygame.display.set_mode((200, 100))
        screen.fill((0, 0, 0))
        pygame.display.flip()

        for i in range(self.num_episodes):
            print(f'Episode: {i + 1}  Timestep:', end='')
            tot_reward = 0
            state = self.env.reset()

            for ts in count():
                print(f' {ts}', end='')
                self.env.render()  # render env

                # Determine next action
                action = self.act(state)
                show_action(screen, action)

                # Get next state and reward
                next_state, reward, done, info = self.env.step(action)

                if self.tame:
                    time.sleep(self.ts_len)
                    human_reward = get_scalar_feedback(screen)
                    if human_reward != 0:
                        self.H.update(state, action, human_reward)
                else:
                    if done and next_state[0] >= 0.5:
                        td_target = reward
                    else:
                        td_target = reward + self.discount_factor * np.max(self.Q.predict(next_state))
                    self.Q.update(state, action, td_target)

                tot_reward += reward

                if done:
                    print(f'  Reward: {tot_reward}')
                    break

                stdout.write('\b' * (len(str(ts)) + 1))
                state = next_state

            # Decay epsilon
            if self.epsilon > min_eps:
                self.epsilon -= self.epsilon_step

        self.env.close()

    def play(self):
        """ Run an episode with trained agent """
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

    env = gym.make('MountainCar-v0')

    # hyperparameters
    discount_factor = 1
    epsilon = 0  # vanilla Q learning actually works well with no random exploration
    min_eps = 0
    num_episodes = 1
    tame = True  # set to false for vanilla Q learning

    # set a timestep for training TAMER
    # the more time per step, the easier for the human but the longer it takes to train (in real time)
    # 0.2 seconds is fast but doable
    tamer_training_timestep = 0.2  # seconds

    agent = TAMERAgent(env, discount_factor, epsilon, min_eps, num_episodes, tame, tamer_training_timestep)
    agent.train()
    agent.play()
