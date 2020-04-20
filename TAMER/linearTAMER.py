from itertools import count
from sys import stdout
from pathlib import Path
import pickle
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn import pipeline, preprocessing

MOUNTAINCAR_ACTION_MAP = {0: "left", 1: "none", 2: "right"}
MODELS_DIR = Path(__file__).parent.joinpath('models')


class LinearFunctionApproximator:
    def __init__(self, env):
        """
        SGD function approximator, with preprocessing steps from:
        https://github.com/dennybritz/reinforcement-learning/blob/master/FA/Q-Learning%20with%20Value%20Function%20Approximation%20Solution.ipynb
        """
        # Feature preprocessing: Normalize to zero mean and unit variance
        # We use a few samples from the observation space to do this
        observation_examples = np.array(
            [env.observation_space.sample() for _ in range(10000)], dtype="float64"
        )
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        # Used to convert a state to a featurized represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        self.featurizer = pipeline.FeatureUnion(
            [
                ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
                ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=100)),
            ]
        )
        self.featurizer.fit(self.scaler.transform(observation_examples))

        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
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

    def __init__(
        self,
        env,
        discount_factor,
        epsilon,
        min_eps,
        num_episodes,
        tame=True,
        ts_len=0.2,
        model_file_to_load=None  # filename of pretrained model
    ):
        self.tame = tame
        self.ts_len = ts_len  # length of timestep for training TAMER
        self.env = env

        # init model
        if model_file_to_load is not None:
            print(f'Loaded pretrained model: {model_file_to_load}')
            self.load_model(filename=model_file_to_load)
        else:
            if tame:
                self.H = LinearFunctionApproximator(env)  # init H function
            else:  # optionally run as standard Q Learning
                self.Q = LinearFunctionApproximator(env)  # init Q function

        # Hyperparameters
        self.discount_factor = discount_factor  # not used for TAMER
        self.epsilon = epsilon if not tame else 0  # no epsilon for TAMER
        self.num_episodes = num_episodes
        self.min_eps = min_eps

        # Calculate episodic reduction in epsilon
        self.epsilon_step = (epsilon - min_eps) / num_episodes

        # Reward logging
        self.reward_log = pd.DataFrame(columns=['Episode', 'Ep start ts','Feedback ts', 'Reward'])

    def act(self, state):
        """ Epsilon-greedy Policy """
        if np.random.random() < 1 - self.epsilon:
            preds = self.H.predict(state) if self.tame else self.Q.predict(state)
            return np.argmax(preds)
        else:
            return np.random.randint(0, self.env.action_space.n)

    def train(self, model_file_to_save=None, input_protocol='wait'):
        """
        TAMER (or Q learning) training loop
        There are 2 ways to configure inputs for TAMER:
            'wait': python sleeps for ts_len then grabs the first input
                    from the pygame queue. This seems to train better agents
                    but feels laggy and sometimes inputs don't register
            'loop': python loops for ts_len listening for the first input. This
                    feels much smoother to play but seems to train worse.
        Will continue testing and pick one eventually.
        Args:
            model_file_to_save: save Q or H model to this filename
            input_protocol: 'wait' or 'loop'
        """
        if self.tame:
            # only init pygame display if we're actually training tamer
            matplotlib.use("Agg")  # stops python crashing
            from .interface import Interface
            disp = Interface(action_map=MOUNTAINCAR_ACTION_MAP)

        for i in range(self.num_episodes):
            print(f"Episode: {i + 1}  Timestep:", end="")
            tot_reward = 0
            ep_start_time = pd.datetime.now().time()
            state = self.env.reset()

            for ts in count():
                print(f" {ts}", end="")
                self.env.render()  # render env

                # Determine next action
                action = self.act(state)
                if self.tame:
                    disp.show_action(action)

                # Get next state and reward
                next_state, reward, done, info = self.env.step(action)

                if self.tame:
                    if input_protocol == 'wait':
                        time.sleep(self.ts_len)
                        human_reward = disp.get_scalar_feedback()
                        if human_reward != 0:
                            self.H.update(state, action, human_reward)
                    elif input_protocol == 'loop':
                        now = time.time()
                        while time.time() < now + self.ts_len:
                            time.sleep(0.01)
                            human_reward = disp.get_scalar_feedback()
                            if human_reward != 0:
                                self.H.update(state, action, human_reward)
                                break
                else:
                    if done and next_state[0] >= 0.5:
                        td_target = reward
                    else:
                        td_target = reward + self.discount_factor * np.max(
                            self.Q.predict(next_state)
                        )
                    self.Q.update(state, action, td_target)

                tot_reward += reward

                if done:
                    print(f"  Reward: {tot_reward}")
                    break

                stdout.write("\b" * (len(str(ts)) + 1))
                state = next_state

            # Decay epsilon
            if self.epsilon > self.min_eps:
                self.epsilon -= self.epsilon_step

        self.env.close()
        if model_file_to_save is not None:
            self.save_model(filename=model_file_to_save)

    def play(self, n_episdoes=1, render=False):
        """
        Run episodes with trained agent
        Args:
            n_episdoes: number of episodes
            render: optionally render episodes

        Returns: list of cumulative episode rewards
        """
        self.epsilon = 0
        ep_rewards = []
        for i in range(n_episdoes):
            state = self.env.reset()
            done = False
            tot_reward = 0
            while not done:
                action = self.act(state)
                next_state, reward, done, info = self.env.step(action)
                tot_reward += reward
                if render:
                    self.env.render()
                state = next_state
            ep_rewards.append(tot_reward)
            print(f'Episode: {i + 1} Reward: {tot_reward}')
        self.env.close()

        return ep_rewards

    def evaluate(self, n_episdoes=100):
        print('Evaluating agent')
        rewards = self.play(n_episdoes=n_episdoes)
        avg_reward = np.mean(rewards)
        print(f'Average total episode reward over {n_episdoes} '
              f'episodes: {avg_reward:.2f}')
        return avg_reward

    def save_model(self, filename):
        """
        Save H or Q model to models dir
        Args:
            filename: name of pickled file
        """
        model = self.H if self.tame else self.Q
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, filename):
        """
        Load H or Q model from models dir
        Args:
            filename: name of pickled file
        """
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'rb') as f:
            model = pickle.load(f)
        if self.tame:
            self.H = model
        else:
            self.Q = model
