import datetime as dt
import os
import pickle
import time
import uuid
from itertools import count
from pathlib import Path
from sys import stdout
from csv import DictWriter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import pipeline, preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from FaceClassifier.predict import prediction

MOUNTAINCAR_ACTION_MAP = {0: "left", 1: "none", 2: "right"}
MODELS_DIR = Path(__file__).parent.joinpath("models")
LOGS_DIR = Path(__file__).parent.joinpath("logs")


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
        output_dir=LOGS_DIR,
        face_classifier_path=None,
        model_file_to_load=None,  # filename of pretrained model
    ):
        self.tame = tame
        self.ts_len = ts_len  # length of timestep for training TAMER
        self.env = env
        self.uuid = uuid.uuid4()
        self.output_dir = output_dir
        self.face_classifier_path = face_classifier_path

        # init model
        if model_file_to_load is not None:
            print(f"Loaded pretrained model: {model_file_to_load}")
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
        self.reward_log_columns = [
            "Episode",
            "Ep start ts",
            "Feedback ts",
            "Human Reward",
            "Environment Reward",
        ]
        self.reward_log_path = os.path.join(self.output_dir, f"{self.uuid}.csv")

    def act(self, state):
        """ Epsilon-greedy Policy """
        if np.random.random() < 1 - self.epsilon:
            preds = self.H.predict(state) if self.tame else self.Q.predict(state)
            return np.argmax(preds)
        else:
            return np.random.randint(0, self.env.action_space.n)

    def _train_episode(self, episode_index, disp, rec=None):
        print(f"Episode: {episode_index + 1}  Timestep:", end="")
        rng = np.random.default_rng()
        tot_reward = 0
        state = self.env.reset()
        ep_start_time = dt.datetime.now().time()
        with open(self.reward_log_path, "a+", newline="") as write_obj:
            dict_writer = DictWriter(write_obj, fieldnames=self.reward_log_columns)
            dict_writer.writeheader()
            for ts in count():
                print(f" {ts}", end="")
                self.env.render()

                # Determine next action
                action = self.act(state)
                if self.tame:
                    disp.show_action(action)

                # Get next state and reward
                next_state, reward, done, info = self.env.step(action)

                if not self.tame:
                    if done and next_state[0] >= 0.5:
                        td_target = reward
                    else:
                        td_target = reward + self.discount_factor * np.max(
                            self.Q.predict(next_state)
                        )
                    self.Q.update(state, action, td_target)
                else:
                    now = time.time()
                    while time.time() < now + self.ts_len:
                        frame = None
                        if rec is not None:
                            frame = rec.get_frame()
                            rec.show_frame(frame)
                            rec.write_frame(frame)

                        time.sleep(0.01)  # save the CPU

                        human_reward = disp.get_scalar_feedback()
                        feedback_ts = dt.datetime.now().time()
                        if human_reward != 0:
                            if rec is not None:
                                face_reward = self.get_reward_from_frame(frame)
                                rec.write_frame_image(frame, str(feedback_ts))
                            dict_writer.writerow(
                                {
                                    "Episode": episode_index + 1,
                                    "Ep start ts": ep_start_time,
                                    "Feedback ts": feedback_ts,
                                    "Human Reward": human_reward,
                                    "Environment Reward": reward
                                }
                            )
                            ## vanilla Tamer training
                            # self.H.update(state, action, human_reward)

                            ## Tamer training for Experiment A
                            self.H.update(state, action, face_reward)

                            ## Tamer training for Experiment B
                            # self.H.update(state, action, face_reward + human_reward)
                            break
                        else:
                            # Sometimes save a frame without human feedback
                            # TODO: choose a better or dynamic probability
                            prob_save = 0.005
                            if rng.random() < prob_save:
                                dict_writer.writerow(
                                    {
                                        "Episode": episode_index + 1,
                                        "Ep start ts": ep_start_time,
                                        "Feedback ts": feedback_ts,
                                        "Human Reward": 0,
                                        "Environment Reward": reward,
                                    }
                                )
                                if rec is not None:
                                    rec.write_frame_image(frame, str(feedback_ts))
                                break

                tot_reward += reward

                if done:
                    print(f"  Reward: {tot_reward}")
                    break

                stdout.write("\b" * (len(str(ts)) + 1))
                state = next_state

        # Decay epsilon
        if self.epsilon > self.min_eps:
            self.epsilon -= self.epsilon_step

    async def train(self, model_file_to_save=None, capture_video=False):
        """
        TAMER (or Q learning) training loop
        Args:
            model_file_to_save: save Q or H model to this filename
            capture_video: whether or not to capture webcam feed and save frames
        """
        # render first so that pygame display shows up on top
        self.env.render()
        disp = None
        if self.tame:
            # only init pygame display if we're actually training tamer
            matplotlib.use("Agg")  # stops python crashing
            from .interface import Interface

            disp = Interface(action_map=MOUNTAINCAR_ACTION_MAP)

        try:
            if capture_video:
                from VideoCap.videocap import RecordFromWebCam

                with RecordFromWebCam(self.uuid, self.output_dir) as rec:
                    for i in range(self.num_episodes):
                        self._train_episode(i, disp, rec)
            else:
                for i in range(self.num_episodes):
                    self._train_episode(i, disp)
        finally:
            print()
            print("Cleaning up linearTAMER")
            self.env.close()
            if model_file_to_save is not None:
                self.save_model(filename=model_file_to_save)

    def play(self, n_episodes=1, render=False):
        """
        Run episodes with trained agent
        Args:
            n_episodes: number of episodes
            render: optionally render episodes

        Returns: list of cumulative episode rewards
        """
        self.epsilon = 0
        ep_rewards = []
        for i in range(n_episodes):
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
            print(f"Episode: {i + 1} Reward: {tot_reward}")
        self.env.close()
        return ep_rewards

    def evaluate(self, n_episodes=100):
        print("Evaluating agent")
        rewards = self.play(n_episodes=n_episodes)
        avg_reward = np.mean(rewards)
        print(
            f"Average total episode reward over {n_episodes} "
            f"episodes: {avg_reward:.2f}"
        )
        return avg_reward

    def save_model(self, filename):
        """
        Save H or Q model to models dir
        Args:
            filename: name of pickled file
        """
        model = self.H if self.tame else self.Q
        filename = filename + ".p" if not filename.endswith(".p") else filename
        with open(MODELS_DIR.joinpath(filename), "wb") as f:
            pickle.dump(model, f)

    def load_model(self, filename):
        """
        Load H or Q model from models dir
        Args:
            filename: name of pickled file
        """
        filename = filename + ".p" if not filename.endswith(".p") else filename
        with open(MODELS_DIR.joinpath(filename), "rb") as f:
            model = pickle.load(f)
        if self.tame:
            self.H = model
        else:
            self.Q = model

    def get_reward_from_frame(self, frame):
        df = pd.read_csv('FaceClassifier/master.csv')
        classes = df.columns[1:].to_list()
        preds = prediction(frame, model_path=self.face_classifier_path, classes=classes)

