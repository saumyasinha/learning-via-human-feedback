import numpy as np
import gym
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # stops python crashing


class QLearningAgent:
    def __init__(
        self, env, learning_rate, discount_factor, epsilon, min_eps, num_episodes
    ):

        env.reset()
        self.env = env

        # Discretizing state space (discretization strategy taken from an online blog)
        num_states = (
            env.observation_space.high - env.observation_space.low
        ) * np.array([10, 100])
        num_states = np.round(num_states, 0).astype(int) + 1

        # Initialize Q table
        self.Q = np.random.uniform(
            low=-1, high=1, size=(num_states[0], num_states[1], env.action_space.n)
        )

        # Hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.num_episodes = num_episodes

        # Calculate episodic reduction in epsilon
        self.epsilon_step = (epsilon - min_eps) / num_episodes

        # Rewards
        self.reward_list = []
        self.avg_reward_list = []

    def act(self, state_adj):
        """ Epsilon-greedy Policy """
        if np.random.random() < 1 - self.epsilon:
            return np.argmax(self.Q[state_adj[0], state_adj[1]])
        else:
            return np.random.randint(0, self.env.action_space.n)

    def train(self):

        # Run Q learning algorithm
        for i in range(self.num_episodes):

            done = False
            tot_reward, reward = 0, 0
            state = self.env.reset()

            # Discretize state
            state_adj = self.discretize_state(state)

            while not done:
                # Render environment for last five episodes
                # if i >= (episodes - 5):
                #     self.env.render()

                # Determine next action
                action = self.act(state_adj)

                # Get next state and reward
                next_state, reward, done, info = self.env.step(action)

                # Discretize next state
                next_state_adj = self.discretize_state(next_state)

                # For terminal states
                if done and next_state[0] >= 0.5:
                    self.Q[state_adj[0], state_adj[1], action] = reward

                # Update Q value for current state
                else:
                    delta = self.learning_rate * (
                        reward
                        + self.discount_factor
                        * np.max(self.Q[next_state_adj[0], next_state_adj[1]])
                        - self.Q[state_adj[0], state_adj[1], action]
                    )
                    self.Q[state_adj[0], state_adj[1], action] += delta

                # Update variables
                tot_reward += reward
                state_adj = next_state_adj

            # Decay epsilon
            if self.epsilon > min_eps:
                self.epsilon -= self.epsilon_step

            # Save rewards
            self.reward_list.append(tot_reward)

            if (i + 1) % 100 == 0:
                avg_reward = np.mean(self.reward_list)
                self.avg_reward_list.append(avg_reward)
                self.reward_list = []
                print("Episode {} Average Reward: {}".format(i + 1, avg_reward))

        self.env.close()

    def play(self):
        state_adj = self.discretize_state(self.env.reset())
        done = False
        while not done:
            action = self.act(state_adj)
            next_state, reward, done, info = self.env.step(action)
            self.env.render()
            state_adj = self.discretize_state(next_state)
        self.env.close()

    def discretize_state(self, state):
        state_adj = (state - self.env.observation_space.low) * np.array([10, 100])
        return np.round(state_adj, 0).astype(int)


if __name__ == "__main__":

    env = gym.make("MountainCar-v0")

    # hyperparameters
    learning_rate = 0.2
    discount_factor = 0.9
    epsilon = 0.8
    min_eps = 0
    num_episodes = 1000

    agent = QLearningAgent(
        env, learning_rate, discount_factor, epsilon, min_eps, num_episodes
    )
    agent.train()
    agent.play()

    # Plot Rewards
    rewards = agent.avg_reward_list
    plt.plot(100 * (np.arange(len(rewards)) + 1), rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title("Average Reward vs Episodes")
    plt.savefig("rewards_QLearning.jpg")
    plt.close()
