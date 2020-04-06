import numpy as np
import gym
import matplotlib.pyplot as plt


class QLearningAgent:

    def __init__(self, env, learning_rate, discount_factor, epsilon, min_eps, num_episodes):

        # Discretizing state space (discretization strategy taken from an online blog)
        num_states = (env.observation_space.high - env.observation_space.low) * \
                     np.array([10, 100])
        num_states = np.round(num_states, 0).astype(int) + 1

        # Initialize Q table
        self.Q = np.random.uniform(
            low=-1, high=1, size=(num_states[0], num_states[1], env.action_space.n))

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

    def train(self):

        # Run Q learning algorithm
        for i in range(self.num_episodes):

            done = False
            tot_reward, reward = 0, 0
            state = env.reset()

            # Discretize state
            state_adj = (state - env.observation_space.low) * np.array([10, 100])
            state_adj = np.round(state_adj, 0).astype(int)

            while not done:
                # Render environment for last five episodes
                # if i >= (episodes - 5):
                #     env.render()

                # Determine next action - epsilon greedy strategy
                if np.random.random() < 1 - self.epsilon:
                    action = np.argmax(self.Q[state_adj[0], state_adj[1]])
                else:
                    action = np.random.randint(0, env.action_space.n)

                # Get next state and reward
                next_state, reward, done, info = env.step(action)

                # Discretize next state
                next_state_adj = (next_state - env.observation_space.low) * np.array([10, 100])
                next_state_adj = np.round(next_state_adj, 0).astype(int)

                # For terminal states
                if done and next_state[0] >= 0.5:
                    self.Q[state_adj[0], state_adj[1], action] = reward

                # Update Q value for current state
                else:
                    delta = self.learning_rate * (
                        reward + self.discount_factor * np.max(self.Q[next_state_adj[0], next_state_adj[1]]) -
                        self.Q[state_adj[0], state_adj[1], action])
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
                print('Episode {} Average Reward: {}'.format(i + 1, avg_reward))

        env.close()


if __name__ == '__main__':

    env = gym.make('MountainCar-v0')
    env.reset()

    # hyperparameters
    learning_rate = 0.2
    discount_factor = 0.9
    epsilon = 0.8
    min_eps = 0
    num_episodes = 5000

    agent = QLearningAgent(env, learning_rate, discount_factor, epsilon, min_eps, num_episodes)
    agent.train()

    # Plot Rewards
    rewards = agent.avg_reward_list
    plt.plot(100 * (np.arange(len(rewards)) + 1), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    plt.savefig('rewards_QLearning.jpg')
    plt.close()
