import numpy as np
import gym
import matplotlib.pyplot as plt


def QLearning(env, learning, discount, epsilon, min_eps, episodes):

    # Discretizing state space (discretization strategy taken from an online blog)
    num_states = (env.observation_space.high - env.observation_space.low) * \
                 np.array([10, 100])
    num_states = np.round(num_states, 0).astype(int) + 1

    # Initialize Q table
    Q = np.random.uniform(low=-1, high=1,
                          size=(num_states[0], num_states[1],
                                env.action_space.n))

    reward_list = []
    ave_reward_list = []

    # Calculate episodic reduction in epsilon
    reduction = (epsilon - min_eps) / episodes

    # Run Q learning algorithm
    for i in range(episodes):

        done = False
        tot_reward, reward = 0, 0
        state = env.reset()

        # Discretize state
        state_adj = (state - env.observation_space.low) * np.array([10, 100])
        state_adj = np.round(state_adj, 0).astype(int)

        while done != True:

            # Render environment for last five episodes
            # if i >= (episodes - 5):
            #     env.render()

            # Determine next action - epsilon greedy strategy
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state_adj[0], state_adj[1]])
            else:
                action = np.random.randint(0, env.action_space.n)

            # Get next state and reward
            state2, reward, done, info = env.step(action)

            # Discretize state2
            state2_adj = (state2 - env.observation_space.low) * np.array([10, 100])
            state2_adj = np.round(state2_adj, 0).astype(int)

            # For terminal states
            if done and state2[0] >= 0.5:
                Q[state_adj[0], state_adj[1], action] = reward

            # Update Q value for current state
            else:
                delta = learning * (reward +
                                    discount * np.max(Q[state2_adj[0],
                                                        state2_adj[1]]) -
                                    Q[state_adj[0], state_adj[1], action])
                Q[state_adj[0], state_adj[1], action] += delta

            # Update variables
            tot_reward += reward
            state_adj = state2_adj

        # Decay epsilon
        if epsilon > min_eps:
            epsilon -= reduction

        # Save rewards
        reward_list.append(tot_reward)

        if (i + 1) % 100 == 0:
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = []

        if (i + 1) % 100 == 0:
            print('Episode {} Average Reward: {}'.format(i + 1, ave_reward))

    env.close()

    return ave_reward_list


if __name__ == '__main__':

    env = gym.make('MountainCar-v0')
    env.reset()

    #hyperparameters
    learning = 0.2
    discount = 0.9
    epsilon = 0.8
    min_eps = 0
    episodes = 5000

    rewards = QLearning(env, learning, discount, epsilon, min_eps, episodes)

    # Plot Rewards
    plt.plot(100 * (np.arange(len(rewards)) + 1), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    plt.savefig('rewards_QLearning.jpg')
    plt.close()