import matplotlib.pyplot as plt
import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(m.weight, 0, 1)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.hidden = 200
        self.l1 = nn.Linear(self.state_space, self.hidden, bias=False)
        self.l2 = nn.Linear(self.hidden, self.action_space, bias=False)

    def forward(self, x):
        model = torch.nn.Sequential(self.l1, self.l2,)
        return model(x)


if __name__ == "__main__":

    env = gym.make("MountainCar-v0")
    env.seed(1)
    state = env.reset()
    torch.manual_seed(1)
    np.random.seed(1)

    # hyperparameters
    steps = 2000
    epsilon = 0.3
    gamma = 0.99
    episodes = 3000
    learning_rate = 0.001
    successes = 0

    ave_reward_list = []
    position = []
    reward_list = []

    # Initialize Policy
    policy = Policy()
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(policy.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    for episode in range(episodes):
        episode_reward = 0
        state = env.reset()
        # print("episode: ", episode)

        for s in range(steps):

            # Uncomment to render environment
            # if episode % 100 == 0 and episode > 0:
            #    env.render()
            # print("epoch: ",s)

            # Get Q for all actions from state
            Q = policy(Variable(torch.from_numpy(state).type(torch.FloatTensor)))

            # Choose epsilon-greedy action
            if np.random.rand(1) < epsilon:
                action = np.random.randint(0, 3)
            else:
                _, action = torch.max(Q, -1)
                action = action.item()

            # Step forward and receive next state and reward
            state_1, reward, done, _ = env.step(action)

            # Find max Q for state_1(next state)
            Q1 = policy(Variable(torch.from_numpy(state_1).type(torch.FloatTensor)))
            maxQ1, _ = torch.max(Q1, -1)

            # Create target Q value for training the policy
            Q_target = Q.clone()
            Q_target = Variable(Q_target.data)
            Q_target[action] = reward + torch.mul(maxQ1.detach(), gamma)

            # Calculate loss
            loss = loss_fn(Q, Q_target)

            # Update policy
            policy.zero_grad()
            loss.backward()
            optimizer.step()

            # Save rewards
            episode_reward += reward

            if done:
                # print("done")
                if state_1[0] >= 0.5:

                    # Adjust epsilon
                    epsilon *= 0.99

                    # Adjust learning rate
                    scheduler.step()

                    # Record successful episode
                    successes += 1

                reward_list.append(episode_reward)

                position.append(state_1[0])

                break
            else:
                state = state_1

        if (episode + 1) % 100 == 0:
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = []

        if (episode + 1) % 100 == 0:
            print("Episode {} Average Reward: {}".format(episode + 1, ave_reward))

    env.close()

    print(
        "successful episodes: {:d} - {:.4f}%".format(
            successes, successes / episodes * 100
        )
    )

    # Plot Rewards
    plt.plot(100 * (np.arange(len(ave_reward_list)) + 1), ave_reward_list)
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title("Average Reward vs Episodes")
    plt.savefig("rewards_deepQLearning.jpg")
    plt.close()
