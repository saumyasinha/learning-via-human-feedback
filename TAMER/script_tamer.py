from TAMER_agent import TamerAgent, experience
import gym
import time
import numpy as np


# def getHumanSignal():



## Attempt to implement Algorithm1 of the Deep TAMER paper
if __name__ == '__main__':
    env = gym.make('MountainCar-v0')

    env.reset()

    ## Initialize TamerAgent
    agent = TamerAgent(env)

    ## hyperparameters
    num_episodes = 3000
    buffer_update_interval=100

    reward_list = []
    avg_reward_list = []

    for episode in range(num_episodes):

        done = False
        tot_reward, reward = 0, 0
        state = env.reset()

        ## D (list of all (x,y) pairs)
        agent.total_experiences = []

        ## List of all x; x = (s,a,t,t+1)_
        agent.x_list = []

        i = 0
        j = 0
        k = 0
        while done!=True:
            # Render environment for last five episodes
            # if i >= (num_episodes - 5):
            #     env.render()

            ## Choosing action via a greedy policy; action = argmaxH(s,a)
            action = agent.getAction(state)

            # Step forward and receive next state and reward
            state2, reward, done, info = env.step(action)
            tot_reward += reward

            current_time = time.time()

            ## Add this x in the x_list
            exp = experience(state, action, current_time, current_time + 1,0,0)
            agent.x_list.append(exp)

            ## Proxy for receiving human signal/feedback
            signal = np.random.choice([0,1,2,3],1, p=[0.8,0.05,0.05,0.1])

            # signal = getHumanSignal()

            ## If human gives a feedback
            if signal!=0:

                feedback_time = time.time()
                ## Update weights (assigning credit) of all x and build the set Dj for the current feedback
                agent.updateWeightsforSignal(signal, feedback_time)
                j=j+1

                ## mini-batch SGD update of the reward model(H)
                agent.SGD_update("human")
                k = k+1


            ##Feedback replay buffer used in the paper: update the model at regular intervals
            if (i%buffer_update_interval)==0 and len(agent.total_experiences)>0:
                agent.SGD_update("fixed")
                k=k+1


            state = state2

            i=i+1

        reward_list.append(tot_reward)

        if (i + 1) % 100 == 0:
            avg_reward = np.mean(reward_list)
            avg_reward_list.append(avg_reward)
            reward_list = []

        if (i + 1) % 100 == 0:
            print('Episode {} Average Reward: {}'.format(i + 1, avg_reward))













