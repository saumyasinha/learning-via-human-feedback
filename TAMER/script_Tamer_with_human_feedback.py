from TAMER.TAMER_Agent import TamerAgent, experience
import gym
import time
import numpy as np
import pygame

interval = 60
fps = 60

def getHumanSignal(screen, lasttime):

    pressed = pygame.key.get_pressed()
    now = time.time()
    # positive signal ((now-lasttime)>interval is to overcome the effect of human pressing the key for longer milliseconds
    if pressed[pygame.K_w] and (now-lasttime)>interval:
        screen.fill((0,255,0))
        lasttime = now
        return 1, lasttime
    #negative signal
    elif pressed[pygame.K_a] and (now-lasttime)>interval:
        screen.fill((255,0,0))
        lasttime = now
        return -1, lasttime
    else:
        screen.fill((0, 0, 0))
        return 0, lasttime


## Attempt to implement Algorithm1 of the Deep TAMER paper
if __name__ == '__main__':
    env = gym.make('MountainCar-v0')

    env.reset()
    env.render()

    ## Initialize Tamer Agent
    agent = TamerAgent(env)

    ## hyperparameters
    num_episodes = 100
    buffer_update_interval=75

    reward_list = []
    avg_reward_list = []


    ## pygame initialization
    pygame.init()
    screen = pygame.display.set_mode((64, 48))
    screen.fill((0, 0, 0))
    pygame.display.flip()
    clock = pygame.time.Clock()


    for episode in range(num_episodes):

        done = False
        is_exit = False
        tot_reward, reward = 0, 0
        state = env.reset()
        lasttime = 0  # when was the last time human input is taken

        ## D (list of all experiences in form of (x,y) pairs)
        agent.total_experiences = []

        ## List of all x; x = (s,a,t,t+1)_
        agent.x_list = []

        i = 0
        j = 0
        k = 0
        while done!=True and is_exit!=True:

            ## Choosing action via a greedy policy; action = argmax H(s,a)
            action = agent.getAction(state)

            current_time = time.time()

            ## Add the x in the x_list
            exp = experience(state, action, current_time, current_time + 1, 0, 0)
            agent.x_list.append(exp)

            # Step forward and receive next state and reward
            state2, reward, done, info = env.step(action)
            tot_reward += reward

            # receive human signal (if any)
            signal,lasttime = getHumanSignal(screen, lasttime)
            pygame.display.flip()
            # print(signal)

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

            # process pygame event queue
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    is_exit = True
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    is_exit = True
                    break

            env.render()

            # setting fps for pygame
            clock.tick(fps)

        reward_list.append(tot_reward)

        # obtain the average reward at every 100th episode
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(reward_list)
            avg_reward_list.append(avg_reward)
            reward_list = []

        if (episode + 1) % 100 == 0:
            print('Episode {} Average Reward: {}'.format(episode + 1, avg_reward))


    env.close()













