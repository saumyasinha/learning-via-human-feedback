import gym
from TAMER.linearTAMER import TAMERAgent


if __name__ == '__main__':
    env = gym.make("MountainCar-v0")

    # hyperparameters
    discount_factor = 1
    epsilon = 0  # vanilla Q learning actually works well with no random exploration
    min_eps = 0
    num_episodes = 1
    tame = True  # set to false for vanilla Q learning

    # set a timestep for training TAMER
    # the more time per step, the easier for the human
    # but the longer it takes to train (in real time)
    # 0.2 seconds is fast but doable
    tamer_training_timestep = 0.2  # seconds

    agent = TAMERAgent(
        env,
        discount_factor,
        epsilon,
        min_eps,
        num_episodes,
        tame,
        tamer_training_timestep,
        # make sure this is False if you want to train a fresh model:
        load_last_model=True
    )

    agent.train(auto_save_model=True)
    # agent.play(1, render=True)
    agent.evaluate()
