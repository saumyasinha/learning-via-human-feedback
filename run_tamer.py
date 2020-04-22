import argparse
import asyncio
import os

import gym

from TAMER.linearTAMER import TAMERAgent, LOGS_DIR


async def main(args):
    env = gym.make("MountainCar-v0")

    # TODO: move to args?
    # hyperparameters
    discount_factor = 1
    epsilon = 0  # vanilla Q learning actually works well with no random exploration
    min_eps = 0
    num_episodes = 1
    tame = True  # set to false for vanilla Q learning

    agent = TAMERAgent(
        env,
        discount_factor,
        epsilon,
        min_eps,
        num_episodes,
        tame,
        args.tamer_training_timestep,
        output_dir=args.output,
        model_file_to_load=None,  # pretrained model name here
    )

    # TODO: move capture_video to args

    await agent.train(model_file_to_save="autosave", capture_video=False)
    # agent.load_model(filename='2_episodes_0.2s')
    # agent.play(n_episodes=1, render=True)
    # agent.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default=LOGS_DIR)
    # set a timestep for training TAMER
    # the more time per step, the easier for the human
    # but the longer it takes to train (in real time)
    # 0.2 seconds is fast but doable
    parser.add_argument("-t", "--tamer_training_timestep", default=0.2)
    args = parser.parse_args()
    asyncio.run(main(args))
