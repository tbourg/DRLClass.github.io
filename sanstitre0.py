import argparse
import sys
import matplotlib.pyplot as plt

import gym
from gym import wrappers, logger

BUFFER_SIZE = 10000
MNI_BAATCH_SIZE = 100

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v1', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)
    rewards = []

    #env = wrappers.Monitor(env, force=True)
    #env.seed(0)
    agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        episode_reward = 0
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                rewards.append(episode_reward)
                break
    print(rewards)
    plt.plot(range(1,episode_count+1), rewards)
    plt.show()
    env.close()