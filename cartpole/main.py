import argparse
import sys
import matplotlib.pyplot as plt
import random
import copy

import gym
from gym import wrappers, logger
import torch
from torch.autograd import Variable
import torch.nn as nn

from agent import Agent

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
    agent = Agent(env)

    episode_count = 200
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        prev_ob = ob
        episode_reward = 0
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            agent.learn(prev_ob, action, ob, reward, done)
            prev_ob = ob
            episode_reward += reward
            if done:
                print(i, episode_reward)
                rewards.append(episode_reward)
                break
    print(rewards)
    plt.plot(range(1,episode_count+1), rewards)
    plt.show()
    env.close()