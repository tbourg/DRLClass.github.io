import argparse
import sys
import matplotlib.pyplot as plt
import random

import gym
from gym import wrappers, logger
import torch
import torch.nn as nn

BUFFER_SIZE = 10000
MINI_BATCH_SIZE = 100
EPSILON = 1

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, env, exploration_method='epsilon_greedy'):
        self.action_space = env.action_space
        self.buffer = []
        self.exploration_method = exploration_method
        self.brain = MLP(env.observation_space.shape[0], env.action_space.n)

    def act(self, ob, reward, done):
        if self.exploration_method == 'epsilon_greedy':
            if random.random() < EPSILON:
                return self.action_space.sample()
            else:
                self.brain.eval()
                with torch.no_grad():
                    output = self.brain(ob)
                    index = torch.max(output, 1)
                return self.action_space[i]
    
    def learn(self, prev_ob, action, ob, reward, done):
        interaction = (prev_ob, action, ob, reward, done)
        self.buffer.append(interaction)
        if len(self.buffer) > BUFFER_SIZE:
            self.buffer.pop(0)
            
    def get_minibatch(self):
        size = min(len(self.buffer), MINI_BATCH_SIZE)
        return random.sample(self.buffer, k=size)
    
    def get_q_val(self, ob, action):
        pass
        
    def get_best_action(self, ob):
        pass
        
    def get_q_vals(self, ob):
        self.brain.eval()
        with torch.no_grad():
            output = self.brain(ob)
        return output
    
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, output_size)
        )
        
    def forward(self, x):
        x = self.layers(x)
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v1', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)
    rewards = []
    print(env.action_space.n)

    #env = wrappers.Monitor(env, force=True)
    #env.seed(0)
    agent = RandomAgent(env)

    episode_count = 100
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
                print(ob, action)
                rewards.append(episode_reward)
                break
    print(rewards)
    plt.plot(range(1,episode_count+1), rewards)
    plt.show()
    env.close()