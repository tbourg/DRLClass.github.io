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
import torch.cuda

BUFFER_SIZE = 10000
MINI_BATCH_SIZE = 32
EPSILON = .1
GAMMA = .01

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, env, exploration_method='epsilon_greedy'):
        self.action_space = env.action_space
        self.buffer = []
        self.exploration_method = exploration_method
        self.brain = MLP(env.observation_space.shape[0], env.action_space.n)
        self.brain_bis = copy.deepcopy(self.brain)
        self.cpt = 0

    def act(self, ob, reward, done):
        if self.exploration_method == 'epsilon_greedy':
            if random.random() < EPSILON:
                return self.action_space.sample()
            else:
                return self.get_best_action(ob)
    
    def learn(self, prev_ob, action, ob, reward, done):
        interaction = (prev_ob, action, ob, reward, done)
        self.buffer.append(interaction)
        if len(self.buffer) > BUFFER_SIZE:
            self.buffer.pop(0)
        minibatch = self.get_minibatch()
        for interaction in minibatch:
            self.cpt += 1
            if self.cpt % 100:
                self.brain_bis = copy.deepcopy(self.brain)
            q_vals_pred = self.brain(torch.tensor(interaction[0]).float())
            q_vals_pred_bis = self.brain_bis(torch.tensor(interaction[0]).float())
            q_vals_pred_next = self.brain_bis(torch.tensor(interaction[2]).float())
            q_vals = [None for _ in range(self.action_space.n)]
            if interaction[4]:
                for i in range(self.action_space.n):
                    q_vals[i] = (q_vals_pred_bis[i] - interaction[3]) ** 2
            else:
                for i in range(self.action_space.n):
                    q_vals[i] = (q_vals_pred_bis[i] - (interaction[3] + GAMMA * torch.max(q_vals_pred_next).item()))
            self.brain.train()
            loss = self.brain.loss(q_vals_pred, torch.tensor(q_vals).float())
            loss = Variable(loss, requires_grad=True)
            loss.backward()
            
    def get_minibatch(self):
        size = min(len(self.buffer), MINI_BATCH_SIZE)
        return random.sample(self.buffer, k=size)
    
    def get_q_val(self, ob, action):
        return self.get_q_vals(ob)[action]
        
    def get_best_action(self, ob):
        index = torch.argmax(self.get_q_vals(ob), 0)
        return index.int().item()
        
    def get_max_q_val(self, ob):
        val, _ = torch.max(self.get_q_vals(ob), 0)
        return val.float().item()
        
    def get_q_vals(self, ob):
        self.brain.eval()
        with torch.no_grad():
            ob = torch.tensor(ob).float()
            output = self.brain(ob)
        return output
    
    
    
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.ReLU(),
            nn.Linear(10, output_size)
        )
        self.loss = nn.MSELoss()
        
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
                print(i, episode_reward)
                rewards.append(episode_reward)
                break
    print(rewards)
    plt.plot(range(1,episode_count+1), rewards)
    plt.show()
    env.close()