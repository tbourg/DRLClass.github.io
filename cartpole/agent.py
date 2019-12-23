import copy
import random
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from dqn import MLP
from buffer import ReplayMemory

BUFFER_SIZE = 10000
MINI_BATCH_SIZE = 32
EPSILON = .1
GAMMA = .9

class Agent(object):
    def __init__(self, env, exploration_method='epsilon_greedy'):
        self.action_space = env.action_space
        self.buffer = ReplayMemory(BUFFER_SIZE, MINI_BATCH_SIZE)
        self.exploration_method = exploration_method
        self.brain = MLP(env.observation_space.shape[0], env.action_space.n)
        self.brain_bis = copy.deepcopy(self.brain)
        self.Tensor = torch.Tensor
        self.LongTensor = torch.LongTensor
        self.cpt = 0
        self.optimizer = optim.Adam(self.brain.parameters())

    def act(self, ob, reward, done):
        if self.exploration_method == 'epsilon_greedy':
            if random.random() < EPSILON:
                return self.action_space.sample()
            else:
                return self.get_best_action(ob)
    
    def learn(self, prev_ob, action, ob, reward, done):
        self.cpt += 1
        if self.cpt % 10 == 0:
            self.brain_bis = copy.deepcopy(self.brain)
        self.buffer.add(prev_ob, action, ob, reward, done)
        batch = self.buffer.get_minibatch()
        [states, actions, next_states, rewards, dones] = zip(*batch)
        state_batch = Variable(self.Tensor(states))
        action_batch = Variable(self.LongTensor(actions))
        reward_batch = Variable(self.Tensor(rewards))
        next_states_batch = Variable(self.Tensor(next_states))
        
        state_action_values = self.brain(state_batch).gather(1, action_batch.view(-1,1)).view(-1)
        with torch.no_grad():
            next_state_values = self.brain_bis(next_states_batch).max(1)[0]
            for i in range(len(batch)):
                if dones[i]:
                    next_state_values.data[i]=0
            
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        loss = F.mse_loss(state_action_values, expected_state_action_values)        
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.brain.parameters(), 10)  # Clip gradients (normalising by max value of gradient L2 norm)
        self.optimizer.step()
        '''
        for interaction in minibatch:
            self.cpt += 1
            if self.cpt % 100:
                self.brain_bis = copy.deepcopy(self.brain)
            q_vals_pred = self.brain(torch.tensor(interaction[0]).float())
            q_vals_pred_next = self.brain_bis(torch.tensor(interaction[2]).float())
            q_vals = [None for _ in range(self.action_space.n)]
            if interaction[4]:
                for i in range(self.action_space.n):
                    q_vals[i] = (q_vals_pred[i] - interaction[3]) ** 2
            else:
                for i in range(self.action_space.n):
                    q_vals[i] = (q_vals_pred[i] - (interaction[3] + GAMMA * torch.max(q_vals_pred_next).item())) ** 2
            self.brain.train()
            loss = self.brain.loss(q_vals_pred, torch.tensor(q_vals).float())
            loss = Variable(loss, requires_grad=True)
            loss.backward()
            '''
    
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
    
    
    


