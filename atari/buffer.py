import random

class ReplayMemory():
    def __init__(self, capacity, mb_size):
        self.memory = []
        self.capacity = capacity
        self.mb_size = mb_size

    def add(self, prev_ob, action, ob, reward, done):
        interaction = [prev_ob, action, ob, reward, done]
        self.memory.append(interaction)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)


    def get_minibatch(self):
        size = min(len(self.memory), self.mb_size)
        return random.sample(self.memory, k=size)
        
        
