from collections import deque
import random
from persona3.types import TrainingSample
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, sample: TrainingSample):
        self.buffer.append(sample)
        
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states = np.stack([b.state for b in batch])
        pis = np.stack([b.pi for b in batch])
        zs = np.array([b.z for b in batch], dtype=np.float32)
        players = np.array([b.player for b in batch], dtype=np.float32)
        return states, pis, zs, players
    
    def __len__(self):
        return len(self.buffer)
