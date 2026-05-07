from collections import deque
import random
import numpy as np

from sample import TrainingSample


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, sample: TrainingSample) -> None:
        sample.validate()
        self.buffer.append(sample)

    def extend(self, samples: list[TrainingSample]) -> None:
        for sample in samples:
            self.add(sample)

    def sample_numpy_batch(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states = np.stack([x.state for x in batch]).astype(np.float32)
        pis = np.stack([x.pi for x in batch]).astype(np.float32)
        zs = np.array([x.z for x in batch], dtype=np.float32).reshape(-1, 1)
        return states, pis, zs
