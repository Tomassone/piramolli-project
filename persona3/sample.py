from dataclasses import dataclass
import numpy as np

from config import ACTION_SIZE, BOARD_SIZE, NUM_CHANNELS


@dataclass
class TrainingSample:
    state: np.ndarray
    pi: np.ndarray
    z: np.float32

    def validate(self) -> None:
        if self.state.shape != (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE):
            raise ValueError(f'Invalid state shape: {self.state.shape}')
        if self.state.dtype != np.float32:
            raise ValueError(f'Invalid state dtype: {self.state.dtype}')
        if self.pi.shape != (ACTION_SIZE,):
            raise ValueError(f'Invalid pi shape: {self.pi.shape}')
        if self.pi.dtype != np.float32:
            raise ValueError(f'Invalid pi dtype: {self.pi.dtype}')
        if (self.pi < 0).any():
            raise ValueError('pi contains negative values')
        if not np.isclose(float(self.pi.sum()), 1.0, atol=1e-5):
            raise ValueError(f'pi does not sum to 1: {self.pi.sum()}')
        if float(self.z) not in (-1.0, 0.0, 1.0):
            raise ValueError(f'Invalid z: {self.z}')
