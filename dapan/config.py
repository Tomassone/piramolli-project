from dataclasses import dataclass

BOARD_SIZE = 9
NUM_CHANNELS = 7
ACTION_SIZE = 81 * 81
WHITE = 1
BLACK = -1
DRAW = 0


@dataclass(frozen=True)
class TrainConfig:
    replay_buffer_capacity: int = 50_000
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_mock_samples: int = 512
    steps_per_epoch: int = 20
    epochs: int = 3
    checkpoint_dir: str = 'checkpoints'
    device: str = 'cuda' # 'cpu' se cuda non è disponibile
