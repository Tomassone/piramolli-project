from dataclasses import dataclass
import torch

@dataclass
class Config:
    # Optimizer
    weight_decay: float = 0.0001
    warmup_steps: int = 500
    peak_lr: float = 0.002
    min_lr: float = 0.00001
    total_training_steps: int = 102400
    batch_size: int = 512
    
    # Model
    residual_blocks: int = 8
    filters: int = 128
    
    # MCTS & Self-Play
    mcts_simulations: int = 128
    buffer_size: int = 4200000 
    past_selfplay_ratio: float = 0.25
    iterations: int = 100
    parallel_games: int = 1024
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()
