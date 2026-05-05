from typing import Literal, List, Tuple
import numpy as np
from dataclasses import dataclass

State = dict
Action = Tuple[Tuple[int, int], Tuple[int, int]]
Player = Literal[1, -1]
Winner = Literal[1, -1, 0]

ATTACKER = 1
DEFENDER = -1

@dataclass
class EncodingInput:
    white_positions: List[Tuple[int, int]]
    black_positions: List[Tuple[int, int]]
    king_position: Tuple[int, int]
    side_to_move: Player
    move_count: int
    half_move_clock: int
    position_history: List[str]

@dataclass
class TrainingSample:
    state: np.ndarray   # float32, shape [43, 9, 9] (channels-first)
    pi: np.ndarray      # float32, shape [2592]
    z: float            # float32, scalar in {-1.0, 0.0, +1.0}
    player: Player      # +1 = attacker, -1 = defender
