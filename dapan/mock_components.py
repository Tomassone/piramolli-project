import random
import numpy as np

from actions import action_id, rc_to_idx
from config import ACTION_SIZE, BLACK, DRAW, WHITE
from encoding import encode_state
from sample import TrainingSample


class MockGame:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def random_state(self) -> dict:
        cells = [(r, c) for r in range(9) for c in range(9)]
        self.rng.shuffle(cells)
        king = cells.pop()
        whites = [cells.pop() for _ in range(8)]
        blacks = [cells.pop() for _ in range(16)]
        side_to_move = self.rng.choice([WHITE, BLACK])
        return {
            'white_positions': whites,
            'black_positions': blacks,
            'king_position': king,
            'side_to_move': side_to_move,
        }


class MockMCTS:
    def __init__(self, seed: int = 0, legal_actions_per_state: int = 10):
        self.rng = random.Random(seed)
        self.legal_actions_per_state = legal_actions_per_state

    def policy_target(self, state: dict) -> np.ndarray:
        pi = np.zeros((ACTION_SIZE,), dtype=np.float32)
        occupied = set(state['white_positions']) | set(state['black_positions']) | {state['king_position']}
        from_cells = list(occupied)
        to_cells = [(r, c) for r in range(9) for c in range(9) if (r, c) not in occupied]

        chosen = set()
        while len(chosen) < self.legal_actions_per_state:
            fr = self.rng.choice(from_cells)
            to = self.rng.choice(to_cells)
            fr_idx = rc_to_idx(*fr)
            to_idx = rc_to_idx(*to)
            if fr_idx != to_idx:
                chosen.add(action_id(fr_idx, to_idx))

        weights = np.array([self.rng.random() + 1e-3 for _ in range(len(chosen))], dtype=np.float32)
        weights /= weights.sum()
        for a, w in zip(list(chosen), weights):
            pi[a] = w
        return pi


class MockSelfPlay:
    def __init__(self, game: MockGame, mcts: MockMCTS, seed: int = 0):
        self.game = game
        self.mcts = mcts
        self.rng = random.Random(seed)

    def sample_outcome(self) -> int:
        return self.rng.choice([WHITE, BLACK, DRAW])

    def make_sample(self) -> TrainingSample:
        state = self.game.random_state()
        final_winner = self.sample_outcome()
        z = 0.0 if final_winner == DRAW else (1.0 if final_winner == state['side_to_move'] else -1.0)
        sample = TrainingSample(
            state=encode_state(state),
            pi=self.mcts.policy_target(state),
            z=np.float32(z),
        )
        sample.validate()
        return sample

    def make_dataset(self, n: int) -> list[TrainingSample]:
        return [self.make_sample() for _ in range(n)]