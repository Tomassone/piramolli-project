from persona3.types import State, Action, EncodingInput, ATTACKER, DEFENDER, Winner, Player
from typing import Literal
import random
import numpy as np

class MockGame:
    def __init__(self):
        pass
    def get_initial_state(self) -> State:
        return {'side_to_move': ATTACKER, 'move_count': 0, 'half_move_clock': 0, 'terminal': False, 'white_pos': [], 'black_pos': [(0,0)], 'king_pos': (4,4)}
    def get_valid_moves(self, state: State) -> list[Action]:
        return [((0,0), (0,1)), ((4,4), (4,5))]
    def get_next_state(self, state: State, action: Action) -> State:
        return {'side_to_move': -state['side_to_move'], 'move_count': state['move_count']+1, 'half_move_clock': state['half_move_clock']+1, 'terminal': random.random() < 0.1, 'white_pos': [], 'black_pos': [(0,0)], 'king_pos': (4,4)}
    def is_terminal(self, state: State) -> bool:
        return state['terminal'] or state['move_count'] > 20
    def get_winner(self, state: State) -> Winner:
        if not self.is_terminal(state): return 0
        return random.choice([ATTACKER, DEFENDER, 0])
    def state_to_encoding_input(self, state: State) -> EncodingInput:
        return EncodingInput(
            white_positions=state['white_pos'],
            black_positions=state['black_pos'],
            king_position=state['king_pos'],
            side_to_move=state['side_to_move'],
            move_count=state['move_count'],
            half_move_clock=state['half_move_clock'],
            position_history=[]
        )

def get_action_probs(state: State, net, temperature: float) -> np.ndarray:
    game = MockGame()
    moves = game.get_valid_moves(state)
    probs = np.zeros(2592, dtype=np.float32)
    from persona3.action_encoding import action_to_id
    if moves:
        for m in moves: probs[action_to_id(m)] = 1.0 / len(moves)
    else:
        probs[0] = 1.0
    return probs
