from persona3.types import TrainingSample, ATTACKER, DEFENDER
from persona3.encoding import encode_state
import numpy as np

def run_episode(game, net, get_action_probs_fn):
    history = []
    state = game.get_initial_state()
    
    while not game.is_terminal(state):
        player = state['side_to_move']
        enc_input = game.state_to_encoding_input(state)
        arr_state = encode_state(enc_input)
        
        pi = get_action_probs_fn(state, net, 1.0)
        
        valid_moves = game.get_valid_moves(state)
        from persona3.action_encoding import action_to_id
        if not valid_moves:
            break
            
        move = valid_moves[np.argmax([pi[action_to_id(m)] for m in valid_moves])] # simple greedy for now in mock
        history.append((arr_state, pi, player, state))
        
        state = game.get_next_state(state, move)
        
    winner = game.get_winner(state)
    samples = []
    
    for arr_state, pi, player, _ in history:
        z = 0.0
        if winner != 0:
            if player == winner: z = 1.0
            else: z = -1.0
        samples.append(TrainingSample(state=arr_state, pi=pi, z=z, player=player))
        
    return samples
