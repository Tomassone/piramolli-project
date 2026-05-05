import numpy as np
from persona3.types import EncodingInput, ATTACKER, DEFENDER

def encode_state(enc_input: EncodingInput) -> np.ndarray:
    x = np.zeros((43, 9, 9), dtype=np.float32)
    turn = enc_input.side_to_move
    
    # For now, without full 8-step history from game, encode step 0 (most recent) in 0-4
    # We will just fill plane 0-4
    white_pos = enc_input.white_positions
    black_pos = enc_input.black_positions
    king_pos = enc_input.king_position
    
    if turn == ATTACKER:
        for r, c in black_pos: x[0, r, c] = 1.0 # Friendly
        for r, c in white_pos: x[1, r, c] = 1.0 # Enemy
    else:
        for r, c in white_pos: x[0, r, c] = 1.0
        for r, c in black_pos: x[1, r, c] = 1.0
        
    if king_pos is not None:
        x[2, king_pos[0], king_pos[1]] = 1.0
        if turn == DEFENDER: x[0, king_pos[0], king_pos[1]] = 1.0
        else: x[1, king_pos[0], king_pos[1]] = 1.0
            
    # Aux
    if turn == ATTACKER: x[40, :, :] = 1.0
    x[41, :, :] = enc_input.move_count / 512.0
    x[42, :, :] = enc_input.half_move_clock / 100.0
    
    return x
