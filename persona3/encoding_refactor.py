import numpy as np

# A state is now assumed to have a history or we maintain it in the game wrapper.
# State representation from PDF:
# Per history step (x 8 steps) = 5 planes * 8 = 40
#   - Friendly pieces (1)
#   - Enemy pieces (1)
#   - King (1)
#   - Repetition >= 1 (1)
#   - Repetition >= 2 (1)
# Auxiliary = 3 planes
#   - Player color (current side to move) (1)
#   - Total move count (normalized) (1)
#   - Half-move clock (moves since last capture) (1)

def encode_state_history(hist_states: list) -> np.ndarray:
    x = np.zeros((43, 9, 9), dtype=np.float32)
    # Most recent state is hist_states[-1]
    curr_state = hist_states[-1]
    turn = curr_state['turn_to_move'] # 1=white, 0=black
    
    for i, state in enumerate(reversed(hist_states)):
        if i >= 8: break
        
        base_idx = i * 5
        white_pos = state.get('white_positions', [])
        black_pos = state.get('black_positions', [])
        king_pos = state.get('king_position')
        
        # Friendly and Enemy
        if turn == 1:
            for r, c in white_pos: x[base_idx + 0, r, c] = 1.0
            for r, c in black_pos: x[base_idx + 1, r, c] = 1.0
        else:
            for r, c in black_pos: x[base_idx + 0, r, c] = 1.0
            for r, c in white_pos: x[base_idx + 1, r, c] = 1.0
            
        if king_pos:
            x[base_idx + 2, king_pos[0], king_pos[1]] = 1.0
            if turn == 1:
                x[base_idx + 0, king_pos[0], king_pos[1]] = 1.0 # King is friendly for White
            else:
                x[base_idx + 1, king_pos[0], king_pos[1]] = 1.0 # King is enemy for Black
                
        # For Repetition, we'd check how many times the state appeared.
        rep_count = state.get('repetition_count', 0) # Requires board hash matching
        if rep_count >= 1: x[base_idx + 3, :, :] = 1.0
        if rep_count >= 2: x[base_idx + 4, :, :] = 1.0

    # Aux planes
    if turn == 1:
        x[40, :, :] = 1.0 # White
    else:
        x[40, :, :] = 0.0 # Black
        
    x[41, :, :] = curr_state.get('move_count', 0) / 100.0 # Normalized vaguely
    x[42, :, :] = curr_state.get('half_move_clock', 0) / 100.0 # Moves since capture
    
    return x

