from persona3.types import Action
import numpy as np

def action_to_id(action: Action) -> int:
    r0, c0 = action[0]
    r1, c1 = action[1]
    
    dr = r1 - r0
    dc = c1 - c0
    
    if dr < 0 and dc == 0:
        direction = 0 # N
        distance = -dr
    elif dr > 0 and dc == 0:
        direction = 1 # S
        distance = dr
    elif dr == 0 and dc > 0:
        direction = 2 # E
        distance = dc
    elif dr == 0 and dc < 0:
        direction = 3 # W
        distance = -dc
    else:
        direction = 0; distance = 1
        
    dir_idx = direction * 8 + (distance - 1)
    sq_idx = r0 * 9 + c0
    return sq_idx * 32 + dir_idx

def id_to_action(action_id: int):
    sq_idx = action_id // 32
    dir_idx = action_id % 32
    
    r0 = sq_idx // 9
    c0 = sq_idx % 9
    
    direction = dir_idx // 8
    distance = (dir_idx % 8) + 1
    
    if direction == 0: dr = -distance; dc = 0
    elif direction == 1: dr = distance; dc = 0
    elif direction == 2: dr = 0; dc = distance
    elif direction == 3: dr = 0; dc = -distance
    
    return ((r0, c0), (r0 + dr, c0 + dc))

def get_legal_mask(valid_moves) -> np.ndarray:
    mask = np.zeros(2592, dtype=np.float32)
    for move in valid_moves:
        try:
            mask[action_to_id(move)] = 1.0
        except Exception:
            pass
    return mask
