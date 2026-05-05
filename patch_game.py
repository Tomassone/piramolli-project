import copy

def get_next_state(self, state, action):
    # This is a very simplified get_next_state that applies the move but misses some complex capturing rules.
    # To fully implement Ashton Tablut captures, one should integrate GameAshtonTablut's check_capture logic.
    next_state = copy.deepcopy(state)
    r0, c0 = action[0]
    r1, c1 = action[1]
    
    # move piece
    if tuple(action[0]) in next_state['white_positions']:
        next_state['white_positions'].remove(tuple(action[0]))
        next_state['white_positions'].append(tuple(action[1]))
    elif tuple(action[0]) in next_state['black_positions']:
        next_state['black_positions'].remove(tuple(action[0]))
        next_state['black_positions'].append(tuple(action[1]))
    
    # move king
    if next_state.get('king_position') == tuple(action[0]):
        next_state['king_position'] = tuple(action[1])

    # change turn
    # Assuming turn is either 'side_to_move' or 'turn_to_move', let's toggle both for safety.
    if 'side_to_move' in next_state:
        next_state['side_to_move'] = 1 - next_state['side_to_move']
    if 'turn_to_move' in next_state:
        next_state['turn_to_move'] = 1 - next_state['turn_to_move']
        
    return next_state
