import copy

def is_hostile(r, c, is_defender_turn, king_pos):
    if r == 4 and c == 4:
        # The throne is hostile to the defenders only when the king is not in the throne
        if is_defender_turn and king_pos != (4, 4):
            return True
        elif not is_defender_turn:
            return True
        return False
    # Corners are always hostile
    if (r, c) in [(0,0), (0,8), (8,0), (8,8)]:
        return True
    return False

def check_captures(state):
    # This checks captures based on the new rules:
    # "Any piece can be captured by sandwiching it between two enemy pieces."
    # "The king is captured like any other piece and it can also participate in captures."
    # "The throne and the corners are hostile squares... can be used to capture pieces by both players."
    pass # Implementation of generic sandwich rule
