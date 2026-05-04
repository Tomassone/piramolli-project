import numpy as np

from config import BOARD_SIZE, NUM_CHANNELS, WHITE

TRONE = {(4, 4)}
CAMPS = {
    (0, 3), (0, 4), (0, 5), (1, 4),
    (3, 0), (4, 0), (5, 0), (4, 1),
    (8, 3), (8, 4), (8, 5), (7, 4),
    (3, 8), (4, 8), (5, 8), (4, 7),
}
ESCAPES = {
    (0, 1), (0, 2), (0, 6), (0, 7),
    (1, 0), (2, 0), (6, 0), (7, 0),
    (8, 1), (8, 2), (8, 6), (8, 7),
    (1, 8), (2, 8), (6, 8), (7, 8),
}


def encode_state(state: dict) -> np.ndarray:
    x = np.zeros((NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    for r, c in state['white_positions']:
        x[0, r, c] = 1.0
    for r, c in state['black_positions']:
        x[1, r, c] = 1.0

    kr, kc = state['king_position']
    x[2, kr, kc] = 1.0

    for r, c in TRONE:
        x[3, r, c] = 1.0
    for r, c in CAMPS:
        x[4, r, c] = 1.0
    for r, c in ESCAPES:
        x[5, r, c] = 1.0

    x[6, :, :] = 1.0 if state['side_to_move'] == WHITE else 0.0
    return x
