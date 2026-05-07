from config import BOARD_SIZE


def rc_to_idx(r: int, c: int) -> int:
    return r * BOARD_SIZE + c


def idx_to_rc(idx: int) -> tuple[int, int]:
    return idx // BOARD_SIZE, idx % BOARD_SIZE


def action_id(from_idx: int, to_idx: int) -> int:
    return from_idx * (BOARD_SIZE * BOARD_SIZE) + to_idx


def decode_action(action: int) -> tuple[tuple[int, int], tuple[int, int]]:
    n = BOARD_SIZE * BOARD_SIZE
    from_idx = action // n
    to_idx = action % n
    return idx_to_rc(from_idx), idx_to_rc(to_idx)
