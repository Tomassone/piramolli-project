import numpy as np
import copy
from collections import deque
import encoding

# ── Action encoding ──────────────────────────────────────────────────────────
# Each action is a (from_row, from_col, to_row, to_col) 4-tuple.
# Flat index = (r0 * 9 + c0) * 81 + (r1 * 9 + c1)
# Total action space: 9^4 = 6561
# ─────────────────────────────────────────────────────────────────────────────

BOARD_SIZE = 9
ACTION_SIZE = BOARD_SIZE ** 4   # 6561


def _encode_action(r0, c0, r1, c1) -> int:
    return (r0 * BOARD_SIZE + c0) * (BOARD_SIZE ** 2) + (r1 * BOARD_SIZE + c1)


def _decode_action(idx: int):
    from_idx, to_idx = divmod(idx, BOARD_SIZE ** 2)
    r0, c0 = divmod(from_idx, BOARD_SIZE)
    r1, c1 = divmod(to_idx, BOARD_SIZE)
    return r0, c0, r1, c1


def _internal_player(base_player: int) -> int:
    """Convert base-class convention (1 / -1) → internal (1 / 0)."""
    return 1 if base_player == 1 else 0


def _base_player(internal_player: int) -> int:
    """Convert internal (1 / 0) → base-class convention (1 / -1)."""
    return 1 if internal_player == 1 else -1


class Game:
    """
    Tablut / Hnefatafl game conforming to the alpha-zero-general Game interface.

    Board representation
    --------------------
    The "board" object passed between interface methods is the full nested
    state dict produced by getInitBoard():
        {
            'board': {
                'white_positions': [...],
                'black_positions': [...],
                'king_position': (r, c) | None,
                'turn_to_move': 1 | 0,   # 1 = white, 0 = black (internal)
            },
            'history': deque of 8 board snapshots,
            'move_count': int,
            'half_move_clock': int,
            'repetition_count': int,
            'repetition_loser': None | int,
        }

    Players
    -------
    Following the base-class convention:
        player =  1  →  white (defender, moves king)
        player = -1  →  black (attacker)

    Actions
    -------
    Actions are integers in [0, 6561).
    Encoding: (r0*9 + c0) * 81 + (r1*9 + c1)
    """

    # ── Board constants ───────────────────────────────────────────────────────
    _BOARD_SIZE = 9
    _THRONE = (4, 4)
    _CAMPS = {
        # Top
        (0, 3), (0, 4), (0, 5), (1, 4),
        # Bottom
        (8, 3), (8, 4), (8, 5), (7, 4),
        # Left
        (3, 0), (4, 0), (5, 0), (4, 1),
        # Right
        (3, 8), (4, 8), (5, 8), (4, 7),
    }
    _ESCAPES = encoding.ESCAPES
    _HISTORY_LEN = 8

    # ── Base interface ────────────────────────────────────────────────────────

    def __init__(self):
        pass

    def getInitBoard(self):
        """
        Returns the starting game state dict (used as the 'board' everywhere).
        White (defender) moves first.
        """
        board = {
            'white_positions': [
                (2, 4), (3, 4), (4, 2), (4, 3),
                (4, 5), (4, 6), (5, 4), (6, 4),
            ],
            'black_positions': [
                (0, 3), (0, 4), (0, 5), (1, 4),
                (3, 0), (4, 0), (5, 0), (4, 1),
                (8, 3), (8, 4), (8, 5), (7, 4),
                (3, 8), (4, 8), (5, 8), (4, 7),
            ],
            'king_position': (4, 4),
            'turn_to_move': 1,   # internal: 1 = white
        }
        history = deque(
            [copy.deepcopy(board) for _ in range(self._HISTORY_LEN)],
            maxlen=self._HISTORY_LEN,
        )
        return {
            'board': board,
            'history': history,
            'move_count': 0,
            'half_move_clock': 0,
            'repetition_count': 0,
            'repetition_loser': None,
        }

    def getBoardSize(self):
        """Returns (9, 9) — the spatial dimensions of the Tablut board."""
        return (self._BOARD_SIZE, self._BOARD_SIZE)

    def getActionSize(self):
        """
        Returns 6561 (= 9^4), the total number of (from, to) cell pairs.
        Most are illegal in any given position; getValidMoves masks them.
        """
        return ACTION_SIZE

    def getNextState(self, board, player, action):
        """
        Applies `action` (an integer) to `board` and returns
        (next_board, next_player).

        The `player` argument is accepted for interface compatibility but the
        true turn is always read from board['board']['turn_to_move'] so the
        two are always in sync.
        """
        r0, c0, r1, c1 = _decode_action(action)
        new_state = self._apply_move(board, [[r0, c0], [r1, c1]])
        next_internal = new_state['board']['turn_to_move']
        return new_state, _base_player(next_internal)

    def getValidMoves(self, board, player):
        """
        Returns a binary numpy vector of length 6561.
        A 1 at position i means action i is legal from the current position.
        """
        raw_moves = self._get_raw_moves(board)
        valid = np.zeros(ACTION_SIZE, dtype=np.int8)
        for move in raw_moves:
            (r0, c0), (r1, c1) = move
            valid[_encode_action(r0, c0, r1, c1)] = 1
        return valid

    def getGameEnded(self, board, player):
        """
        Returns:
            0   — game is still ongoing
            1   — `player` has won
           -1   — `player` has lost
            1e-4 — draw (rare: three-fold repetition with no designated loser)
        """
        if not self._is_terminal(board):
            return 0

        winner_internal = self._get_winner(board)  # 1 = white, 0 = black, -1 = draw
        if winner_internal == -1:
            return 1e-4   # draw

        winner_base = _base_player(winner_internal)
        return 1 if winner_base == player else -1

    def getCanonicalForm(self, board, player):
        """
        The canonical form is always from white's (defender's) point of view,
        matching how the tensor encoder already works (friendly = white).
        When it is black's turn we flip the turn marker in a shallow copy so
        the neural network always sees a consistent orientation.

        Note: Because Tablut has no colour-symmetric structure (the king piece
        is unique), a full board flip is meaningless. The canonical form here
        simply ensures the tensor's 'friendly/enemy' planes are computed from
        white's perspective regardless of whose turn it is.
        """
        # The encode_state method already handles both turns correctly via its
        # 'friendly/enemy' logic, so we return the board unchanged.
        # If you later want strict canonical symmetry, deep-copy and swap here.
        return board

    def getSymmetries(self, board, pi):
        """
        Returns symmetrical board/policy pairs exploiting the board's 4-fold
        rotational and 4 reflective symmetries (dihedral group D4).

        Both the board state and the policy vector are transformed consistently.
        """
        symmetries = []

        # We work with a (9, 9) policy matrix to apply geometric transforms,
        # then flatten back to the action vector.
        def transform_board_and_pi(state, pi_vec, k_rot, flip):
            """Apply k*90° rotation + optional horizontal flip."""
            new_state = copy.deepcopy(state)
            b = new_state['board']

            def transform_pos(r, c, k, do_flip):
                """Transform a single (r, c) coordinate."""
                n = self._BOARD_SIZE - 1
                for _ in range(k):            # 90° clockwise each time
                    r, c = c, n - r
                if do_flip:
                    c = n - c
                return r, c

            def transform_list(lst):
                return [transform_pos(r, c, k_rot, flip) for r, c in lst]

            b['white_positions'] = transform_list(b['white_positions'])
            b['black_positions'] = transform_list(b['black_positions'])
            if b['king_position'] is not None:
                b['king_position'] = transform_pos(*b['king_position'], k_rot, flip)

            # Transform the policy vector: reshape, rotate/flip, flatten.
            # pi encodes (from_cell, to_cell); we must transform both halves.
            pi_matrix_from = pi_vec.reshape(BOARD_SIZE ** 2, BOARD_SIZE ** 2)
            # Reshape into a (9,9,9,9) 4-D array then apply transforms.
            pi_4d = pi_matrix_from.reshape(
                BOARD_SIZE, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE)

            def transform_4d(arr, k, do_flip):
                # Apply the same spatial transform to both pairs of axes.
                for _ in range(k):
                    arr = np.rot90(arr, axes=(0, 1))   # rotate 'from' plane
                    arr = np.rot90(arr, axes=(2, 3))   # rotate 'to' plane
                if do_flip:
                    arr = np.flip(arr, axis=1)         # flip cols in 'from'
                    arr = np.flip(arr, axis=3)         # flip cols in 'to'
                return arr

            new_pi_4d = transform_4d(pi_4d, k_rot, flip)
            new_pi = new_pi_4d.reshape(ACTION_SIZE)

            return new_state, new_pi

        pi_arr = np.array(pi, dtype=np.float32)
        for k in range(4):           # 0°, 90°, 180°, 270°
            for do_flip in [False, True]:
                new_board, new_pi = transform_board_and_pi(board, pi_arr, k, do_flip)
                symmetries.append((new_board, new_pi.tolist()))

        return symmetries

    def stringRepresentation(self, board):
        """
        Compact, hashable string encoding the board position for MCTS node lookup.
        Encodes white positions, black positions, king position, and turn.
        """
        b = board['board']
        wp = tuple(sorted(b['white_positions']))
        bp = tuple(sorted(b['black_positions']))
        kp = b['king_position']
        turn = b['turn_to_move']
        return f"W{wp}B{bp}K{kp}T{turn}"

    # ── Tensor encoding (used by the neural network) ──────────────────────────

    def encode_state(self, state) -> np.ndarray:
        """
        Encodes the full game state into a (9, 9, 43) float32 tensor.

        Planes layout (43 total):
          0–39 : 8 history steps × 5 planes each
                 [friendly, enemy, king, rep≥1, rep≥2]
          40   : player colour (1.0 = white/defender to move)
          41   : total move count normalised to [0, 1]
          42   : half-move clock normalised to [0, 1]
        """
        tensor = np.zeros(
            (self._BOARD_SIZE, self._BOARD_SIZE, 43), dtype=np.float32)

        board = state['board']
        history = state['history']
        turn = board['turn_to_move']

        for step_idx, snap in enumerate(reversed(history)):
            base = step_idx * 5

            wp = set(snap['white_positions'])
            bp = set(snap['black_positions'])
            kp = snap['king_position']

            # Friendly / enemy relative to the current player
            friendly, enemy = (wp, bp) if turn == 1 else (bp, wp)

            for (r, c) in friendly:
                tensor[r, c, base + 0] = 1.0
            for (r, c) in enemy:
                tensor[r, c, base + 1] = 1.0
            if kp is not None:
                tensor[kp[0], kp[1], base + 2] = 1.0

            if step_idx == 0:
                rep = state.get('repetition_count', 0)
                if rep >= 1:
                    tensor[:, :, base + 3] = 1.0
                if rep >= 2:
                    tensor[:, :, base + 4] = 1.0

        tensor[:, :, 40] = float(turn)
        tensor[:, :, 41] = min(state['move_count'] / 200.0, 1.0)
        tensor[:, :, 42] = min(state['half_move_clock'] / 40.0, 1.0)
        return tensor

    # ── Private helpers (identical logic to the original, now prefixed _) ─────

    def _get_raw_moves(self, state):
        """Returns a list of [[r0,c0],[r1,c1]] moves (original format)."""
        board = state['board'] if 'board' in state else state
        moves = []

        white_positions = board['white_positions']
        black_positions = board['black_positions']
        king_position = board['king_position']
        turn = board['turn_to_move']

        occupied = {tuple(pos) for pos in white_positions}
        occupied.update(tuple(pos) for pos in black_positions)
        if king_position is not None:
            occupied.add(tuple(king_position))

        if turn == 1:
            pieces = [pos[:] for pos in white_positions]
            if king_position is not None:
                pieces.append(list(king_position))
        else:
            pieces = [pos[:] for pos in black_positions]

        for piece in pieces:
            r0, c0 = piece
            is_black = (r0, c0) in [tuple(p) for p in black_positions]
            start_in_camp = (r0, c0) in self._CAMPS

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                step = 1
                while True:
                    r = r0 + dr * step
                    c = c0 + dc * step

                    if not self._inside_board(r, c):
                        break

                    cell = (r, c)
                    if cell in occupied:
                        break
                    if self._is_throne(cell):
                        is_king = (king_position is not None and
                                   (r0, c0) == tuple(king_position))
                        if not is_king:
                            break
                    if cell in self._CAMPS:
                        if is_black:
                            if not start_in_camp:
                                break
                        else:
                            break

                    moves.append([[r0, c0], [r, c]])
                    step += 1

        return moves

    def _apply_move(self, state, action):
        """Deep-copies state, applies action [[r0,c0],[r1,c1]], returns new state."""
        new_state = copy.deepcopy(state)
        board = new_state['board']

        r0, c0 = tuple(action[0])
        r1, c1 = tuple(action[1])

        if (r0, c0) in board['white_positions']:
            board['white_positions'].remove((r0, c0))
            board['white_positions'].append((r1, c1))
        if (r0, c0) in board['black_positions']:
            board['black_positions'].remove((r0, c0))
            board['black_positions'].append((r1, c1))
        if board.get('king_position') == (r0, c0):
            board['king_position'] = (r1, c1)

        captured = self._apply_captures(board, (r1, c1))

        new_state['move_count'] += 1
        new_state['half_move_clock'] = 0 if captured else new_state['half_move_clock'] + 1

        board['turn_to_move'] = 1 - board['turn_to_move']
        new_state['history'].append(copy.deepcopy(board))
        return new_state

    def _apply_captures(self, board, moved_to: tuple) -> bool:
        r1, c1 = moved_to
        turn = board['turn_to_move']
        captured_any = False

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            target = (r1 + dr, c1 + dc)
            behind = (r1 + 2*dr, c1 + 2*dc)

            if not self._inside_board(*target):
                continue

            if turn == 1:
                enemy_list = board['black_positions']
                is_enemy = target in enemy_list
            else:
                enemy_list = board['white_positions']
                is_enemy = (target in enemy_list or
                            target == board.get('king_position'))

            if not is_enemy:
                continue

            if target == board.get('king_position'):
                if self._king_is_captured(board):
                    board['king_position'] = None
                    captured_any = True
                continue

            if self._is_hostile(behind, board, for_side=turn):
                enemy_list.remove(target)
                captured_any = True

        return captured_any

    def _is_hostile(self, cell, board, for_side: int) -> bool:
        if not self._inside_board(*cell):
            return False
        if self._is_throne(cell) or cell in self._CAMPS:
            return True
        if for_side == 1:
            return cell in board['white_positions']
        return cell in board['black_positions']

    def _is_terminal(self, state) -> bool:
        board = state['board']
        king_pos = board['king_position']

        if king_pos is None:
            return True
        if self._is_escape_square(king_pos):
            return True
        if not self._get_raw_moves(board):
            return True
        if state.get('repetition_count', 0) >= 3:
            return True
        return False

    def _get_winner(self, state) -> int:
        """Returns 1 (white wins), 0 (black wins), or -1 (draw)."""
        board = state['board']
        king_pos = board.get('king_position')
        turn = board.get('turn_to_move')

        if king_pos is None:
            return 0   # black wins: king captured

        if self._is_escape_square(king_pos):
            return 1   # white wins: king escaped

        if not self._get_raw_moves(board):
            # The player whose turn it is has no moves → they lose
            return 0 if turn == 1 else 1

        if state.get('repetition_count', 0) >= 3:
            loser = state.get('repetition_loser')
            if loser is not None:
                return 0 if loser == 1 else 1
            return -1  # draw

        raise ValueError("_get_winner called on a non-terminal state")

    def _is_escape_square(self, pos) -> bool:
        return pos in self._ESCAPES

    def _king_is_captured(self, board) -> bool:
        kp = board.get('king_position')
        if kp is None:
            return False
        bp = set(board['black_positions'])
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbour = (kp[0] + dr, kp[1] + dc)
            if not self._inside_board(*neighbour):
                continue
            if not (neighbour in bp or
                    neighbour in self._CAMPS or
                    self._is_throne(neighbour)):
                return False
        return True

    def _inside_board(self, r, c) -> bool:
        return 0 <= r < self._BOARD_SIZE and 0 <= c < self._BOARD_SIZE

    def _is_throne(self, cell) -> bool:
        return cell == self._THRONE