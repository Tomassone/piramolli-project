import numpy as np
import copy
from collections import deque
import encoding

BOARD_SIZE = 9
ACTION_SIZE = BOARD_SIZE ** 4  # 6561


def _encode_action(r0, c0, r1, c1) -> int:
    return (r0 * BOARD_SIZE + c0) * (BOARD_SIZE ** 2) + (r1 * BOARD_SIZE + c1)


def _decode_action(idx: int):
    from_idx, to_idx = divmod(idx, BOARD_SIZE ** 2)
    r0, c0 = divmod(from_idx, BOARD_SIZE)
    r1, c1 = divmod(to_idx, BOARD_SIZE)
    return r0, c0, r1, c1


def _internal_player(base_player: int) -> int:
    """Base-class convention (1 / -1) → internal (1 / 0)."""
    return 1 if base_player == 1 else 0


def _base_player(internal_player: int) -> int:
    """Internal (1 / 0) → base-class convention (1 / -1)."""
    return 1 if internal_player == 1 else -1


class Game:
    """
    Tablut / Hnefatafl conforming to the alpha-zero-general Game interface,
    with rules matched to the Unibo GameAshtonTablut Java server.
    """

    _BOARD_SIZE = 9
    _THRONE = (4, 4)

    # These are the (row, col) equivalents of the Java citadel strings.
    # Java uses chess notation (e.g. "a4" = col 0, row 3), so we translate:
    #   column letter a-i → 0-8, row number 1-9 → 0-8 (row = number - 1)
    _CAMPS = {
        # Left (column a = col 0): a4, a5, a6, b5
        (3, 0), (4, 0), (5, 0), (4, 1),
        # Top (row 1 = row 0): d1, e1, f1, e2
        (0, 3), (0, 4), (0, 5), (1, 4),
        # Right (column i = col 8): i4, i5, i6, h5
        (3, 8), (4, 8), (5, 8), (4, 7),
        # Bottom (row 9 = row 8): d9, e9, f9, e8
        (8, 3), (8, 4), (8, 5), (7, 4),
    }

    # The four "deep" camp cells that the Java explicitly excludes from acting
    # as the second jaw of a white-captures-black sandwich. These correspond
    # to the Java's "strangeCitadels" comments: e1, a5, i5, e9.
    #   e1 = (0,4), a5 = (4,0), i5 = (4,8), e9 = (8,4)
    _STRANGE_CAMPS = {(0, 4), (4, 0), (4, 8), (8, 4)}

    _ESCAPES = encoding.ESCAPES
    _HISTORY_LEN = 8

    # Maximum distance a black piece can travel within its own camp group.
    # The Java rejects moves where |col_from - col_to| > 5 or |row_from - row_to| > 5
    # when both endpoints are inside citadels. This prevents crossing from one
    # camp cluster to the opposite one in a single move.
    _MAX_CAMP_SLIDE = 5

    def __init__(self):
        pass

    # ── Base interface ────────────────────────────────────────────────────────

    def getInitBoard(self):
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
            'turn_to_move': 1,  # 1 = white (defender)
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
            # We now store a list of board string-hashes seen since the last
            # capture, mirroring the Java's drawConditions list.
            'draw_history': [],
            'repetition_count': 0,
            'repetition_loser': None,
        }

    def getBoardSize(self):
        return (self._BOARD_SIZE, self._BOARD_SIZE)

    def getActionSize(self):
        return ACTION_SIZE

    def getNextState(self, board, player, action):
        r0, c0, r1, c1 = _decode_action(action)
        new_state = self._apply_move(board, [[r0, c0], [r1, c1]])
        next_internal = new_state['board']['turn_to_move']
        return new_state, _base_player(next_internal)

    def getValidMoves(self, board, player):
        raw_moves = self._get_raw_moves(board)
        valid = np.zeros(ACTION_SIZE, dtype=np.int8)
        for move in raw_moves:
            (r0, c0), (r1, c1) = move
            valid[_encode_action(r0, c0, r1, c1)] = 1
        return valid

    def getGameEnded(self, board, player):
        if not self._is_terminal(board):
            return 0
        winner_internal = self._get_winner(board)
        if winner_internal == -1:
            return 1e-4  # draw
        winner_base = _base_player(winner_internal)
        return 1 if winner_base == player else -1

    def getCanonicalForm(self, board, player):
        # Tablut has no colour-symmetric structure; the canonical form is
        # always from white's perspective, which encode_state already handles.
        return board

    def getSymmetries(self, board, pi):
        symmetries = []
        pi_arr = np.array(pi, dtype=np.float32)

        def transform_pos(r, c, k_rot, do_flip):
            n = self._BOARD_SIZE - 1
            for _ in range(k_rot):
                r, c = c, n - r
            if do_flip:
                c = n - c
            return r, c

        def transform_state(state, k_rot, do_flip):
            new_state = copy.deepcopy(state)
            b = new_state['board']
            b['white_positions'] = [
                transform_pos(r, c, k_rot, do_flip)
                for r, c in b['white_positions']
            ]
            b['black_positions'] = [
                transform_pos(r, c, k_rot, do_flip)
                for r, c in b['black_positions']
            ]
            if b['king_position'] is not None:
                b['king_position'] = transform_pos(
                    *b['king_position'], k_rot, do_flip)
            return new_state

        def transform_pi(pi_vec, k_rot, do_flip):
            pi_4d = pi_vec.reshape(
                BOARD_SIZE, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE)
            for _ in range(k_rot):
                pi_4d = np.rot90(pi_4d, axes=(0, 1))
                pi_4d = np.rot90(pi_4d, axes=(2, 3))
            if do_flip:
                pi_4d = np.flip(pi_4d, axis=1)
                pi_4d = np.flip(pi_4d, axis=3)
            return pi_4d.reshape(ACTION_SIZE)

        for k in range(4):
            for do_flip in [False, True]:
                new_board = transform_state(board, k, do_flip)
                new_pi = transform_pi(pi_arr, k, do_flip)
                symmetries.append((new_board, new_pi.tolist()))

        return symmetries

    def stringRepresentation(self, board):
        b = board['board']
        wp = tuple(sorted(b['white_positions']))
        bp = tuple(sorted(b['black_positions']))
        kp = b['king_position']
        turn = b['turn_to_move']
        return f"W{wp}B{bp}K{kp}T{turn}"

    # ── Tensor encoding ───────────────────────────────────────────────────────

    def encode_state(self, state) -> np.ndarray:
        """
        Encodes the full game state into a (9, 9, 43) float32 tensor.

        Planes 0-39: 8 history steps × 5 planes [friendly, enemy, king, rep≥1, rep≥2]
        Plane 40: player colour (1.0 = white to move)
        Plane 41: move count normalised
        Plane 42: half-move clock normalised
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

    # ── Move generation ───────────────────────────────────────────────────────

    def _get_raw_moves(self, state):
        """
        Generates all legal moves as [[r0,c0],[r1,c1]] pairs.

        Key rules matched to the Java validator:
          - No piece (except the king) may enter or pass through the throne.
          - White pieces may never enter or pass through a camp.
          - Black pieces may enter camps only if they start inside one, AND
            the destination is within _MAX_CAMP_SLIDE squares (prevents
            sliding from one camp cluster to the opposite one).
          - No piece may jump over another piece or the throne.
        """
        board = state['board'] if 'board' in state else state
        moves = []

        wp = set(map(tuple, board['white_positions']))
        bp = set(map(tuple, board['black_positions']))
        kp = board.get('king_position')
        turn = board['turn_to_move']

        occupied = wp | bp
        if kp is not None:
            occupied.add(kp)

        pieces = list(wp | ({kp} if kp else set())) if turn == 1 else list(bp)

        for piece in pieces:
            r0, c0 = piece
            is_king = (kp is not None and piece == kp)
            is_black = piece in bp
            start_in_camp = (r0, c0) in self._CAMPS

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                step = 1
                while True:
                    r = r0 + dr * step
                    c = c0 + dc * step

                    if not self._inside_board(r, c):
                        break
                    if (r, c) in occupied:
                        break

                    # The throne blocks everyone except the king.
                    if self._is_throne((r, c)) and not is_king:
                        break

                    if (r, c) in self._CAMPS:
                        if is_black:
                            # Black can only re-enter its own camp cluster,
                            # and cannot slide more than 5 squares to do so.
                            if not start_in_camp:
                                break
                            dist = abs(r - r0) + abs(c - c0)
                            if dist > self._MAX_CAMP_SLIDE:
                                break
                        else:
                            # White (and the king) can never enter a camp.
                            break

                    moves.append([[r0, c0], [r, c]])
                    step += 1

        return moves

    # ── State transition ──────────────────────────────────────────────────────

    def _apply_move(self, state, action):
        """
        Deep-copies state, moves the piece, applies captures, updates all
        counters (including the draw_history list), and toggles the turn.
        """
        new_state = copy.deepcopy(state)
        board = new_state['board']

        r0, c0 = tuple(action[0])
        r1, c1 = tuple(action[1])

        # Move the piece on the board.
        if (r0, c0) in board['white_positions']:
            board['white_positions'].remove((r0, c0))
            board['white_positions'].append((r1, c1))
        if (r0, c0) in board['black_positions']:
            board['black_positions'].remove((r0, c0))
            board['black_positions'].append((r1, c1))
        if board.get('king_position') == (r0, c0):
            board['king_position'] = (r1, c1)

        captured = self._apply_captures(board, (r1, c1))

        # Counters — mirroring movesWithoutCapturing in the Java.
        new_state['move_count'] += 1
        if captured:
            new_state['half_move_clock'] = 0
            # Clear draw history on capture, exactly as the Java does.
            new_state['draw_history'] = []
            new_state['repetition_count'] = 0
        else:
            new_state['half_move_clock'] += 1

        board['turn_to_move'] = 1 - board['turn_to_move']

        # --- Repetition detection (mirrors Java's drawConditions list) ---
        # We hash the new board position and count how many times it has
        # appeared in draw_history since the last capture.
        board_hash = self._board_hash(board)
        occurrences = new_state['draw_history'].count(board_hash)
        new_state['repetition_count'] = occurrences

        # Java uses repeated_moves_allowed = 2 by default before declaring draw,
        # meaning the position must appear more than 2 times (i.e. 3rd occurrence).
        new_state['draw_history'].append(board_hash)

        new_state['history'].append(copy.deepcopy(board))
        return new_state

    def _board_hash(self, board) -> str:
        """Compact string uniquely identifying a board position for draw detection."""
        wp = tuple(sorted(board['white_positions']))
        bp = tuple(sorted(board['black_positions']))
        kp = board['king_position']
        turn = board['turn_to_move']
        return f"W{wp}B{bp}K{kp}T{turn}"

    # ── Capture logic ─────────────────────────────────────────────────────────

    def _apply_captures(self, board, moved_to: tuple) -> bool:
        """
        Dispatcher: after a piece moves to `moved_to`, check all four
        directions for captures and return True if anything was captured.
        """
        turn = board['turn_to_move']  # still the mover's turn at this point

        if turn == 1:
            return self._captures_by_white(board, moved_to)
        else:
            return self._captures_by_black(board, moved_to)

    def _captures_by_white(self, board, moved_to: tuple) -> bool:
        """
        White just moved to `moved_to`. Check if any adjacent black pieces
        are sandwiched. Mirrors Java's checkCaptureWhite().

        A black piece at `target` is captured if the cell `behind` it
        (on the far side from the moving piece) is:
          - a white piece, OR
          - the throne, OR
          - the king, OR
          - a camp cell that is NOT one of the four _STRANGE_CAMPS.
        """
        r1, c1 = moved_to
        captured_any = False
        bp = board['black_positions']

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            target = (r1 + dr, c1 + dc)
            behind = (r1 + 2*dr, c1 + 2*dc)

            if not self._inside_board(*target):
                continue
            if target not in bp:
                continue
            if not self._inside_board(*behind):
                continue

            behind_is_anvil = (
                behind in board['white_positions']
                or self._is_throne(behind)
                or behind == board.get('king_position')
                or (behind in self._CAMPS and behind not in self._STRANGE_CAMPS)
            )

            if behind_is_anvil:
                bp.remove(target)
                captured_any = True

        return captured_any

    def _captures_by_black(self, board, moved_to: tuple) -> bool:
        """
        Black just moved to `moved_to`. Mirrors Java's checkCaptureBlack(),
        which delegates to separate pawn and king capture methods.
        """
        captured_any = False

        # --- Capture normal white pawns (not the king) ---
        r1, c1 = moved_to
        wp = board['white_positions']
        kp = board.get('king_position')

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            target = (r1 + dr, c1 + dc)
            behind = (r1 + 2*dr, c1 + 2*dc)

            if not self._inside_board(*target):
                continue

            # Skip the king — he is handled separately below.
            if target == kp:
                continue

            if target not in wp:
                continue
            if not self._inside_board(*behind):
                continue

            # A white pawn is captured if behind it is: another black piece,
            # the throne (empty or not), or a camp cell.
            # This mirrors checkCaptureBlackPawn{Right,Left,Up,Down}.
            behind_is_anvil = (
                behind in board['black_positions']
                or self._is_throne(behind)   # Java checks "T" (throne marker)
                or behind in self._CAMPS
            )

            if behind_is_anvil:
                wp.remove(target)
                captured_any = True

        # --- Capture the king (position-sensitive) ---
        if kp is not None and self._king_is_captured(board, moved_to):
            board['king_position'] = None
            captured_any = True

        return captured_any

    def _king_is_captured(self, board, moved_to: tuple) -> bool:
        """
        Determines whether the king is captured after black moves to `moved_to`.

        This faithfully mirrors the Java's four directional methods
        (checkCaptureBlackKing{Left,Right,Up,Down}), which apply different
        rules depending on the king's position relative to the throne:

        - King ON the throne (e5 = 4,4): needs all 4 sides covered by black.
        - King ADJACENT to the throne (e4, e6, d5, f5): needs 3 sides covered
          (the throne itself counts as one side automatically).
        - King ELSEWHERE: standard 2-piece sandwich — the cell behind must be
          black or a camp cell.

        We check only the direction in which `moved_to` is adjacent to the
        king, matching how the Java triggers one of the four directional methods.
        """
        kp = board.get('king_position')
        if kp is None:
            return False

        kr, kc = kp
        bp = set(board['black_positions'])

        # Is the piece that just moved actually adjacent to the king?
        mr, mc = moved_to
        if abs(mr - kr) + abs(mc - kc) != 1:
            return False  # not adjacent — no capture possible from this move

        def is_blocking(cell):
            """True if this cell counts as a hostile blocker for king capture."""
            return (cell in bp or cell in self._CAMPS or self._is_throne(cell))

        # ── King is ON the throne (4,4) ───────────────────────────────────
        # Java: all four neighbours must be black (throne has no passive help
        # here because the king *is* on the throne).
        if kp == self._THRONE:
            return all(
                is_blocking((kr + dr, kc + dc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if self._inside_board(kr + dr, kc + dc)
            )

        # ── King is ADJACENT to the throne ───────────────────────────────
        # The throne always counts as one blocking side. We need the remaining
        # three sides (i.e. all sides except the throne direction) to be covered
        # by black pieces. The Java hardcodes these checks per cell, which we
        # replicate here generically.
        throne_adjacent = {(3, 4), (5, 4), (4, 3), (4, 5)}  # e4,e6,d5,f5
        if kp in throne_adjacent:
            # Count how many of the four sides are blocked (throne counts too).
            blocked_sides = sum(
                1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if self._inside_board(kr + dr, kc + dc)
                and is_blocking((kr + dr, kc + dc))
            )
            # The king is adjacent to the throne, so at least one of those
            # is_blocking() calls will return True for the throne cell.
            # Capture requires all exposed sides (all 4 for a non-edge cell,
            # which all throne-adjacent cells are) to be covered.
            return blocked_sides == 4

        # ── King is ELSEWHERE ────────────────────────────────────────────
        # Standard sandwich: the cell directly opposite the moving piece
        # (i.e. `behind` relative to the mover) must be black or a camp.
        dr = kr - mr   # direction from mover toward king
        dc = kc - mc
        behind = (kr + dr, kc + dc)   # cell on far side of king

        if not self._inside_board(*behind):
            # King is against the board edge — cannot be captured this way
            # (the Java also does not capture in this situation since the
            # "behind" index would be out of bounds).
            return False

        return is_blocking(behind)

    def _is_hostile(self, cell, board, for_side: int) -> bool:
        """General hostility check used internally (not for king capture)."""
        if not self._inside_board(*cell):
            return False
        if self._is_throne(cell) or cell in self._CAMPS:
            return True
        if for_side == 1:
            return cell in board['white_positions']
        return cell in board['black_positions']

    # ── Terminal / winner ─────────────────────────────────────────────────────

    def _is_terminal(self, state) -> bool:
        board = state['board']
        kp = board['king_position']

        if kp is None:
            return True
        if self._is_escape_square(kp):
            return True
        if not self._get_raw_moves(board):
            return True
        # Draw: position has appeared more than repeated_moves_allowed times.
        # We use 2 here, matching the Java default of repeated_moves_allowed=2,
        # meaning a position appearing a 3rd time triggers a draw.
        if state.get('repetition_count', 0) >= 2:
            return True
        return False

    def _get_winner(self, state) -> int:
        """Returns 1 (white), 0 (black), or -1 (draw)."""
        board = state['board']
        kp = board.get('king_position')
        turn = board.get('turn_to_move')

        if kp is None:
            return 0  # black wins: king captured

        if self._is_escape_square(kp):
            return 1  # white wins: king escaped

        if not self._get_raw_moves(board):
            # Player whose turn it is cannot move → they lose.
            return 0 if turn == 1 else 1

        if state.get('repetition_count', 0) >= 2:
            return -1  # draw

        raise ValueError("_get_winner called on a non-terminal state")

    def _is_escape_square(self, pos) -> bool:
        return pos in self._ESCAPES

    # ── Utility ───────────────────────────────────────────────────────────────

    def _inside_board(self, r, c) -> bool:
        return 0 <= r < self._BOARD_SIZE and 0 <= c < self._BOARD_SIZE

    def _is_throne(self, cell) -> bool:
        return cell == self._THRONE