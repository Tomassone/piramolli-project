import numpy as np
import copy
from collections import deque
import encoding

# STATO:
# Tensore 9*9*(7+36), gli "strati" rappresentano le seguenti informazioni:
# - Strato 0: posizione dei pezzi bianchi (1 se c'è un pezzo bianco, 0 altrimenti)
# - Strato 1: posizione dei pezzi neri (1 se c'è un pezzo nero, 0 altrimenti)
# - Strato 2: posizione del re
# - Strato 3: posizione del trono
# - Strato 4: posizione dei camps
# - Strato 5: posizione delle vie di fuga
# - Strato 6: turno del giocatore (1 per bianco, 0 per nero)


class Game:
    # Stato contiene white, black, re, turn_to_move
    BOARD_SIZE = 9
    THRONE = (4, 4)
    CAMPS = {
        # Top
        (0, 3), (0, 4), (0, 5), (1, 4),
        # Bottom
        (8, 3), (8, 4), (8, 5), (7, 4),
        # Left
        (3, 0), (4, 0), (5, 0), (4, 1),
        # Right
        (3, 8), (4, 8), (5, 8), (4, 7),
    }
    ESCAPES = encoding.ESCAPES
    HISTORY_LEN = 8

    def get_initial_state(self):
        """
        Returns the full game state, now including:
        - the raw board dict (for logic)
        - a history deque of the last 8 board snapshots (for tensor encoding)
        - auxiliary counters
        """
        board = {
            'white_positions': [
                (2, 4), (3, 4), (4, 2), (4, 3),
                (4, 5), (4, 6), (5, 4), (6, 4)
            ],
            'black_positions': [
                (0, 3), (0, 4), (0, 5), (1, 4),
                (3, 0), (4, 0), (5, 0), (4, 1),
                (8, 3), (8, 4), (8, 5), (7, 4),
                (3, 8), (4, 8), (5, 8), (4, 7)
            ],
            'king_position': (4, 4),
            'turn_to_move': 1,  # 1 = white (defender), 0 = black (attacker)
        }
        # History: deque of raw board snapshots (most recent last)
        # Initialise by repeating the starting position 8 times
        history = deque(
            [copy.deepcopy(board) for _ in range(self.HISTORY_LEN)],
            maxlen=self.HISTORY_LEN
        )
        return {
            'board': board,
            'history': history,
            'move_count': 0,
            'half_move_clock': 0,       # resets on capture
            'repetition_count': 0,
            'repetition_loser': None,
        }

    def encode_state(self, state) -> np.ndarray:
        """
        Encodes the full game state into a (9, 9, 43) float32 tensor.

        Planes layout (43 total):
          0-39 : 8 history steps × 5 planes each
                 [friendly, enemy, king, rep>=1, rep>=2]
          40   : player color (1 = white/defender to move)
          41   : total move count normalised to [0, 1]
          42   : half-move clock normalised to [0, 1]
        """
        tensor = np.zeros(
            (self.BOARD_SIZE, self.BOARD_SIZE, 43), dtype=np.float32)

        board = state['board']
        # deque, index 0 = oldest, -1 = most recent
        history = state['history']
        turn = board['turn_to_move']  # 1 = white, 0 = black

        # --- History planes (5 planes × 8 steps = 40) ---
        # We iterate newest → oldest so that plane 0 = most recent
        for step_idx, snap in enumerate(reversed(history)):
            base = step_idx * 5   # 5 planes per step

            wp = set(snap['white_positions'])
            bp = set(snap['black_positions'])
            kp = snap['king_position']

            # "Friendly" and "enemy" are relative to the current player
            if turn == 1:   # white to move: friendly=white, enemy=black
                friendly, enemy = wp, bp
            else:           # black to move: friendly=black, enemy=white
                friendly, enemy = bp, wp

            # Plane 0: friendly taflmen (excluding king)
            for (r, c) in friendly:
                tensor[r, c, base + 0] = 1.0

            # Plane 1: enemy taflmen
            for (r, c) in enemy:
                tensor[r, c, base + 1] = 1.0

            # Plane 2: king
            if kp is not None:
                tensor[kp[0], kp[1], base + 2] = 1.0

            # Planes 3-4: repetition flags (set from current state, not history snap)
            # Only meaningful for the most recent step
            if step_idx == 0:
                rep = state.get('repetition_count', 0)
                if rep >= 1:
                    tensor[:, :, base + 3] = 1.0
                if rep >= 2:
                    tensor[:, :, base + 4] = 1.0

        # --- Auxiliary planes ---
        # 1 if white to move, 0 if black
        tensor[:, :, 40] = float(turn)
        tensor[:, :, 41] = min(state['move_count'] / 200.0, 1.0)   # normalised
        tensor[:, :, 42] = min(state['half_move_clock'] / 40.0, 1.0)

        return tensor

    def get_valid_moves(self, state):
        # Support both nested state and flat board dict
        if 'board' in state:
            state = state['board']
        moves = []

        white_positions = state["white_positions"]
        black_positions = state["black_positions"]
        king_position = state["king_position"]
        turn = state["turn_to_move"]

        occupied = {tuple(pos) for pos in white_positions}
        occupied.update(tuple(pos) for pos in black_positions)
        if king_position is not None:
            occupied.add(tuple(king_position))

        if turn == 1:
            pieces = [pos[:] for pos in white_positions]
            if king_position is not None:
                pieces.append(king_position[:])
        else:
            pieces = [pos[:] for pos in black_positions]

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for piece in pieces:
            r0, c0 = piece
            is_king = (king_position is not None and piece == king_position)
            is_black = piece in black_positions
            start_in_camp = (r0, c0) in self.CAMPS

            for dr, dc in directions:
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
                        if not is_king:
                            break

                    if cell in self.CAMPS:
                        if is_black:
                            if not start_in_camp:
                                break
                        else:
                            break

                    moves.append([[r0, c0], [r, c]])
                    step += 1

        return moves

    def _inside_board(self, r, c):
        return 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE

    def _is_throne(self, cell):
        return cell[0] == self.THRONE[0] and cell[1] == self.THRONE[1]

    def get_next_state(self, state, action):
        new_state = copy.deepcopy(state)
        board = new_state['board']

        r0, c0 = tuple(action[0])
        r1, c1 = tuple(action[1])

        captured = False

        # --- Move pieces ---
        if (r0, c0) in board['white_positions']:
            board['white_positions'].remove((r0, c0))
            board['white_positions'].append((r1, c1))

        if (r0, c0) in board['black_positions']:
            board['black_positions'].remove((r0, c0))
            board['black_positions'].append((r1, c1))

        if board.get('king_position') == (r0, c0):
            board['king_position'] = (r1, c1)

        # --- Capture logic ---
        captured = self._apply_captures(board, (r1, c1))

        # --- Counters ---
        new_state['move_count'] += 1
        if captured:
            new_state['half_move_clock'] = 0
        else:
            new_state['half_move_clock'] += 1

        # --- Toggle turn ---
        board['turn_to_move'] = 1 - board['turn_to_move']

        # --- Push snapshot to history ---
        new_state['history'].append(copy.deepcopy(board))

        return new_state

    def _apply_captures(self, board, moved_to: tuple) -> bool:
        """
        After a piece moves to `moved_to`, check all 4 neighbours for captures.
        Returns True if at least one piece was captured.
        """
        r1, c1 = moved_to
        turn = board['turn_to_move']   # still the mover's turn at this point
        captured_any = False

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in directions:
            target = (r1 + dr, c1 + dc)
            behind = (r1 + 2*dr, c1 + 2*dc)

            if not self._inside_board(*target):
                continue

            # Is there an enemy piece at target?
            if turn == 1:   # white moved
                enemy_list = board['black_positions']
                is_enemy = target in enemy_list
            else:           # black moved
                enemy_list = board['white_positions']
                is_enemy = target in enemy_list or target == board.get(
                    'king_position')

            if not is_enemy:
                continue

            # King requires 4-sided capture — handled separately
            if target == board.get('king_position'):
                if self._king_is_captured(board):
                    board['king_position'] = None
                    captured_any = True
                continue

            # Normal piece: check if `behind` is a hostile square
            if self._is_hostile(behind, board, for_side=turn):
                enemy_list.remove(target)
                captured_any = True

        return captured_any

    def _is_hostile(self, cell, board, for_side: int) -> bool:
        if not self._inside_board(*cell):
            return False
        if self._is_throne(cell):
            return True
        if cell in self.CAMPS:
            return True
        if for_side == 1:
            return cell in board['white_positions']
        else:
            return cell in board['black_positions']

    def is_terminal(self, state) -> bool:
        board = state['board']
        king_pos = board["king_position"]

        if king_pos is None:
            return True

        if self._is_escape_square(king_pos):
            return True

        valid_moves = self.get_valid_moves(board)
        if len(valid_moves) == 0:
            return True

        repetition_count = state.get("repetition_count", 0)
        if repetition_count >= 3:
            return True

        return False

    def get_winner(self, state):
        board = state['board']
        king_pos = board.get("king_position")
        turn_to_move = board.get("turn_to_move")

        if king_pos is None:
            return 0

        if self._is_escape_square(king_pos):
            return 1

        valid_moves = self.get_valid_moves(board)
        if len(valid_moves) == 0:
            if turn_to_move == 1:
                return 0
            else:
                return 1

        repetition_count = state.get("repetition_count", 0)
        if repetition_count >= 3:
            repetition_loser = state.get("repetition_loser")
            if repetition_loser is not None:
                return 0 if repetition_loser == 1 else 1
            return -1

        raise ValueError("get_winner called on a non-terminal state")

    def _is_escape_square(self, pos) -> bool:
        return pos in encoding.ESCAPES

    def _king_is_captured(self, board) -> bool:
        """
        King must be surrounded on all 4 sides by black pieces,
        camps, or the throne. If on/adjacent to the throne, all
        4 sides must be covered (throne counts as one side).
        """
        kp = board.get('king_position')
        if kp is None:
            return False

        bp = set(board['black_positions'])
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in directions:
            neighbour = (kp[0] + dr, kp[1] + dc)
            if not self._inside_board(*neighbour):
                continue  # edge counts as blocking? — adjust per your ruleset
            is_blocking = (
                neighbour in bp
                or neighbour in self.CAMPS
                or self._is_throne(neighbour)
            )
            if not is_blocking:
                return False
        return True
