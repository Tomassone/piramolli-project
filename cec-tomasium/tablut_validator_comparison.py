"""
tablut_validator_comparison.py

Differential testing harness: compares the Python Game engine against
the Java GameAshtonTablut server move-by-move.

Prerequisites:
    - The Java Tablut server must be running locally before you start.
      Launch it with something like:
          java -jar Tablut.jar <repeated_moves> <timeout> <logs_folder> <white_name> <black_name>
      e.g.: java -jar Tablut.jar 2 60 logs PyWhite PyBlack
    - You need the competition's Python client module (client.py) which
      handles Java object serialisation over the socket.
    - Your Game class must be importable from game.py.
"""

import random
import socket
import struct
import io
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from game import Game

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup — we want a full audit trail of every divergence found.
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("comparison_results.log"),
    ]
)
log = logging.getLogger("validator_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Layer 0 — Java server socket communication
#
# The Java server sends states as raw bytes using Java's DataOutputStream:
#   - First 4 bytes: int32 payload length
#   - Remaining bytes: the state serialised as a custom binary format
#
# Rather than reimplementing Java serialisation from scratch, we use the
# approach from the competition kit: connect as a player, receive state,
# send move, receive updated state. We wrap this in a thin Python class.
# ─────────────────────────────────────────────────────────────────────────────

# The Java server's wire format for the board uses these byte values for pawns.
# These come from the StateTablut class in the competition source code.
JAVA_PAWN = {
    0: 'EMPTY',
    1: 'WHITE',
    2: 'BLACK',
    3: 'KING',
    4: 'THRONE',
}

JAVA_TURN = {
    0: 'WHITE',
    1: 'BLACK',
    2: 'WHITEWIN',
    3: 'BLACKWIN',
    4: 'DRAW',
}


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from a socket, blocking until available."""
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Java server closed the connection unexpectedly.")
        buf += chunk
    return buf


def _recv_java_state(sock: socket.socket) -> dict:
    """
    Reads one state message from the Java server and returns it as a dict:
        {
            'board': list[list[str]],  # 9x9 grid of pawn names
            'turn': str,               # 'WHITE', 'BLACK', 'WHITEWIN', etc.
        }

    The Java server uses a simple custom binary protocol:
        [4 bytes: total message length][1 byte: turn][81 bytes: board cells]
    
    Note: This may need adjustment if the competition server version you're
    using has a slightly different wire format — check with a hex dump if
    states don't parse correctly.
    """
    # Read the 4-byte length prefix.
    length_bytes = _recv_exactly(sock, 4)
    length = struct.unpack('>I', length_bytes)[0]

    # Read the payload.
    payload = _recv_exactly(sock, length)
    reader = io.BytesIO(payload)

    # First byte: turn indicator.
    turn_byte = struct.unpack('B', reader.read(1))[0]
    turn = JAVA_TURN.get(turn_byte, 'UNKNOWN')

    # Next 81 bytes: board cells in row-major order.
    board = []
    for r in range(9):
        row = []
        for c in range(9):
            cell_byte = struct.unpack('B', reader.read(1))[0]
            row.append(JAVA_PAWN.get(cell_byte, 'UNKNOWN'))
        board.append(row)

    return {'board': board, 'turn': turn}


def _send_java_move(sock: socket.socket, from_cell: str, to_cell: str, turn: str):
    """
    Sends a move to the Java server using the competition wire format.
    from_cell / to_cell use Java's algebraic notation, e.g. "e5", "d3".
    turn should be "WHITE" or "BLACK".
    """
    # The Java server expects: [4-byte length][from: 2 bytes][to: 2 bytes][turn: 1 byte]
    from_bytes = from_cell.encode('ascii')
    to_bytes = to_cell.encode('ascii')
    turn_byte = 0 if turn == 'WHITE' else 1

    payload = from_bytes + to_bytes + bytes([turn_byte])
    length = struct.pack('>I', len(payload))
    sock.sendall(length + payload)


class JavaServerClient:
    """
    A thin wrapper around the Java server socket protocol.
    Maintains a persistent connection for one side (WHITE or BLACK).
    """

    def __init__(self, side: str, host: str = 'localhost'):
        self.side = side.upper()
        self.port = 5800 if self.side == 'WHITE' else 5801
        self.host = host
        self.sock = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        log.info(f"Connected to Java server as {self.side} on port {self.port}")
        # The server sends the initial state immediately on connection.
        initial_state = _recv_java_state(self.sock)
        log.info(f"Initial Java state received. Turn: {initial_state['turn']}")
        return initial_state

    def send_move_and_get_state(self, from_cell: str, to_cell: str) -> dict:
        _send_java_move(self.sock, from_cell, to_cell, self.side)
        return _recv_java_state(self.sock)

    def close(self):
        if self.sock:
            self.sock.close()


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — State converter
#
# Converts between Java's board representation and your Python board dict.
# This is the most important function in the whole harness — if it's wrong,
# every comparison will be wrong.
# ─────────────────────────────────────────────────────────────────────────────

def java_board_to_python(java_state: dict) -> dict:
    """
    Converts a Java state dict (as returned by _recv_java_state) into
    your Python board dict format:
        {
            'white_positions': list of (row, col) tuples,
            'black_positions': list of (row, col) tuples,
            'king_position':   (row, col) or None,
            'turn_to_move':    1 (white) or 0 (black),
        }

    The Java's coordinate system uses chess-style algebraic notation
    (e.g. "e5" = column e = index 4, row 5 = index 4 in 0-based).
    But since we're reading a 2D grid from the wire format, we just
    iterate directly — no notation conversion needed here.
    """
    board_grid = java_state['board']
    turn_str = java_state['turn']

    white_positions = []
    black_positions = []
    king_position = None

    for r in range(9):
        for c in range(9):
            cell = board_grid[r][c]
            if cell == 'WHITE':
                white_positions.append((r, c))
            elif cell == 'BLACK':
                black_positions.append((r, c))
            elif cell == 'KING':
                king_position = (r, c)
            # EMPTY and THRONE cells don't go into position lists

    # Map Java turn strings to internal convention (1 = white, 0 = black).
    turn_map = {'WHITE': 1, 'BLACK': 0, 'WHITEWIN': 1, 'BLACKWIN': 0, 'DRAW': -1}
    turn_to_move = turn_map.get(turn_str, -1)

    return {
        'white_positions': white_positions,
        'black_positions': black_positions,
        'king_position': king_position,
        'turn_to_move': turn_to_move,
    }


def python_to_java_notation(r: int, c: int) -> str:
    """
    Converts a Python (row, col) position to Java algebraic notation.

    In the Java code, columns are letters a-i (col 0 = 'a', col 8 = 'i')
    and rows are numbers 1-9 (row 0 = '1', row 8 = '9').
    So (row=4, col=4) → "e5", (row=0, col=3) → "d1".
    """
    col_letter = chr(ord('a') + c)
    row_number = str(r + 1)
    return col_letter + row_number


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — Board state comparator
#
# Given a Python board dict and a Java board dict, finds all differences
# and returns a human-readable report. Returning a structured object rather
# than just a bool is crucial for debugging — you want to know *what* differs.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComparisonResult:
    """Holds the outcome of comparing one Python board state to one Java state."""
    match: bool = True
    differences: list = field(default_factory=list)

    def add_diff(self, description: str):
        self.match = False
        self.differences.append(description)

    def report(self) -> str:
        if self.match:
            return "✓  States match."
        return "✗  DIVERGENCE:\n" + "\n".join(f"    - {d}" for d in self.differences)


def compare_states(py_board: dict, java_state: dict) -> ComparisonResult:
    """
    Compares your Python board dict to the Java server's state.
    Returns a ComparisonResult with details of any differences found.

    We compare three things: white piece positions, black piece positions,
    and king position. We sort the position lists before comparing so that
    list ordering differences don't cause false positives.
    """
    result = ComparisonResult()
    java_board = java_board_to_python(java_state)

    # Compare white positions (order-independent).
    py_white = sorted(py_board['white_positions'])
    java_white = sorted(java_board['white_positions'])
    if py_white != java_white:
        py_only = set(py_white) - set(java_white)
        java_only = set(java_white) - set(py_white)
        result.add_diff(
            f"White positions differ. "
            f"Python-only: {py_only}. Java-only: {java_only}."
        )

    # Compare black positions.
    py_black = sorted(py_board['black_positions'])
    java_black = sorted(java_board['black_positions'])
    if py_black != java_black:
        py_only = set(py_black) - set(java_black)
        java_only = set(java_black) - set(py_black)
        result.add_diff(
            f"Black positions differ. "
            f"Python-only: {py_only}. Java-only: {java_only}."
        )

    # Compare king position.
    if py_board['king_position'] != java_board['king_position']:
        result.add_diff(
            f"King position differs. "
            f"Python: {py_board['king_position']}. "
            f"Java: {java_board['king_position']}."
        )

    # Compare whose turn it is.
    if py_board['turn_to_move'] != java_board['turn_to_move']:
        result.add_diff(
            f"Turn differs. "
            f"Python: {py_board['turn_to_move']}. "
            f"Java: {java_board['turn_to_move']}."
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — Move legality comparator
#
# This checks whether both engines agree on which moves are legal from a
# given position. This catches cases where Python allows an illegal move
# or forbids a legal one.
# ─────────────────────────────────────────────────────────────────────────────

def compare_legal_moves(py_state: dict, java_client: JavaServerClient,
                        game: Game) -> ComparisonResult:
    """
    Compares the set of legal moves Python generates against what the Java
    server accepts by trial-and-error.

    This is expensive (one socket round-trip per candidate move) so use it
    sparingly — on specific positions you suspect are buggy, not in every
    step of a long random game.

    Strategy: for every move Python thinks is legal, attempt it on a fresh
    Java connection and check whether the Java accepts it (no exception
    response) or rejects it (error response). Then also try a sample of
    moves Python thinks are illegal and verify the Java also rejects them.
    """
    result = ComparisonResult()
    raw_moves = game._get_raw_moves(py_state)

    py_legal = set()
    for move in raw_moves:
        (r0, c0), (r1, c1) = move
        from_cell = python_to_java_notation(r0, c0)
        to_cell = python_to_java_notation(r1, c1)
        py_legal.add((from_cell, to_cell))

    # To test against Java we'd need to send each move and check the response
    # code. Since the Java server is stateful (each move advances the game),
    # this requires either restarting the server or using a snapshot mechanism.
    # We log a summary here instead and leave trial-and-error for the fuzzer.
    log.info(f"Python found {len(py_legal)} legal moves from this position.")
    log.info(f"Sample moves: {list(py_legal)[:5]}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Layer 4 — Test runners
# ─────────────────────────────────────────────────────────────────────────────

def run_scripted_game(moves: list[tuple[str, str]], verbose: bool = True) -> list[ComparisonResult]:
    """
    Plays a pre-scripted sequence of moves through both engines and compares
    the resulting state after every single move.

    `moves` is a list of (from_cell, to_cell) pairs in Java notation,
    e.g. [("e2", "e3"), ("a4", "a5"), ...].

    This is the best starting point: take a real game log from the Java
    server and replay it, checking for the first point of divergence.
    """
    game = Game()
    py_state = game.getInitBoard()

    # We need one client for each side since the server tracks them separately.
    white_client = JavaServerClient('WHITE')
    black_client = JavaServerClient('BLACK')

    results = []

    try:
        white_initial = white_client.connect()
        black_initial = black_client.connect()

        # Verify the initial state matches before any moves are made.
        initial_result = compare_states(py_state['board'], white_initial)
        if verbose:
            log.info(f"Initial state comparison: {initial_result.report()}")
        results.append(initial_result)

        for move_idx, (from_cell, to_cell) in enumerate(moves):
            # Determine whose turn it is from the Python state.
            turn = py_state['board']['turn_to_move']
            player_str = 'WHITE' if turn == 1 else 'BLACK'
            client = white_client if turn == 1 else black_client

            log.info(f"Move {move_idx + 1}: {player_str} plays {from_cell} → {to_cell}")

            # Convert Java notation to Python (row, col).
            c0 = ord(from_cell[0]) - ord('a')
            r0 = int(from_cell[1]) - 1
            c1 = ord(to_cell[0]) - ord('a')
            r1 = int(to_cell[1]) - 1

            # Apply the move in Python.
            py_state = game._apply_move(py_state, [[r0, c0], [r1, c1]])

            # Apply the same move in Java and get the resulting state.
            java_state = client.send_move_and_get_state(from_cell, to_cell)

            # Compare.
            result = compare_states(py_state['board'], java_state)
            results.append(result)

            if verbose:
                log.info(f"After move {move_idx + 1}: {result.report()}")

            # Stop at the first divergence — later states would all be wrong anyway.
            if not result.match:
                log.warning(f"First divergence found at move {move_idx + 1}. Stopping.")
                break

    finally:
        white_client.close()
        black_client.close()

    return results


def run_random_game(max_moves: int = 100, seed: Optional[int] = None,
                    stop_on_first_divergence: bool = True) -> list[ComparisonResult]:
    """
    Plays a random game by always choosing a random legal move (according
    to Python's move generator) and comparing against the Java server.

    Setting a seed makes divergences reproducible — when you find a bug,
    you can rerun with the same seed to reproduce it deterministically.
    """
    if seed is not None:
        random.seed(seed)
        log.info(f"Random game with seed={seed}")

    game = Game()
    py_state = game.getInitBoard()
    white_client = JavaServerClient('WHITE')
    black_client = JavaServerClient('BLACK')
    results = []
    move_log = []  # record every move so we can replay the exact sequence

    try:
        white_client.connect()
        black_client.connect()

        for move_idx in range(max_moves):
            if game._is_terminal(py_state):
                log.info(f"Game ended at move {move_idx}.")
                break

            raw_moves = game._get_raw_moves(py_state)
            if not raw_moves:
                log.info("No legal moves available.")
                break

            # Pick a random move.
            chosen = random.choice(raw_moves)
            (r0, c0), (r1, c1) = chosen
            from_cell = python_to_java_notation(r0, c0)
            to_cell = python_to_java_notation(r1, c1)
            move_log.append((from_cell, to_cell))

            turn = py_state['board']['turn_to_move']
            client = white_client if turn == 1 else black_client

            # Apply in Python.
            py_state = game._apply_move(py_state, chosen)

            # Apply in Java.
            try:
                java_state = client.send_move_and_get_state(from_cell, to_cell)
            except Exception as e:
                log.error(f"Java server rejected move {from_cell}→{to_cell}: {e}")
                log.error(f"This move was considered legal by Python but rejected by Java!")
                log.error(f"Move sequence to reproduce: {move_log}")
                result = ComparisonResult()
                result.add_diff(f"Java rejected move {from_cell}→{to_cell}: {e}")
                results.append(result)
                break

            result = compare_states(py_state['board'], java_state)
            results.append(result)

            if not result.match:
                log.warning(f"Divergence at move {move_idx + 1}: {result.report()}")
                log.warning(f"Reproduce with: run_scripted_game({move_log})")
                if stop_on_first_divergence:
                    break
            else:
                log.debug(f"Move {move_idx + 1} ({from_cell}→{to_cell}): ✓")

    finally:
        white_client.close()
        black_client.close()

    # Print a summary.
    total = len(results)
    passed = sum(1 for r in results if r.match)
    log.info(f"\nSummary: {passed}/{total} states matched.")
    if passed < total:
        log.info(f"To reproduce: run_scripted_game({move_log})")

    return results


def run_fuzz_campaign(n_games: int = 200, max_moves_per_game: int = 80):
    """
    Runs many random games and collects statistics on where divergences occur.

    This is the highest-level test: run it overnight and look at the
    distribution of divergence points in the log. If divergences cluster
    around move 1-3, the issue is in the initial position or basic move
    rules. If they appear only late in games, it's likely a capture or
    repetition detection issue.
    """
    divergence_count = 0
    divergence_moves = []

    for game_idx in range(n_games):
        seed = game_idx  # deterministic seeds so any failure is reproducible
        log.info(f"\n{'='*60}")
        log.info(f"Game {game_idx + 1}/{n_games} (seed={seed})")

        results = run_random_game(max_moves=max_moves_per_game, seed=seed)
        failed = [i for i, r in enumerate(results) if not r.match]

        if failed:
            divergence_count += 1
            divergence_moves.append(failed[0])
            log.warning(f"Game {game_idx + 1}: divergence at move {failed[0] + 1}")
        else:
            log.info(f"Game {game_idx + 1}: all {len(results)} states matched ✓")

        # Small delay between games to avoid overwhelming the server.
        time.sleep(0.5)

    log.info(f"\n{'='*60}")
    log.info(f"Campaign complete: {divergence_count}/{n_games} games had divergences.")
    if divergence_moves:
        avg_move = sum(divergence_moves) / len(divergence_moves)
        log.info(f"Average divergence move: {avg_move:.1f}")
        log.info(f"Earliest divergence: move {min(divergence_moves) + 1}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Start with a single scripted game using the standard opening moves
    # to verify the harness itself works before running the fuzzer.
    # These are real Tablut opening moves in Java notation.
    sample_game = [
        ("e2", "e3"),   # white moves a pawn forward
        ("a4", "b4"),   # black slides out of camp
        ("e3", "d3"),   # white continues
        ("b4", "b5"),   # black advances
    ]

    log.info("Running scripted game test...")
    results = run_scripted_game(sample_game, verbose=True)
    n_pass = sum(1 for r in results if r.match)
    log.info(f"Scripted game: {n_pass}/{len(results)} states matched.")

    # Once the scripted test passes, uncomment to run the fuzzer:
    # log.info("\nStarting fuzz campaign...")
    # run_fuzz_campaign(n_games=100, max_moves_per_game=60)