import encoding # per la codifica dello stato del gioco

# STATO:
# Tensore 9*9*7, gli "strati" rappresentano le seguenti informazioni:
# - Strato 0: posizione dei pezzi bianchi (1 se c'è un pezzo bianco, 0 altrimenti)
# - Strato 1: posizione dei pezzi neri (1 se c'è un pezzo nero, 0 altrimenti)
# - Strato 2: posizione del re
# - Strato 3: posizione del trono
# - Strato 4: posizione dei camps
# - Strato 5: posizione delle vie di fuga
# - Strato 6: turno del giocatore (1 per bianco, 0 per nero)

class Game:
    # Stato contiene white, black, re, side_to_move
    def get_initial_state(self):
        return {
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
                'side_to_move': 1
            }
        

    def get_valid_moves(self, state):
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
        import copy
        new_state = {k: copy.deepcopy(v) for k, v in state.items()}

        r0, c0 = tuple(action[0])
        r1, c1 = tuple(action[1])

        # Move White
        if (r0, c0) in new_state.get('white_positions', []):
            new_state['white_positions'].remove((r0, c0))
            new_state['white_positions'].append((r1, c1))
            
        # Move Black
        if (r0, c0) in new_state.get('black_positions', []):
            new_state['black_positions'].remove((r0, c0))
            new_state['black_positions'].append((r1, c1))

        # Move King
        if new_state.get('king_position') == (r0, c0):
            new_state['king_position'] = (r1, c1)

        # Toggle turn
        if 'side_to_move' in new_state:
            new_state['side_to_move'] = 1 - new_state['side_to_move']
        if 'turn_to_move' in new_state:
            new_state['turn_to_move'] = 1 - new_state['turn_to_move']

        # TODO: Implement complete capturing logic (Ashton Tablut sandwich rules)
        return new_state

    def is_terminal(self, state) -> bool:
        king_pos = state["king_position"]

        # Re catturato
        if king_pos is None:
            return True

        # Re arrivato al bordo
        if self._is_escape_square(king_pos):
            return True

        # Nessuna mossa legale per il player di turno
        valid_moves = self.get_valid_moves(state)
        if len(valid_moves) == 0:
            return True

        # Ripetizione, solo se la tieni nello stato o altrove
        repetition_count = state.get("repetition_count", 0)
        if repetition_count >= 3:
            return True

        return False

    def _is_escape_square(self, pos) -> bool:
        return pos in encoding.ESCAPES
    

    def get_winner(self, state):
        king_pos = state.get("king_position")
        side_to_move = state.get("side_to_move", state.get("turn_to_move"))

        # 1) Re catturato -> vince Black
        if king_pos is None:
            return 0  # BLACK

        # 2) Re arrivato su una escape square -> vince White
        if self._is_escape_square(king_pos):
            return 1  # WHITE

        # 3) Nessuna mossa legale -> perde il giocatore di turno
        valid_moves = self.get_valid_moves(state)
        if len(valid_moves) == 0:
            if side_to_move == 1:
                return 0  # tocca al bianco e non può muovere -> vince Black
            else:
                return 1  # tocca al nero e non può muovere -> vince White

        # 4) Ripetizione
        repetition_count = state.get("repetition_count", 0)
        if repetition_count >= 3:
            repetition_loser = state.get("repetition_loser")
            if repetition_loser is not None:
                return 0 if repetition_loser == 1 else 1
            return -1  # draw / sconosciuto

        # Se non terminale, non dovrebbe essere chiamato
        raise ValueError("get_winner called on a non-terminal state")