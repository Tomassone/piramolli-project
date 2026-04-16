"""
Game engine inspired by the Ashton Rules of Tablut

Traduzione Python di GameAshtonTablut.java
Autori originali: A. Piretti, Andrea Galassi
"""

import logging
import os
from datetime import datetime
from copy import deepcopy
from typing import List

# Si assume che questi moduli esistano nel progetto Python equivalente
from state import State
from action import Action
from exceptions import (
    BoardException, ActionException, StopException, PawnException,
    DiagonalException, ClimbingException, ThroneException,
    OccupitedException, ClimbingCitadelException, CitadelException
)


class GameAshtonTablut:
    """
    Game engine ispirato alle regole Ashton di Tablut.
    """

    def __init__(self, repeated_moves_allowed: int, cache_size: int,
                 logs_folder: str, white_name: str, black_name: str,
                 state: State = None):
        """
        :param repeated_moves_allowed: numero di stati ripetuti prima del pareggio
        :param cache_size: numero di stati tenuti in memoria (-1 = infinito)
        :param logs_folder: cartella per i log di partita
        :param white_name: nome del giocatore bianco
        :param black_name: nome del giocatore nero
        :param state: stato iniziale (opzionale, usa StateTablut di default)
        """
        if state is None:
            from state_tablut import StateTablut
            state = StateTablut()

        self._state = state  # conservato per riferimento interno se necessario
        self.repeated_moves_allowed = repeated_moves_allowed
        self.cache_size = cache_size
        self.moves_without_capturing = 0

        # Costruzione del path del log
        log_filename = (
            f"_{white_name}_vs_{black_name}_{int(datetime.now().timestamp() * 1000)}_gameLog.txt"
        )
        log_path = os.path.abspath(os.path.join(logs_folder, log_filename))
        self.game_log_name = log_path

        # Creazione cartella e file di log
        os.makedirs(logs_folder, exist_ok=True)
        if not os.path.exists(log_path):
            open(log_path, 'w').close()
        self.game_log = log_path

        # Configurazione logger
        self.logg_game = logging.getLogger("GameLog")
        self.logg_game.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logg_game.addHandler(fh)

        self.logg_game.debug(f"Players:\t{white_name}\tvs\t{black_name}")
        self.logg_game.debug(f"Repeated moves allowed:\t{repeated_moves_allowed}\tCache:\t{cache_size}")
        self.logg_game.debug("Inizio partita")
        self.logg_game.debug(f"Stato:\n{state}")

        # Condizioni di pareggio (lista di stati)
        self.draw_conditions: List[State] = []

        # Citadels (caselle fortezza)
        self.citadels: List[str] = [
            "a4", "a5", "a6", "b5",
            "d1", "e1", "f1", "e2",
            "i4", "i5", "i6", "h5",
            "d9", "e9", "f9", "e8",
        ]

    # ------------------------------------------------------------------
    # Metodo principale: verifica e applica una mossa
    # ------------------------------------------------------------------

    def check_move(self, state: State, a: Action) -> State:
        """
        Controlla la validità della mossa e aggiorna lo stato.
        Lancia eccezioni se la mossa non è valida.
        """
        self.logg_game.debug(str(a))

        # Controllo formato mossa
        if len(a.get_to()) != 2 or len(a.get_from()) != 2:
            self.logg_game.warning("Formato mossa errato")
            raise ActionException(a)

        column_from = a.get_column_from()
        column_to   = a.get_column_to()
        row_from    = a.get_row_from()
        row_to      = a.get_row_to()

        board_size = len(state.get_board())

        # Controllo se sono fuori dal tabellone
        if (column_from > board_size - 1 or row_from > board_size - 1
                or row_to > board_size - 1 or column_to > board_size - 1
                or column_from < 0 or row_from < 0 or row_to < 0 or column_to < 0):
            self.logg_game.warning("Mossa fuori tabellone")
            raise BoardException(a)

        # Controllo che non vada sul trono
        if state.get_pawn(row_to, column_to).equals_pawn(State.Pawn.THRONE.value):
            self.logg_game.warning("Mossa sul trono")
            raise ThroneException(a)

        # Controllo la casella di arrivo (deve essere vuota)
        if not state.get_pawn(row_to, column_to).equals_pawn(State.Pawn.EMPTY.value):
            self.logg_game.warning("Mossa sopra una casella occupata")
            raise OccupitedException(a)

        # Controllo citadel di arrivo: non ci si può andare se non si parte da una citadel
        if (self.citadels.__contains__(state.get_box(row_to, column_to))
                and not self.citadels.__contains__(state.get_box(row_from, column_from))):
            self.logg_game.warning("Mossa che arriva sopra una citadel")
            raise CitadelException(a)

        # Controllo distanza massima tra citadels
        if (self.citadels.__contains__(state.get_box(row_to, column_to))
                and self.citadels.__contains__(state.get_box(row_from, column_from))):
            if row_from == row_to:
                if column_from - column_to > 5 or column_from - column_to < -5:
                    self.logg_game.warning("Mossa che arriva sopra una citadel")
                    raise CitadelException(a)
            else:
                if row_from - row_to > 5 or row_from - row_to < -5:
                    self.logg_game.warning("Mossa che arriva sopra una citadel")
                    raise CitadelException(a)

        # Controllo se si cerca di stare fermi
        if row_from == row_to and column_from == column_to:
            self.logg_game.warning("Nessuna mossa")
            raise StopException(a)

        # Controllo pedina giusta per il turno
        if state.get_turn().equals_turn(State.Turn.WHITE.value):
            if (not state.get_pawn(row_from, column_from).equals_pawn("W")
                    and not state.get_pawn(row_from, column_from).equals_pawn("K")):
                self.logg_game.warning(
                    f"Giocatore {a.get_turn()} cerca di muovere una pedina avversaria"
                )
                raise PawnException(a)

        if state.get_turn().equals_turn(State.Turn.BLACK.value):
            if not state.get_pawn(row_from, column_from).equals_pawn("B"):
                self.logg_game.warning(
                    f"Giocatore {a.get_turn()} cerca di muovere una pedina avversaria"
                )
                raise PawnException(a)

        # Controllo di non muovere in diagonale
        if row_from != row_to and column_from != column_to:
            self.logg_game.warning("Mossa in diagonale")
            raise DiagonalException(a)

        # Controllo di non scavalcare pedine (movimento orizzontale)
        if row_from == row_to:
            if column_from > column_to:
                for i in range(column_to, column_from):
                    if not state.get_pawn(row_from, i).equals_pawn(State.Pawn.EMPTY.value):
                        if state.get_pawn(row_from, i).equals_pawn(State.Pawn.THRONE.value):
                            self.logg_game.warning("Mossa che scavalca il trono")
                        else:
                            self.logg_game.warning("Mossa che scavalca una pedina")
                        raise ClimbingException(a)
                    if (self.citadels.__contains__(state.get_box(row_from, i))
                            and not self.citadels.__contains__(
                                state.get_box(a.get_row_from(), a.get_column_from()))):
                        self.logg_game.warning("Mossa che scavalca una citadel")
                        raise ClimbingCitadelException(a)
            else:
                for i in range(column_from + 1, column_to + 1):
                    if not state.get_pawn(row_from, i).equals_pawn(State.Pawn.EMPTY.value):
                        if state.get_pawn(row_from, i).equals_pawn(State.Pawn.THRONE.value):
                            self.logg_game.warning("Mossa che scavalca il trono")
                        else:
                            self.logg_game.warning("Mossa che scavalca una pedina")
                        raise ClimbingException(a)
                    if (self.citadels.__contains__(state.get_box(row_from, i))
                            and not self.citadels.__contains__(
                                state.get_box(a.get_row_from(), a.get_column_from()))):
                        self.logg_game.warning("Mossa che scavalca una citadel")
                        raise ClimbingCitadelException(a)
        # Movimento verticale
        else:
            if row_from > row_to:
                for i in range(row_to, row_from):
                    if not state.get_pawn(i, column_from).equals_pawn(State.Pawn.EMPTY.value):
                        if state.get_pawn(i, column_from).equals_pawn(State.Pawn.THRONE.value):
                            self.logg_game.warning("Mossa che scavalca il trono")
                        else:
                            self.logg_game.warning("Mossa che scavalca una pedina")
                        raise ClimbingException(a)
                    if (self.citadels.__contains__(state.get_box(i, column_from))
                            and not self.citadels.__contains__(
                                state.get_box(a.get_row_from(), a.get_column_from()))):
                        self.logg_game.warning("Mossa che scavalca una citadel")
                        raise ClimbingCitadelException(a)
            else:
                for i in range(row_from + 1, row_to + 1):
                    if not state.get_pawn(i, column_from).equals_pawn(State.Pawn.EMPTY.value):
                        if state.get_pawn(i, column_from).equals_pawn(State.Pawn.THRONE.value):
                            self.logg_game.warning("Mossa che scavalca il trono")
                        else:
                            self.logg_game.warning("Mossa che scavalca una pedina")
                        raise ClimbingException(a)
                    if (self.citadels.__contains__(state.get_box(i, column_from))
                            and not self.citadels.__contains__(
                                state.get_box(a.get_row_from(), a.get_column_from()))):
                        self.logg_game.warning("Mossa che scavalca una citadel")
                        raise ClimbingCitadelException(a)

        # Se sono arrivato qui, muovo la pedina
        state = self._move_pawn(state, a)

        # Controllo catture dopo il movimento
        if state.get_turn().equals_turn("W"):
            state = self._check_capture_black(state, a)
        elif state.get_turn().equals_turn("B"):
            state = self._check_capture_white(state, a)

        # Se qualcosa è stato catturato, pulisco la cache per i pareggi
        if self.moves_without_capturing == 0:
            self.draw_conditions.clear()
            self.logg_game.debug("Capture! Draw cache cleared!")

        # Controllo pareggio per stati ripetuti
        trovati = 0
        for s in self.draw_conditions:
            print(str(s))
            if s == state:
                trovati += 1
                if trovati > self.repeated_moves_allowed:
                    state.set_turn(State.Turn.DRAW)
                    self.logg_game.debug(
                        "Partita terminata in pareggio per numero di stati ripetuti"
                    )
                    break

        if trovati > 0:
            self.logg_game.debug(f"Equal states found: {trovati}")

        # Gestione dimensione cache
        if self.cache_size >= 0 and len(self.draw_conditions) > self.cache_size:
            self.draw_conditions.pop(0)
        self.draw_conditions.append(deepcopy(state))

        self.logg_game.debug(f"Current draw cache size: {len(self.draw_conditions)}")
        self.logg_game.debug(f"Stato:\n{state}")
        print(f"Stato:\n{state}")

        return state

    # ------------------------------------------------------------------
    # Catture per il bianco (dopo mossa nera)
    # ------------------------------------------------------------------

    def _check_capture_white(self, state: State, a: Action) -> State:
        board_size = len(state.get_board())
        rt = a.get_row_to()
        ct = a.get_column_to()

        # Mangio a destra
        if (ct < board_size - 2
                and state.get_pawn(rt, ct + 1).equals_pawn("B")
                and (state.get_pawn(rt, ct + 2).equals_pawn("W")
                     or state.get_pawn(rt, ct + 2).equals_pawn("T")
                     or state.get_pawn(rt, ct + 2).equals_pawn("K")
                     or (self.citadels.__contains__(state.get_box(rt, ct + 2))
                         and not (ct + 2 == 8 and rt == 4)
                         and not (ct + 2 == 4 and rt == 0)
                         and not (ct + 2 == 4 and rt == 8)
                         and not (ct + 2 == 0 and rt == 4)))):
            state.remove_pawn(rt, ct + 1)
            self.moves_without_capturing = -1
            self.logg_game.debug(f"Pedina nera rimossa in: {state.get_box(rt, ct + 1)}")

        # Mangio a sinistra
        if (ct > 1
                and state.get_pawn(rt, ct - 1).equals_pawn("B")
                and (state.get_pawn(rt, ct - 2).equals_pawn("W")
                     or state.get_pawn(rt, ct - 2).equals_pawn("T")
                     or state.get_pawn(rt, ct - 2).equals_pawn("K")
                     or (self.citadels.__contains__(state.get_box(rt, ct - 2))
                         and not (ct - 2 == 8 and rt == 4)
                         and not (ct - 2 == 4 and rt == 0)
                         and not (ct - 2 == 4 and rt == 8)
                         and not (ct - 2 == 0 and rt == 4)))):
            state.remove_pawn(rt, ct - 1)
            self.moves_without_capturing = -1
            self.logg_game.debug(f"Pedina nera rimossa in: {state.get_box(rt, ct - 1)}")

        # Mangio sopra
        if (rt > 1
                and state.get_pawn(rt - 1, ct).equals_pawn("B")
                and (state.get_pawn(rt - 2, ct).equals_pawn("W")
                     or state.get_pawn(rt - 2, ct).equals_pawn("T")
                     or state.get_pawn(rt - 2, ct).equals_pawn("K")
                     or (self.citadels.__contains__(state.get_box(rt - 2, ct))
                         and not (ct == 8 and rt - 2 == 4)
                         and not (ct == 4 and rt - 2 == 0)
                         and not (ct == 4 and rt - 2 == 8)
                         and not (ct == 0 and rt - 2 == 4)))):
            state.remove_pawn(rt - 1, ct)
            self.moves_without_capturing = -1
            self.logg_game.debug(f"Pedina nera rimossa in: {state.get_box(rt - 1, ct)}")

        # Mangio sotto
        if (rt < board_size - 2
                and state.get_pawn(rt + 1, ct).equals_pawn("B")
                and (state.get_pawn(rt + 2, ct).equals_pawn("W")
                     or state.get_pawn(rt + 2, ct).equals_pawn("T")
                     or state.get_pawn(rt + 2, ct).equals_pawn("K")
                     or (self.citadels.__contains__(state.get_box(rt + 2, ct))
                         and not (ct == 8 and rt + 2 == 4)
                         and not (ct == 4 and rt + 2 == 0)
                         and not (ct == 4 and rt + 2 == 8)
                         and not (ct == 0 and rt + 2 == 4)))):
            state.remove_pawn(rt + 1, ct)
            self.moves_without_capturing = -1
            self.logg_game.debug(f"Pedina nera rimossa in: {state.get_box(rt + 1, ct)}")

        # Controllo vittoria bianco (re raggiunge il bordo)
        if (rt == 0 or rt == board_size - 1 or ct == 0 or ct == board_size - 1):
            if state.get_pawn(rt, ct).equals_pawn("K"):
                state.set_turn(State.Turn.WHITEWIN)
                self.logg_game.debug(f"Bianco vince con re in {a.get_to()}")

        # TODO: implementare la condizione di vittoria per cattura dell'ultima pedina nera

        self.moves_without_capturing += 1
        return state

    # ------------------------------------------------------------------
    # Catture del re da parte del nero
    # ------------------------------------------------------------------

    def _check_capture_black_king_left(self, state: State, a: Action) -> State:
        rt = a.get_row_to()
        ct = a.get_column_to()

        # Re sulla sinistra
        if ct > 1 and state.get_pawn(rt, ct - 1).equals_pawn("K"):
            king_box = state.get_box(rt, ct - 1)

            # Re sul trono
            if king_box == "e5":
                if (state.get_pawn(3, 4).equals_pawn("B")
                        and state.get_pawn(4, 3).equals_pawn("B")
                        and state.get_pawn(5, 4).equals_pawn("B")):
                    state.set_turn(State.Turn.BLACKWIN)
                    self.logg_game.debug(f"Nero vince con re catturato in: {king_box}")

            # Re adiacente al trono
            elif king_box == "e4":
                if (state.get_pawn(2, 4).equals_pawn("B")
                        and state.get_pawn(3, 3).equals_pawn("B")):
                    state.set_turn(State.Turn.BLACKWIN)
                    self.logg_game.debug(f"Nero vince con re catturato in: {king_box}")

            elif king_box == "f5":
                if (state.get_pawn(5, 5).equals_pawn("B")
                        and state.get_pawn(3, 5).equals_pawn("B")):
                    state.set_turn(State.Turn.BLACKWIN)
                    self.logg_game.debug(f"Nero vince con re catturato in: {king_box}")

            elif king_box == "e6":
                if (state.get_pawn(6, 4).equals_pawn("B")
                        and state.get_pawn(5, 3).equals_pawn("B")):
                    state.set_turn(State.Turn.BLACKWIN)
                    self.logg_game.debug(f"Nero vince con re catturato in: {king_box}")

            # Fuori dalla zona del trono
            elif king_box not in ("e5", "e6", "e4", "f5"):
                if (state.get_pawn(rt, ct - 2).equals_pawn("B")
                        or self.citadels.__contains__(state.get_box(rt, ct - 2))):
                    state.set_turn(State.Turn.BLACKWIN)
                    self.logg_game.debug(f"Nero vince con re catturato in: {king_box}")

        return state

    def _check_capture_black_king_right(self, state: State, a: Action) -> State:
        rt = a.get_row_to()
        ct = a.get_column_to()
        board_size = len(state.get_board())

        # Re sulla destra
        if (ct < board_size - 2
                and state.get_pawn(rt, ct + 1).equals_pawn("K")):
            king_box = state.get_box(rt, ct + 1)

            # Re sul trono
            if king_box == "e5":
                if (state.get_pawn(3, 4).equals_pawn("B")
                        and state.get_pawn(4, 5).equals_pawn("B")
                        and state.get_pawn(5, 4).equals_pawn("B")):
                    state.set_turn(State.Turn.BLACKWIN)
                    self.logg_game.debug(f"Nero vince con re catturato in: {king_box}")

            # Re adiacente al trono
            elif king_box == "e4":
                if (state.get_pawn(2, 4).equals_pawn("B")
                        and state.get_pawn(3, 5).equals_pawn("B")):
                    state.set_turn(State.Turn.BLACKWIN)
                    self.logg_game.debug(f"Nero vince con re catturato in: {king_box}")

            elif king_box == "e6":
                if (state.get_pawn(5, 5).equals_pawn("B")
                        and state.get_pawn(6, 4).equals_pawn("B")):
                    state.set_turn(State.Turn.BLACKWIN)
                    self.logg_game.debug(f"Nero vince con re catturato in: {king_box}")

            elif king_box == "d5":
                if (state.get_pawn(3, 3).equals_pawn("B")
                        and state.get_pawn(5, 3).equals_pawn("B")):
                    state.set_turn(State.Turn.BLACKWIN)
                    self.logg_game.debug(f"Nero vince con re catturato in: {king_box}")

            # Fuori dalla zona del trono
            elif king_box not in ("d5", "e6", "e4", "e5"):
                if (state.get_pawn(rt, ct + 2).equals_pawn("B")
                        or self.citadels.__contains__(state.get_box(rt, ct + 2))):
                    state.set_turn(State.Turn.BLACKWIN)
                    self.logg_game.debug(f"Nero vince con re catturato in: {king_box}")

        return state

    def _check_capture_black_king_down(self, state: State, a: Action) -> State:
        rt = a.get_row_to()
        ct = a.get_column_to()
        board_size = len(state.get_board())

        # Re sotto
        if (rt < board_size - 2
                and state.get_pawn(rt + 1, ct).equals_pawn("K")):
            king_box = state.get_box(rt + 1, ct)

            # Re sul trono
            if king_box == "e5":
                if (state.get_pawn(5, 4).equals_pawn("B")
                        and state.get_pawn(4, 5).equals_pawn("B")
                        and state.get_pawn(4, 3).equals_pawn("B")):
                    state.set_turn(State.Turn.BLACKWIN)
                    self.logg_game.debug(f"Nero vince con re catturato in: {king_box}")

            # Re adiacente al trono
            elif king_box == "e4":
                if (state.get_pawn(3, 3).equals_pawn("B")
                        and state.get_pawn(3, 5).equals_pawn("B")):
                    state.set_turn(State.Turn.BLACKWIN)
                    self.logg_game.debug(f"Nero vince con re catturato in: {king_box}")

            elif king_box == "d5":
                if (state.get_pawn(4, 2).equals_pawn("B")
                        and state.get_pawn(5, 3).equals_pawn("B")):
                    state.set_turn(State.Turn.BLACKWIN)
                    self.logg_game.debug(f"Nero vince con re catturato in: {king_box}")

            elif king_box == "f5":
                if (state.get_pawn(4, 6).equals_pawn("B")
                        and state.get_pawn(5, 5).equals_pawn("B")):
                    state.set_turn(State.Turn.BLACKWIN)
                    self.logg_game.debug(f"Nero vince con re catturato in: {king_box}")

            # Fuori dalla zona del trono
            elif king_box not in ("d5", "e4", "f5", "e5"):
                if (state.get_pawn(rt + 2, ct).equals_pawn("B")
                        or self.citadels.__contains__(state.get_box(rt + 2, ct))):
                    state.set_turn(State.Turn.BLACKWIN)
                    self.logg_game.debug(f"Nero vince con re catturato in: {king_box}")

        return state

    def _check_capture_black_king_up(self, state: State, a: Action) -> State:
        rt = a.get_row_to()
        ct = a.get_column_to()

        # Re sopra
        if rt > 1 and state.get_pawn(rt - 1, ct).equals_pawn("K"):
            king_box = state.get_box(rt - 1, ct)

            # Re sul trono
            if king_box == "e5":
                if (state.get_pawn(3, 4).equals_pawn("B")
                        and state.get_pawn(4, 5).equals_pawn("B")
                        and state.get_pawn(4, 3).equals_pawn("B")):
                    state.set_turn(State.Turn.BLACKWIN)
                    self.logg_game.debug(f"Nero vince con re catturato in: {king_box}")

            # Re adiacente al trono
            elif king_box == "e6":
                if (state.get_pawn(5, 3).equals_pawn("B")
                        and state.get_pawn(5, 5).equals_pawn("B")):
                    state.set_turn(State.Turn.BLACKWIN)
                    self.logg_game.debug(f"Nero vince con re catturato in: {king_box}")

            elif king_box == "d5":
                if (state.get_pawn(4, 2).equals_pawn("B")
                        and state.get_pawn(3, 3).equals_pawn("B")):
                    state.set_turn(State.Turn.BLACKWIN)
                    self.logg_game.debug(f"Nero vince con re catturato in: {king_box}")

            elif king_box == "f5":
                if (state.get_pawn(4, 6).equals_pawn("B")
                        and state.get_pawn(3, 5).equals_pawn("B")):
                    state.set_turn(State.Turn.BLACKWIN)
                    self.logg_game.debug(f"Nero vince con re catturato in: {king_box}")

            # Fuori dalla zona del trono
            elif king_box not in ("d5", "e6", "f5", "e5"):
                if (state.get_pawn(rt - 2, ct).equals_pawn("B")
                        or self.citadels.__contains__(state.get_box(rt - 2, ct))):
                    state.set_turn(State.Turn.BLACKWIN)
                    self.logg_game.debug(f"Nero vince con re catturato in: {king_box}")

        return state

    # ------------------------------------------------------------------
    # Catture pedine bianche da parte del nero
    # ------------------------------------------------------------------

    def _check_capture_black_pawn_right(self, state: State, a: Action) -> State:
        rt = a.get_row_to()
        ct = a.get_column_to()
        board_size = len(state.get_board())

        # Mangio a destra
        if (ct < board_size - 2
                and state.get_pawn(rt, ct + 1).equals_pawn("W")):
            if state.get_pawn(rt, ct + 2).equals_pawn("B"):
                state.remove_pawn(rt, ct + 1)
                self.moves_without_capturing = -1
                self.logg_game.debug(f"Pedina bianca rimossa in: {state.get_box(rt, ct + 1)}")
            elif state.get_pawn(rt, ct + 2).equals_pawn("T"):
                state.remove_pawn(rt, ct + 1)
                self.moves_without_capturing = -1
                self.logg_game.debug(f"Pedina bianca rimossa in: {state.get_box(rt, ct + 1)}")
            elif self.citadels.__contains__(state.get_box(rt, ct + 2)):
                state.remove_pawn(rt, ct + 1)
                self.moves_without_capturing = -1
                self.logg_game.debug(f"Pedina bianca rimossa in: {state.get_box(rt, ct + 1)}")
            elif state.get_box(rt, ct + 2) == "e5":
                state.remove_pawn(rt, ct + 1)
                self.moves_without_capturing = -1
                self.logg_game.debug(f"Pedina bianca rimossa in: {state.get_box(rt, ct + 1)}")

        return state

    def _check_capture_black_pawn_left(self, state: State, a: Action) -> State:
        rt = a.get_row_to()
        ct = a.get_column_to()

        # Mangio a sinistra
        if (ct > 1
                and state.get_pawn(rt, ct - 1).equals_pawn("W")
                and (state.get_pawn(rt, ct - 2).equals_pawn("B")
                     or state.get_pawn(rt, ct - 2).equals_pawn("T")
                     or self.citadels.__contains__(state.get_box(rt, ct - 2))
                     or state.get_box(rt, ct - 2) == "e5")):
            state.remove_pawn(rt, ct - 1)
            self.moves_without_capturing = -1
            self.logg_game.debug(f"Pedina bianca rimossa in: {state.get_box(rt, ct - 1)}")

        return state

    def _check_capture_black_pawn_up(self, state: State, a: Action) -> State:
        rt = a.get_row_to()
        ct = a.get_column_to()

        # Controllo se mangio sopra
        if (rt > 1
                and state.get_pawn(rt - 1, ct).equals_pawn("W")
                and (state.get_pawn(rt - 2, ct).equals_pawn("B")
                     or state.get_pawn(rt - 2, ct).equals_pawn("T")
                     or self.citadels.__contains__(state.get_box(rt - 2, ct))
                     or state.get_box(rt - 2, ct) == "e5")):
            state.remove_pawn(rt - 1, ct)
            self.moves_without_capturing = -1
            self.logg_game.debug(f"Pedina bianca rimossa in: {state.get_box(rt - 1, ct)}")

        return state

    def _check_capture_black_pawn_down(self, state: State, a: Action) -> State:
        rt = a.get_row_to()
        ct = a.get_column_to()
        board_size = len(state.get_board())

        # Controllo se mangio sotto
        if (rt < board_size - 2
                and state.get_pawn(rt + 1, ct).equals_pawn("W")
                and (state.get_pawn(rt + 2, ct).equals_pawn("B")
                     or state.get_pawn(rt + 2, ct).equals_pawn("T")
                     or self.citadels.__contains__(state.get_box(rt + 2, ct))
                     or state.get_box(rt + 2, ct) == "e5")):
            state.remove_pawn(rt + 1, ct)
            self.moves_without_capturing = -1
            self.logg_game.debug(f"Pedina bianca rimossa in: {state.get_box(rt + 1, ct)}")

        return state

    def _check_capture_black(self, state: State, a: Action) -> State:
        """Controlla tutte le possibili catture dopo una mossa nera."""
        self._check_capture_black_pawn_right(state, a)
        self._check_capture_black_pawn_left(state, a)
        self._check_capture_black_pawn_up(state, a)
        self._check_capture_black_pawn_down(state, a)
        self._check_capture_black_king_right(state, a)
        self._check_capture_black_king_left(state, a)
        self._check_capture_black_king_down(state, a)
        self._check_capture_black_king_up(state, a)

        self.moves_without_capturing += 1
        return state

    # ------------------------------------------------------------------
    # Movimento pedina
    # ------------------------------------------------------------------

    def _move_pawn(self, state: State, a: Action) -> State:
        pawn = state.get_pawn(a.get_row_from(), a.get_column_from())
        new_board = state.get_board()
        self.logg_game.debug("Movimento pedina")

        # Libero il trono o una casella qualunque
        if a.get_column_from() == 4 and a.get_row_from() == 4:
            new_board[a.get_row_from()][a.get_column_from()] = State.Pawn.THRONE
        else:
            new_board[a.get_row_from()][a.get_column_from()] = State.Pawn.EMPTY

        # Metto nel nuovo tabellone la pedina mossa
        new_board[a.get_row_to()][a.get_column_to()] = pawn

        # Aggiorno il tabellone
        state.set_board(new_board)

        # Cambio il turno
        if state.get_turn().equals_turn(State.Turn.WHITE.value):
            state.set_turn(State.Turn.BLACK)
        else:
            state.set_turn(State.Turn.WHITE)

        return state

    # ------------------------------------------------------------------
    # Getter / setter / utilità
    # ------------------------------------------------------------------

    def get_game_log(self) -> str:
        return self.game_log

    def get_moves_without_capturing(self) -> int:
        return self.moves_without_capturing

    def _set_moves_without_capturing(self, moves_without_capturing: int):
        self.moves_without_capturing = moves_without_capturing

    def get_repeated_moves_allowed(self) -> int:
        return self.repeated_moves_allowed

    def get_cache_size(self) -> int:
        return self.cache_size

    def get_draw_conditions(self) -> List[State]:
        return self.draw_conditions

    def clear_draw_conditions(self):
        self.draw_conditions.clear()

    def end_game(self, state: State):
        self.logg_game.debug(f"Stato:\n{state}")
