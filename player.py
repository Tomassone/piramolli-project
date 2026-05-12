import time
import json
import socket
import struct
import onnxruntime as ort
import sys
import argparse
from tablut.TablutGame import TablutGame 
import copy
import numpy as np
from OnnxNetWrapper import ONNXNetWrapper  as onnet
import MCTS
from utils import dotdict

# ==============================================================================
# QUI IMPORTI I CODICI DEI TUOI COLLEGHI (QUANDO SARANNO PRONTI)
# Esempio:
# from tablut_ambiente import TablutBoard  # Dal Membro 1
# from mcts_player import MCTS             # Dal Membro 2
# ==============================================================================

# scacchiera json
#{
#    "board": [
#        ["O", "O", "O", "B", "B", "B", "O", "O", "O"],
#        ["O", "O", "O", "O", "B", "O", "O", "O", "O"],
#        ["O", "O", "O", "O", "W", "O", "O", "O", "O"],
#        ["B", "O", "O", "O", "W", "O", "O", "O", "B"],
#        ["B", "B", "W", "W", "K", "W", "W", "B", "B"],
#        ["B", "O", "O", "O", "W", "O", "O", "O", "B"],
#        ["O", "O", "O", "O", "W", "O", "O", "O", "O"],
#        ["O", "O", "O", "O", "B", "O", "O", "O", "O"],
#        ["O", "O", "O", "B", "B", "B", "O", "O", "O"]
#    ],
#    "turn": "W"
#}
 
def json_to_board_dict(scacchiera_json):
    """
    Prende il JSON del server (matrice 9x9 di stringhe) e restituisce 
    SOLO il dizionario dei pezzi che il tuo TablutGame si aspetta.
    """
    matrix = scacchiera_json['board']
    turn_str = scacchiera_json['turn']
    
    white_pos = []
    black_pos = []
    king_pos = None
    
    for r in range(9):
        for c in range(9):
            val = matrix[r][c]
            if val == 'W':
                white_pos.append((r, c))
            elif val == 'B':
                black_pos.append((r, c))
            elif val == 'K':
                king_pos = (r, c)
                
    # Nel TablutGame del tuo collega: 1 = BIANCO, 0 = NERO
    turn_to_move = 1 if turn_str == "W" else 0
    
    return {
        'white_positions': white_pos,
        'black_positions': black_pos,
        'king_position': king_pos,
        'turn_to_move': turn_to_move
    }



def pensa_e_muovi(game, scacchiera, motore_onnx,tempo_inizio, tempo_sicuro_disponibile):
    
    # -------------------------------------------------------------
    # IL MEMBRO 2 INIZIALIZZA L'ALBERO MCTS
    # Esempio:
    # mcts = MCTS(motore=motore_onnx)
    # -------------------------------------------------------------

    mcts = MCTS.MCTS(game, motore_onnx, dotdict({'numMCTSSims': 999999, 'cpuct': 1.0}))
    # Poi estrai la mossa migliore manualmente
   
        
    iterazioni = 0
    
    while (time.time() - tempo_inizio) < tempo_sicuro_disponibile:
        
        # ---------------------------------------------------------
        # QUI L'MCTS ESPLORA I RAMI (Questa è la riga pesante)
        # Esempio:
        mcts.search(scacchiera)
        # ---------------------------------------------------------
        
        iterazioni += 1
        
        # -- RIGA FINTO-MCTS (DA CANCELLARE QUANDO HAI IL CODICE VERO) --
        time.sleep(0.001)  # Finge di pensare per 1 millisecondo
        # ---------------------------------------------------------------
     
    print(f"[*] Fine pensiero. Valutati {iterazioni} percorsi.")

    # -------------------------------------------------------------
    # CHIEDI AL MCTS QUAL È LA MOSSA MIGLIORE E PREPARA LA STRINGA
    # Esempio:
    s = game.stringRepresentation(scacchiera)
    counts = [mcts.Nsa.get((s, a), 0) for a in range(game.getActionSize())]
    best_action = int(np.argmax(counts))
    # -------------------------------------------------------------
    
    # --- DECODIFICA DELL'AZIONE IN COORDINATE (riga, colonna) ---
    # Come definito in TablutGame.py: (r0 * 9 + c0) * 81 + (r1 * 9 + c1)
    from_idx, to_idx = divmod(best_action, 9 ** 2)
    r0, c0 = divmod(from_idx, 9)
    r1, c1 = divmod(to_idx, 9)
    
    # --- TRADUZIONE IN FORMATO JAVA PER IL SERVER ---
    # Il server Java usa coordinate tipo scacchi: 'a'-'i' per le colonne, '1'-'9' per le righe.
    # colonna = lettera (a = 0, b = 1, ... i = 8)
    # riga = numero stringa ('1' = 0, '2' = 1, ... '9' = 8) - NOTA: nel tuo codice la 
    # matrice JSON [0][0] corrisponde in alto a sinistra (a9), bisogna fare attenzione all'orientamento.
    # Assumendo l'orientamento classico: a=0, 1=0.
    
    lettere = "abcdefghi"
    casella_partenza = f"{lettere[c0]}{r0 + 1}"
    casella_arrivo   = f"{lettere[c1]}{r1 + 1}"
    
    # Capiamo se stiamo giocando col Bianco o col Nero per metterlo nel JSON
    ruolo_giocatore = "W" if scacchiera['board']['turn_to_move'] == 1 else "B"

    # Formazione mossa per server
    dizionario_mossa = {
        "from": casella_partenza,
        "to": casella_arrivo,
        "turn": ruolo_giocatore
    }

    mossa = json.dumps(dizionario_mossa)
    return mossa

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


def ricevi_scacchiera(sock):
    raw_len = recvall(sock, 4)
    if raw_len is None:
        return None # Il server ha chiuso la connessione
    
    len_bytes = struct.unpack('>i', raw_len)[0]
    
    json_bytes = recvall(sock, len_bytes)
    if json_bytes is None:
        return None
        
    current_state = json.loads(json_bytes.decode('utf-8'))
    return current_state

def invia_scacchiera(sock, mossa):
    payload_bytes = mossa.encode('utf-8')
    sock.send(struct.pack('>i', len(payload_bytes)))
    sock.send(payload_bytes)

def connettiti_all_arbitro(ip_arbitro, porta_arbitro, ruolo):
    """
    Simula la connessione al server Arbitro via Socket.
    """
    
    # Questo print usa i parametri che abbiamo estratto tramite argparse
    print(f"[*] In attesa di connettersi all'Arbitro su {ip_arbitro}:{porta_arbitro} come {ruolo}...")
    
    # --- LOGICA DI CONNESSIONE SOCKET VERA  ---
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((ip_arbitro, porta_arbitro))
    player_name= "Piramollo"
    s.send(struct.pack('>i', len(player_name)))
    s.send(player_name.encode())
    # -----------------------------------------------------------------
    print(f"Connesso all'arbitro!")

    return s

def gioca_partita(s, ruolo, timeout):
    game = TablutGame()

    motore = onnet("modello.onnx", game)


    scacchiera_iniziale = game.getInitBoard()['board']
    
    history_8 =  [copy.deepcopy(scacchiera_iniziale) for _ in range(8)]     # Le ultime 8 scacchiere
    draw_history = []    # Per i pareggi
    pezzi_totali = 25    # Per capire se qualcuno è morto (Tablut: 16 neri + 9 bianchi)
    move_count=0
    while True:
        # L'arbitro ci manda il JSON della scacchiera

        scacchiera_ricevuta = ricevi_scacchiera(s)
        turn_del_server = scacchiera_ricevuta.get('turn')
        
        # Controllo condizioni di fine partita
        if turn_del_server == "WW":
            print("LA PARTITA È FINITA: HA VINTO IL BIANCO!")
            break
        elif turn_del_server == "BW":
            print("LA PARTITA È FINITA: HA VINTO IL NERO!")
            break
        elif turn_del_server == "D":
            print("LA PARTITA È FINITA: PAREGGIO!")
            break

        board_dict=json_to_board_dict(scacchiera_ricevuta)
        history_8.pop(0)
        history_8.append(board_dict)
        pezzi_attuali = len(board_dict['white_positions']) + len(board_dict['black_positions'])
        if pezzi_attuali < pezzi_totali:
            draw_history = []  # Qualcuno è morto, resetto la lista delle configurazioni già viste (le altre non sono più raggiungibili)
            pezzi_totali = pezzi_attuali

        hash_corrente=game._board_hash(board_dict)
        draw_history.append(hash_corrente)

        if (ruolo == "WHITE" and turn_del_server == "W") or \
           (ruolo == "BLACK" and turn_del_server == "B"):
            tempo_inizio = time.time()
            
            print(f"[*] È il mio turno ({ruolo}).")
            #  TablutGame si aspetta questo formato di stato
            state_root = {
                'board': board_dict,
                'history': list(history_8),
                'move_count': move_count, 
                'half_move_clock':  len(draw_history) - 1, #mosse dall'ultima cattura, funziona considerato che la resetto ogni volta che ne vedo una
                'draw_history': list(draw_history),
                'repetition_count': draw_history.count(hash_corrente) - 1 
            }
            
            mossa_decisa = pensa_e_muovi(game, state_root, motore,tempo_inizio, timeout)
            
            print(f"[*] Invio mossa all'Arbitro: {mossa_decisa}")
            invia_scacchiera(s, mossa_decisa) 
            
        else:
            # È il turno dell'avversario. Non devi fare nulla, solo aspettare che lui muova 
            # e che il server ti mandi la prossima scacchiera aggiornata.
            print(f"[*] Turno dell'avversario. In attesa...")
        
        move_count=move_count+1
            
    # Fine del ciclo while (partita terminata)
    s.close()
    print("[*] Connessione chiusa. Uscita.") 
        


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tablut AI Player - AlphaZero-like bot for the Tablut Challenge"
    )
    # argomenti posizionali
    parser.add_argument(
        "role",
        type=str.upper, # Converte in automatico in maiuscolo (es. "white" -> "WHITE")
        choices=["WHITE", "BLACK"],
        help="Il ruolo assegnato al giocatore (WHITE o BLACK)"
    )
    
    parser.add_argument(
        "timeout",
        type=float,
        help="Il tempo massimo a disposizione per mossa in secondi (es. 60)"
    )
    
    parser.add_argument(
        "server_ip",
        type=str,
        help="L'indirizzo IP del Server Arbitro"
    )

    # argomenti opzionali
    parser.add_argument(
        "--port",
        type=int,
        default=-1,
        help="Porta del Server Arbitro (default:  BIANCO: 5800, NERO: 5801)"
    )
    parser.add_argument(
        "--time_margin",
        type=int,
        default=2,
        help="Margine di tempo rispetto al timer fornito dall'arbitro"
    )

    
    try:
        args = parser.parse_args()
    except SystemExit:
        # Se i parametri passati da bash sono errati, argparse fa exit in automatico.
        # Stampiamo un messaggio chiaro per il debug.
        print("\n[!] Errore nei parametri passati a Python. Controlla runmyplayer.sh")
        sys.exit(1)

    ruolo = args.role
    timeout_imposto = args.timeout
    ip_server = args.server_ip
    porta_server = (5800 if ruolo == "WHITE" else 5801) if args.port < 0 else args.port
    margin= args.time_margin

    # Calcola il tuo margine di sicurezza brutalmente vitale (es. -2.0 secondi)
    timeout_sicuro = timeout_imposto - margin
    if timeout_sicuro <= 0:
        # Previeni crash se il prof lancia la gara a tempi assurdi (es. 1 secondo)
        timeout_sicuro = timeout_imposto * 0.8 

    print(f"[*] Avvio Python: Ruolo={ruolo} | Timeout di sicurezza={timeout_sicuro:.1f}s | Arbitro={ip_server}:{porta_server}")

    sock= connettiti_all_arbitro(ip_arbitro=ip_server, porta_arbitro=porta_server, ruolo=ruolo)
    gioca_partita(sock, ruolo, timeout_sicuro)