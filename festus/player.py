import time
import json
import socket
import struct
import onnxruntime as ort
import sys
import argparse
from TablutGame import TablutGame 
import copy

# ==============================================================================
# QUI IMPORTI I CODICI DEI TUOI COLLEGHI (QUANDO SARANNO PRONTI)
# Esempio:
# from tablut_ambiente import TablutBoard  # Dal Membro 1
# from mcts_player import MCTS             # Dal Membro 2
# ==============================================================================

def init_onnx_engine(model_path="modello_onnx_finale.onnx"):
    """
    Inizializza ONNX Runtime sfruttando tutti e 4 i Core della Macchina Virtuale.
    Questo si esegue PRIMA che inizi la partita, così non ruba tempo al timer.
    """
    print(f"[*] Avvio motore ONNX (Model: {model_path})...")
    opzioni = ort.SessionOptions()
    # 4 Thread per sfruttare i 4 vCPU della gara
    opzioni.intra_op_num_threads = 4
    
    try:
        motore = ort.InferenceSession(model_path, opzioni)
        print("[*] Motore ONNX pronto.")
        return motore
    except Exception as e:
        print(f"[!] ERRORE ONNX: Impossibile caricare il modello. {e}")
        return None

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


def pensa_e_muovi(scacchiera_json, motore_onnx, tempo_sicuro_disponibile):
    """
    mantenere lo satto esternamente 

    """
    print(f"[*] Turno iniziato! Orologio partito per {tempo_sicuro_disponibile:.1f} secondi.")
    tempo_inizio = time.time()
    
    # -------------------------------------------------------------
    # 1. QUI IL MEMBRO 1 INIZIALIZZA LA SCACCHIERA DAL JSON
    # Esempio:
    # stato_attuale = TablutBoard.from_json(scacchiera_json)
    # -------------------------------------------------------------
    
    # -------------------------------------------------------------
    # 2. QUI IL MEMBRO 2 INIZIALIZZA L'ALBERO MCTS
    # Esempio:
    # mcts = MCTS(motore=motore_onnx)
    # -------------------------------------------------------------
    
    iterazioni = 0
    
    # 3. IL CICLO DELLA SICUREZZA
    while (time.time() - tempo_inizio) < tempo_sicuro_disponibile:
        
        # ---------------------------------------------------------
        # QUI L'MCTS ESPLORA I RAMI (Questa è la riga pesante)
        # Esempio:
        # mcts.esplora_un_ramo(stato_attuale)
        # ---------------------------------------------------------
        
        iterazioni += 1
        
        # -- RIGA FINTO-MCTS (DA CANCELLARE QUANDO HAI IL CODICE VERO) --
        time.sleep(0.001)  # Finge di pensare per 1 millisecondo
        # ---------------------------------------------------------------
     
    print(f"[*] Fine pensiero. Valutati {iterazioni} percorsi.")

    # -------------------------------------------------------------
    # 4. CHIEDI AL MCTS QUAL È LA MOSSA MIGLIORE E PREPARA LA STRINGA
    # Esempio:
    # mossa_migliore = mcts.dammi_mossa_migliore()
    # -------------------------------------------------------------
    
    # Ritorna la mossa in formato stringa per mandarla all'arbitro
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
    motore = init_onnx_engine()


    game = TablutGame()
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
            
            mossa_decisa = pensa_e_muovi(state_root, motore, timeout)
            
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