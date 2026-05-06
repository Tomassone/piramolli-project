import time
import json
import socket
import onnxruntime as ort
import sys
import argparse

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

    """
    Questa funzione fa partire il TIMER DI 58 SECONDI.
    Si innesca nel momento esatto in cui riceviamo la scacchiera dall'arbitro.
    """
    print("[*] Turno iniziato! Orologio partito.")
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
    
    # 3. IL CICLO DELLA SICUREZZA (58.0 secondi)
    while (time.time() - tempo_inizio) < 58.0:
        
        # ---------------------------------------------------------
        # QUI L'MCTS ESPLORA I RAMI (Questa è la riga pesante)
        # Esempio:
        # mcts.esplora_un_ramo(stato_attuale)
        # ---------------------------------------------------------
        
        iterazioni += 1
        
        # -- RIGA FINTO-MCTS (DA CANCELLARE QUANDO HAI IL CODICE VERO) --
        time.sleep(0.001)  # Finge di pensare per 1 millisecondo
        # ---------------------------------------------------------------

    # Il tempo di 58.0 secondi è scaduto!
    print(f"[*] Fine pensiero. Ho valutato {iterazioni} percorsi possibili.")
    
    # -------------------------------------------------------------
    # 4. CHIEDI AL MCTS QUAL È LA MOSSA MIGLIORE E PREPARA LA STRINGA
    # Esempio:
    # mossa_migliore = mcts.dammi_mossa_migliore()
    # -------------------------------------------------------------
    
    # Ritorna la mossa in formato stringa per mandarla all'arbitro
    mossa_finta = "E2-E4"  # Sostituisci con la vera mossa decisa dall'MCTS
    return mossa_finta

def pensa_e_muovi(scacchiera_json, motore_onnx, tempo_sicuro_disponibile):
    """
    Questa funzione fa partire il TIMER.
    Si ferma dinamicamente basandosi sul parametro 'tempo_sicuro_disponibile'.
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
    mossa_finta = "E2-E4"  # Sostituisci con la vera mossa decisa dall'MCTS
    return mossa_finta

def connettiti_all_arbitro(ip_arbitro, porta_arbitro, ruolo, timeout):
    """
    Simula la connessione al server Arbitro via Socket.
    """
    motore = init_onnx_engine()
    
    # Questo print usa i parametri che abbiamo estratto tramite argparse
    print(f"[*] In attesa di connettersi all'Arbitro su {ip_arbitro}:{porta_arbitro} come {ruolo}...")
    
    # --- LOGICA DI CONNESSIONE SOCKET VERA (Da inserire in futuro) ---
    # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.connect((ip_arbitro, porta_arbitro))
    # s.send(ruolo.encode())  # Es: Diciamo al server "Sono il WHITE"
    # -----------------------------------------------------------------

    # Finto ciclo per simulare la gara in attesa del vero codice socket
    for turno in range(1, 4):
        print(f"\n--- In attesa del Turno {turno} dall'Arbitro ---")
        time.sleep(2) # Simula l'attesa fisica
        
        # L'arbitro ci manda il JSON della scacchiera
        finto_json_ricevuto_dall_arbitro = '{"turn": "WHITE", "board": "..."}'
        print("[*] L'Arbitro ha mandato la scacchiera! È il mio turno.")
        
        # -> NOTA CHIAVE: Qui passiamo il 'timeout' dinamico calcolato da argparse!
        mossa_decisa = pensa_e_muovi(finto_json_ricevuto_dall_arbitro, motore, timeout)
        
        print(f"[*] Invio mossa all'Arbitro: {mossa_decisa}")
        # s.send(mossa_decisa.encode())
        
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
        default=8901,
        help="Porta del Server Arbitro (default: 8901)"
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
    porta_server = args.port
    margin= args.time_margin

    # Calcola il tuo margine di sicurezza brutalmente vitale (es. -2.0 secondi)
    timeout_sicuro = timeout_imposto - margin
    if timeout_sicuro <= 0:
        # Previeni crash se il prof lancia la gara a tempi assurdi (es. 1 secondo)
        timeout_sicuro = timeout_imposto * 0.8 

    print(f"[*] Avvio Python: Ruolo={ruolo} | Timeout di sicurezza={timeout_sicuro:.1f}s | Arbitro={ip_server}:{porta_server}")

    connettiti_all_arbitro(ip_arbitro=ip_server, porta_arbitro=porta_server, ruolo=ruolo, timeout=timeout_sicuro)