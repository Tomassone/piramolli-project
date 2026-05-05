import time
import json
import socket
import onnxruntime as ort

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

def pensa_e_muovi(scacchiera_json, motore_onnx):
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

def connettiti_all_arbitro(ip_arbitro="127.0.0.1", porta_arbitro=8901):
    """
    Simula la connessione al server Arbitro via Socket.
    (Il Membro 1 o 2 dovranno adattare questa parte al protocollo JSON esatto richiesto)
    """
    motore = init_onnx_engine()
    
    print(f"[*] In attesa di connettersi all'Arbitro su {ip_arbitro}:{porta_arbitro}...")
    
    # Questo è un finto ciclo di gioco (per testarlo in locale)
    # Nella competizione reale, qui ci sarà un socket TCP (s.connect) che resta in ascolto
    for turno in range(1, 4):  # Simuliamo 3 turni
        print(f"\n--- In attesa del Turno {turno} dall'Arbitro ---")
        time.sleep(2) # Simula l'attesa per la mossa dell'avversario
        
        # L'arbitro ci manda il JSON della scacchiera
        finto_json_ricevuto_dall_arbitro = '{"turn": "WHITE", "board": "..."}'
        
        print("[*] L'Arbitro ha mandato la scacchiera! È il mio turno.")
        
        # Chiamiamo la nostra AI (innesca i 58 secondi)
        mossa_decisa = pensa_e_muovi(finto_json_ricevuto_dall_arbitro, motore)
        
        # Mandiamo la mossa all'arbitro
        print(f"[*] Invio mossa all'Arbitro: {mossa_decisa}")
        
if __name__ == "__main__":
    # Quando la giuria lancia il file, fa partire questo:
    connettiti_all_arbitro()