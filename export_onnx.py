from tablut.TablutGame import TablutGame
from tablut.NNet import NNetWrapper

def main():
    # 1. Istanziamo il gioco
    game = TablutGame()
    
    # 2. Inizializziamo il wrapper della rete neurale
    nnet = NNetWrapper(game)
    
    # 3. Carichiamo l'ultimo checkpoint allenato
    # Cambia './temp/' col percorso effettivo dove AlphaZero salva i modelli
    try:
        nnet.load_checkpoint(folder='./checkpoints/patch/', filename='temp.pth.tar')
        print("Modello caricato con successo!")
    except Exception as e:
        print(f"Errore nel caricare il modello: {e}")
        return

    # 4. Esportiamo in formato ONNX
    # Verrà creato un file chiamato "modello_4teste.onnx" nella cartella corrente
    nnet.export_onnx(folder='.', filename='modello_4teste.onnx')
    print("Modello ONNX esportato correttamente!")

if __name__ == "__main__":
    main()