from tablut.TablutGame import TablutGame
from tablut.NNet import NNetWrapper

def main():
    """""script semplice di conversione dei pesi del modello con parametri hardcoded :("""
    game = TablutGame()
    
    nnet = NNetWrapper(game)
    
    try:
        nnet.load_checkpoint(folder='./checkpoints/patch/', filename='best.pth.tar')
        print("Modello caricato con successo!")
    except Exception as e:
        print(f"Errore nel caricare il modello: {e}")
        return

    nnet.export_onnx(folder='.', filename='modello.onnx')
    print("Modello ONNX esportato correttamente!")

if __name__ == "__main__":
    main()