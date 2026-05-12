import numpy as np
import onnxruntime as ort

class ONNXNetWrapper:
    def __init__(self, onnx_model_path, game):
        """
        Carica il modello ONNX e si prepara a imitare il comportamento
        della classe NNetWrapper originale.
        """
        self.game=game
        print(f"[*] Avvio motore ONNX (Model: {onnx_model_path})...")
        opzioni = ort.SessionOptions()
        # 4 Thread per sfruttare i 4 vCPU della gara
        opzioni.intra_op_num_threads = 4
        
        try:
            motore = ort.InferenceSession(onnx_model_path, opzioni)
            print("[*] Motore ONNX pronto.")
            self.session=motore
        except Exception as e:
            print(f"[!] ERRORE ONNX: Impossibile caricare il modello. {e}")
            return None
    
    def predict(self, board):
        """
        Sostituisce esattamente il vecchio nnet.predict() di PyTorch.
        L'MCTS chiamerà questo metodo passando il dizionario o l'array del board.
        """
        if isinstance(board, dict) and 'turn_to_move' in board:
            turn_is_white = (board['turn_to_move'] == 1)
        else:
            turn_is_white = True
            if isinstance(board, np.ndarray) and len(board.shape) == 3:
                turn_is_white = (board[0, 0, 25] == 1.0)
        
        if isinstance(board, dict):
            board_numpy = self.game.encode_state(board)
        else:
            board_numpy = board

        board_input = np.expand_dims(np.transpose(board_numpy, (2, 0, 1)), axis=0)
        
        board_input = board_input.astype(np.float32)
        outputs = self.session.run(None, {'input_board': board_input})
        
        # outputs[0] -> pw_logits 
        # outputs[1] -> vw_val    
        # outputs[2] -> pb_logits 
        # outputs[3] -> vb_val    
        
        if turn_is_white:
            logits = outputs[0][0]  
            v = outputs[1][0][0]    
        else:
            logits = outputs[2][0]
            v = outputs[3][0][0]
            
        exp_logits = np.exp(logits - np.max(logits))
        pi = exp_logits / np.sum(exp_logits)
        
        return pi, float(v)