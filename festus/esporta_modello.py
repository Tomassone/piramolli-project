# FILE CHE DEVE USARE IL MEMBRO 3 (CHI ADDESTRA LA RETE)
import torch
from model import TablutNet  # Chiedigli se la classe si chiama così

# Carica i pesi del modello
model = TablutNet()
model.load_state_dict(torch.load("modello_addestrato.pth"))
model.eval()

# Crea un falso input (43 fogli, 9x9 caselle)
dummy_input = torch.randn(1, 43, 9, 9)

# Converte in formato ONNX
torch.onnx.export(
    model, 
    dummy_input, 
    "modello_onnx_finale.onnx", # Questo è il file che lui dovrà mandare a TE
    input_names=['board_state'],
    output_names=['policy', 'value'],
    dynamic_axes={'board_state': {0: 'batch_size'}, 'policy': {0: 'batch_size'}}
)
print("Modello ONNX generato!")