import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from NeuralNet import NeuralNet

class TablutNet(nn.Module):
    def __init__(self, board_height=9, board_width=9, num_channels=28, action_size=6561):
        super(TablutNet, self).__init__()
        self.board_height = board_height
        self.board_width = board_width
        
        self.conv1 = nn.Conv2d(num_channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        
        self.res_blocks = nn.ModuleList([
            self._make_residual_block(128, 128) for _ in range(3)
        ])
        
        # ─── 1. Policy Head BIANCO ───
        self.policy_w_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.policy_w_bn = nn.BatchNorm2d(32)
        self.policy_w_fc = nn.Linear(32 * board_height * board_width, action_size)
        
        # ─── 2. Policy Head NERO ───
        self.policy_b_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.policy_b_bn = nn.BatchNorm2d(32)
        self.policy_b_fc = nn.Linear(32 * board_height * board_width, action_size)
        
        # ─── 3. Value Head BIANCO ───
        self.value_w_conv = nn.Conv2d(128, 3, kernel_size=1)
        self.value_w_bn = nn.BatchNorm2d(3)
        self.value_w_fc1 = nn.Linear(3 * board_height * board_width, 64)
        self.value_w_fc2 = nn.Linear(64, 1)
        
        # ─── 4. Value Head NERO ───
        self.value_b_conv = nn.Conv2d(128, 3, kernel_size=1)
        self.value_b_bn = nn.BatchNorm2d(3)
        self.value_b_fc1 = nn.Linear(3 * board_height * board_width, 64)
        self.value_b_fc2 = nn.Linear(64, 1)

    def _make_residual_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, state):
        x = torch.relu(self.bn1(self.conv1(state)))
        for res_block in self.res_blocks:
            x = torch.relu(x + res_block(x))
            
        # Policy White
        pw = torch.relu(self.policy_w_bn(self.policy_w_conv(x)))
        pw = pw.view(pw.size(0), -1)
        pw_logits = self.policy_w_fc(pw)
        
        # Policy Black
        pb = torch.relu(self.policy_b_bn(self.policy_b_conv(x)))
        pb = pb.view(pb.size(0), -1)
        pb_logits = self.policy_b_fc(pb)
        
        # Value White
        vw = torch.relu(self.value_w_bn(self.value_w_conv(x)))
        vw = vw.view(vw.size(0), -1)
        vw = torch.relu(self.value_w_fc1(vw))
        vw_val = torch.tanh(self.value_w_fc2(vw))
        
        # Value Black
        vb = torch.relu(self.value_b_bn(self.value_b_conv(x)))
        vb = vb.view(vb.size(0), -1)
        vb = torch.relu(self.value_b_fc1(vb))
        vb_val = torch.tanh(self.value_b_fc2(vb))
        
        # Restituisce tutto, sarà il wrapper a decidere cosa usare
        return pw_logits, vw_val, pb_logits, vb_val


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.game = game
        self.board_height, self.board_width = game.getBoardSize()
        self.action_size = game.getActionSize()
        
        self.model = TablutNet(self.board_height, self.board_width, 28, self.action_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, examples):
        if not examples: return 0.0
        self.model.train()
        
        boards, pis, vs = [], [], []
        for board, pi, v in examples:
            boards.append(torch.FloatTensor(board))
            pis.append(torch.FloatTensor(pi))
            vs.append(torch.FloatTensor([v]))
            
        boards = torch.stack(boards).to(self.device) # (B, 9, 9, 28)
        pis = torch.stack(pis).to(self.device)       # (B, ActionSize)
        vs = torch.stack(vs).to(self.device)         # (B, 1)
        
        # Estraiamo il turno PRIMA di fare il permute
        # In encode_state il Plane 25 (indice 25) è 1.0 se Bianco, 0.0 se Nero
        turns = boards[:, 0, 0, 25].unsqueeze(1)     # (B, 1) - Maschera Bianco
        mask_white = turns
        mask_black = 1.0 - turns                     # (B, 1) - Maschera Nero
        
        boards = boards.permute(0, 3, 1, 2)          # (B, 28, 9, 9)
        
        # Forward pass (calcola tutte e 4 le teste)
        pw_logits, vw_val, pb_logits, vb_val = self.model(boards)
        
        # Loss per le Policy (usiamo la formula manuale per evitare problemi con i batch mascherati)
        loss_pw = -torch.sum(pis * torch.log_softmax(pw_logits, dim=1), dim=1, keepdim=True)
        loss_pb = -torch.sum(pis * torch.log_softmax(pb_logits, dim=1), dim=1, keepdim=True)
        
        # Loss per le Value
        loss_vw = (vs - vw_val) ** 2
        loss_vb = (vs - vb_val) ** 2
        
        # IL TRUCCO: Moltiplichiamo le loss per le maschere.
        # Se era il turno del Bianco (mask_white=1), la loss nera si azzera e viceversa!
        loss_white = (loss_pw + loss_vw) * mask_white
        loss_black = (loss_pb + loss_vb) * mask_black
        
        total_loss = (loss_white + loss_black).mean()
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()

    def predict(self, board):
        self.model.eval()
        with torch.no_grad():
            # Leggiamo chi deve muovere direttamente dal tensore
            turn_is_white = (board[0, 0, 25] == 1.0)
            
            board_tensor = torch.FloatTensor(board).to(self.device)
            board_tensor = board_tensor.permute(2, 0, 1).unsqueeze(0)
            
            pw_logits, vw_val, pb_logits, vb_val = self.model(board_tensor)
            
            # Restituiamo all'MCTS SOLO l'output del giocatore che deve muovere
            if turn_is_white:
                pi = torch.softmax(pw_logits, dim=1)[0].cpu().numpy()
                v = vw_val[0].item()
            else:
                pi = torch.softmax(pb_logits, dim=1)[0].cpu().numpy()
                v = vb_val[0].item()
                
        return pi, v

    def save_checkpoint(self, folder, filename):
        if not os.path.exists(folder): os.makedirs(folder)
        torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, os.path.join(folder, filename))

    def load_checkpoint(self, folder, filename):
        checkpoint = torch.load(os.path.join(folder, filename), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def export_onnx(self, folder, filename="modello_4teste.onnx"):
        if not os.path.exists(folder): os.makedirs(folder)
        filepath = os.path.join(folder, filename)
        self.model.eval()
        dummy_input = torch.randn(1, 28, self.board_height, self.board_width).to(self.device)
        torch.onnx.export(
            self.model, dummy_input, filepath, export_params=True, opset_version=14,
            do_constant_folding=True, input_names=['input_board'],
            output_names=['pw_logits', 'vw_val', 'pb_logits', 'vb_val'],
            dynamic_axes={'input_board': {0: 'batch_size'}}
        )


