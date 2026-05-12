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
        pw = pw.reshape(pw.size(0), -1)
        pw_logits = self.policy_w_fc(pw)
        
        # Policy Black
        pb = torch.relu(self.policy_b_bn(self.policy_b_conv(x)))
        pb = pb.reshape(pb.size(0), -1)
        pb_logits = self.policy_b_fc(pb)
        
        # Value White
        vw = torch.relu(self.value_w_bn(self.value_w_conv(x)))
        vw = vw.reshape(vw.size(0), -1)
        vw = torch.relu(self.value_w_fc1(vw))
        vw_val = torch.tanh(self.value_w_fc2(vw))
        
        # Value Black
        vb = torch.relu(self.value_b_bn(self.value_b_conv(x)))
        vb = vb.reshape(vb.size(0), -1)
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
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0002, weight_decay=1e-4)

    def train(self, examples):
        if not examples: return 0.0
        self.model.train()

        batch_size = 256
        epochs = 10  

        import random
        import gc

        total_training_loss = 0.0
        total_batches_all_epochs = 0
        
        for epoch in range(epochs):
            np.random.shuffle(examples)
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(examples), batch_size):
                batch_examples = examples[i:i + batch_size]

                boards, pis, vs = [], [], []
                for board, pi, v in batch_examples:
                    if isinstance(board, dict):
                        tensor_board = self.game.encode_state(board)
                    else:
                        tensor_board = board
                    boards.append(torch.FloatTensor(tensor_board))
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
                
                # Moltiplichiamo le loss per le maschere
                loss_white = (loss_pw + loss_vw) * mask_white
                loss_black = (loss_pb + loss_vb) * mask_black
                
                total_loss = (loss_white + loss_black).mean()
                
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

                epoch_loss += total_loss.item()
                num_batches += 1
            
            total_training_loss += epoch_loss
            total_batches_all_epochs += num_batches

        gc.collect()
        torch.cuda.empty_cache()

        # Restituisce la loss media dell'intera epoca
        return total_training_loss / total_batches_all_epochs if total_batches_all_epochs > 0 else 0.0

    def train2(self, examples):
        if not examples:
            return 0.0

        self.model.train()

        epochs = 10
        batch_size = 128

        def extract_board(e):
            b = e[0]
            if isinstance(b, dict):
                return torch.as_tensor(
                    self.game.encode_state(b), dtype=torch.float32
                )
            return torch.as_tensor(b, dtype=torch.float32)

        boards_all = torch.stack(
            [extract_board(e) for e in examples],
            dim=0
        ).to(self.device)

        pis_all = torch.stack(
            [torch.as_tensor(e[1], dtype=torch.float32) for e in examples],
            dim=0
        ).to(self.device)  # (N, action_size)

        vs_all = torch.as_tensor(
            [[e[2]] for e in examples],
            dtype=torch.float32,
            device=self.device
        )  # (N, 1)

        turns = boards_all[:, 0, 0, 25]  # (N,)
        idx_white = (turns == 1.0).nonzero(as_tuple=True)[0]
        idx_black = (turns != 1.0).nonzero(as_tuple=True)[0]

        total_loss_accum = 0.0
        num_steps = 0

        for _ in range(epochs):

            if idx_white.numel() > 0:
                perm_w = idx_white[torch.randperm(idx_white.numel(), device=self.device)]
                for i in range(0, perm_w.numel(), batch_size):
                    idx = perm_w[i:i + batch_size]

                    b = boards_all[idx].permute(0, 3, 1, 2)  # (B, 28, 9, 9)
                    pi = pis_all[idx]
                    v = vs_all[idx]

                    pw_logits, vw_val, _, _ = self.model(b)

                    loss_pi_w = -torch.sum(
                        pi * torch.log_softmax(pw_logits, dim=1), dim=1
                    ).mean()
                    loss_v_w = torch.mean((v - vw_val) ** 2)
                    loss = loss_pi_w + loss_v_w

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    total_loss_accum += loss.detach().item()
                    num_steps += 1

            if idx_black.numel() > 0:
                perm_b = idx_black[torch.randperm(idx_black.numel(), device=self.device)]
                for i in range(0, perm_b.numel(), batch_size):
                    idx = perm_b[i:i + batch_size]

                    b = boards_all[idx].permute(0, 3, 1, 2)  # (B, 28, 9, 9)
                    pi = pis_all[idx]
                    v = vs_all[idx]

                    _, _, pb_logits, vb_val = self.model(b)

                    loss_pi_b = -torch.sum(
                        pi * torch.log_softmax(pb_logits, dim=1), dim=1
                    ).mean()
                    loss_v_b = torch.mean((v - vb_val) ** 2)
                    loss = loss_pi_b + loss_v_b

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    total_loss_accum += loss.detach().item()
                    num_steps += 1

        return total_loss_accum / max(num_steps, 1)

    def predict(self, board):
        self.model.eval()
        with torch.no_grad():
            # Leggiamo chi deve muovere direttamente dal tensore
            if isinstance(board, dict) and 'turn_to_move' in board:
                turn_is_white = (board['turn_to_move'] == 1)
            else:
                # Fallback di sicurezza: se non troviamo l'info, proviamo a guardare il piano 25
                # se per caso board è già stato codificato in un array Numpy
                turn_is_white = True
                if isinstance(board, np.ndarray) and len(board.shape) == 3:
                    turn_is_white = (board[0, 0, 25] == 1.0)

            if isinstance(board, dict):
                tensor_board = self.game.encode_state(board)
            else:
                tensor_board = board

            board_tensor = torch.FloatTensor(tensor_board).to(self.device)
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


    def export_onnx(self, folder='.', filename='modello_4teste.onnx'):
        import torch
        import torch.onnx
        import os

        filepath = os.path.join(folder, filename)
        
        # modello in modalità valutazione
        self.model.eval()
        
        # crea un tensore dummy della forma esatta che si aspetta la rete
        dummy_input = torch.randn(1, 28, 9, 9).to(self.device)
        
        # esporta usando il tracer classico
        import torch.onnx
        
        torch.onnx.export(
            self.model,                     # il modello PyTorch
            dummy_input,                    # l'input di test
            filepath,                       # dove salvare il file
            export_params=True,             # salva i pesi allenati all'interno del file
            opset_version=14,               # la versione ONNX (14 è stabilissima)
            do_constant_folding=True,       # ottimizzazione
            input_names=['input_board'],    # il nome dell'input che useremo in inferenza
            output_names=['pw_logits', 'vw_val', 'pb_logits', 'vb_val'], # i 4 output
            dynamic_axes={
                'input_board': {0: 'batch_size'},
                'pw_logits': {0: 'batch_size'},
                'vw_val': {0: 'batch_size'},
                'pb_logits': {0: 'batch_size'},
                'vb_val': {0: 'batch_size'}
            },
            dynamo=False  
        )


