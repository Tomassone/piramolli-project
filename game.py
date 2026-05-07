import numpy as np
import copy
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import random

# Si assume che questi moduli esistano nel progetto
from state import State
from action import Action
from GameAshtonTablut import GameAshtonTablut

class TablutGame:
    def __init__(self, repeated_moves_allowed: int = 3, cache_size: int = -1):
        self.game_engine = GameAshtonTablut(
            repeated_moves_allowed=repeated_moves_allowed,
            cache_size=cache_size,
            logs_folder="logs",
            white_name="White",
            black_name="Black"
        )
        self.board_size = 9
        # Action space: (from_row * 9 + from_col) * 81 + (to_row * 9 + to_col)
        # 81 * 81 = 6561
        self.action_size = 6561
        
        # Fixed spatial structures
        self.citadels = {
            (0,3), (0,4), (0,5), (1,4),
            (3,0), (4,0), (5,0), (4,1),
            (3,8), (4,8), (5,8), (4,7),
            (8,3), (8,4), (8,5), (7,4)
        }
        self.throne = {(4, 4)}
        self.escapes = {
            (0,1), (0,2), (0,6), (0,7),
            (8,1), (8,2), (8,6), (8,7),
            (1,0), (2,0), (6,0), (7,0),
            (1,8), (2,8), (6,8), (7,8)
        }

    def canonical_board(self, state: State) -> np.ndarray:
        """
        Ritorna uno stack di 8 piani per rappresentare lo stato in una prospettiva fissa:
        0: Pedine del giocatore corrente
        1: Pedine dell'avversario
        2: Re
        3: Trono 
        4: Campi (Citadels)
        5: Escapes
        6: Il giocatore di turno è Bianco (1) o Nero (0)
        """
        board = state.get_board()
        planes = np.zeros((7, 9, 9), dtype=np.float32)
        
        current_turn = state.get_turn().get_turn()
        # Se lo stato terminale è WW / BW / D il get_turn ha quella stringa, ci proteggiamo dal crash
        if current_turn not in ["W", "B"]:
            turn_is_white = 1.0  # Caso degenere per stato terminale, ma al terminal non chiamiamo la NN
        else:
            turn_is_white = 1.0 if current_turn == "W" else 0.0

        planes[6, :, :] = turn_is_white

        for r in range(9):
            for c in range(9):
                pawn = board[r][c].get_pawn()
                
                # Rappresentazione assoluta
                if pawn == "W":
                    planes[0, r, c] = 1.0
                elif pawn == "B":
                    planes[1, r, c] = 1.0
                elif pawn == "K":
                    planes[2, r, c] = 1.0

                if (r, c) in self.throne:
                    planes[3, r, c] = 1.0
                if (r, c) in self.citadels:
                    planes[4, r, c] = 1.0
                if (r, c) in self.escapes:
                    planes[5, r, c] = 1.0
        return planes

    def action_to_move(self, action: int) -> Action:
        from_idx = action // 81
        to_idx = action % 81
        r_from, c_from = from_idx // 9, from_idx % 9
        r_to, c_to = to_idx // 9, to_idx % 9
        
        column_from = chr(c_from + 97) # 'a' + c_from
        column_to = chr(c_to + 97)
        row_from = str(r_from + 1)
        row_to = str(r_to + 1)
        
        from_str = f"{column_from}{row_from}"
        to_str = f"{column_to}{row_to}"
        return Action(from_str, to_str, State.Turn.WHITE) 

    def get_valid_moves(self, state: State) -> np.ndarray:
        """
        Versione migliorata rispettando regole Ashton. 
        Nota: per garanzia massimale in torneo dovremo usare l'engine server Java,
        ma un buon mask ci aiuta moltissimo a filtrare a priori.
        """
        valid_moves = np.zeros(self.action_size, dtype=np.int8)
        turn = state.get_turn().get_turn()
        if turn not in ["W", "B"]:
            return valid_moves  # terminal state
            
        board = state.get_board()

        allowed_pawns = ["W", "K"] if turn == "W" else ["B"]
        
        for r in range(9):
            for c in range(9):
                pawn = board[r][c].get_pawn()
                if pawn not in allowed_pawns:
                    continue
                
                # Check 4 orthorgonal directions
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    tr, tc = r + dr, c + dc
                    while 0 <= tr < 9 and 0 <= tc < 9:
                        # Ostacoli: pedine
                        target_pawn = board[tr][tc].get_pawn()
                        if target_pawn != "E" and target_pawn != "T":
                            break
                        
                        # Ostacolo: Trono (Solo Re si ferma, ma nessuno ci passa in Ashton Tablut)
                        if (tr, tc) in self.throne and pawn != "K":
                            break
                            
                        # Campi (Citadels): un pezzo bianco non entra mai.
                        # Un pezzo nero entra SOLO se è partito da quel campo.
                        if (tr, tc) in self.citadels:
                            if turn == "W":
                                break
                            if turn == "B" and (r, c) not in self.citadels:
                                break
                            
                        # Mossa valida trovata
                        from_idx = r * 9 + c
                        to_idx = tr * 9 + tc
                        valid_moves[from_idx * 81 + to_idx] = 1
                        
                        tr += dr
                        tc += dc

        # Check safety (opzionale): filtrare con self.game_engine.check_move se vogliamo il 100%. 
        # (Lento, omesso per ora, fidiamoci delle mask ortogonali)
        # Se la valid mask fosse vuota (stallo) ritorneremmo tutto 0.
        return valid_moves

    def get_next_state(self, state: State, action: int) -> State:
        # Usa GameAshtonTablut per applicare la mossa ed effettuare i check
        act = self.action_to_move(action)
        act.set_turn(state.get_turn()) 
        next_state = copy.deepcopy(state)
        # Se la mossa qui passa la mask ma per l'engine è illegale (e.g. cattura omessa, eccezioni), 
        # check_move crasherà con ActionException / Exception varie.
        # Il try-catch servirà se la maschera è imprecisa. Assumiamo funzioni per ora.
        next_state = self.game_engine.check_move(next_state, act)
        return next_state

    def get_game_ended(self, state: State, perspective_turn: str) -> float:
        """
        Ritorna outcome dalla prospettiva del giocatore (perspective_turn = "W" o "B").
        Vittoria = 1.0, Sconfitta = -1.0, Pareggio = 0.0, Non terminale = None.
        """
        turn = state.get_turn().get_turn()
        # White Win
        if turn == "WW": 
            return 1.0 if perspective_turn == "W" else -1.0
        # Black Win
        elif turn == "BW":
            return 1.0 if perspective_turn == "B" else -1.0
        # Draw (Ripetizione 3x)
        elif turn == "D":
            return 0.0
            
        # Non terminato
        return None 

    def get_canonical_board(self, state: State):
        return self.canonical_board(state)

# Step 5: Architettura più leggera per VM (AlphaZero ResNet lite o CNN piccola)
class TablutNNet(nn.Module):
    def __init__(self):
        super(TablutNNet, self).__init__()
        # Input: 7 canali, per mantenere bassi i pesi usiamo 64 filters in body network
        self.conv1 = nn.Conv2d(7, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Policy Head (alpha-zero style, conv to reduce weights)
        self.conv_policy = nn.Conv2d(64, 2, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(2)
        self.fc_policy = nn.Linear(2 * 9 * 9, 6561)
        
        # Value Head
        self.conv_value = nn.Conv2d(64, 1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value1 = nn.Linear(9 * 9, 32)
        self.fc_value2 = nn.Linear(32, 1)

    def forward(self, x):
        # Body
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Policy
        p = F.relu(self.bn_policy(self.conv_policy(x)))
        p = p.view(-1, 2 * 9 * 9)
        policy_logits = self.fc_policy(p)
        
        # Value
        v = F.relu(self.bn_value(self.conv_value(x)))
        v = v.view(-1, 9 * 9)
        v = F.relu(self.fc_value1(v))
        value = torch.tanh(self.fc_value2(v))
        
        return policy_logits, value

    def predict(self, state_planes, valid_mask):
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state_planes).unsqueeze(0)
            logits, v = self.forward(x)
            
            logits = logits.squeeze(0).numpy()
            v = v.item()
            
            # Masking out illegal moves with very low logits
            logits[valid_mask == 0] = -1e8
            
            # Safety check: if no legal moves at all (stallo), return uniform over mask 
            # In Tablut se uno non ha mosse (improbabile prima della vittoria dell'altro), perde/patta.
            
            # Max pre-softmax
            m = np.max(logits)
            pi = np.exp(logits - m)
            pi_sum = np.sum(pi)
            if pi_sum > 0:
                pi /= pi_sum
            else:
                # Se tutti invalid, patta matematica per non crashare: rari casin MCTS
                pi = valid_mask / max(1, np.sum(valid_mask))
            
            return pi, v


class MCTS:
    def __init__(self, game: TablutGame, net: TablutNNet, num_simulations=50, cpuct=1.5):
        self.game = game
        self.net = net
        self.num_simulations = num_simulations
        self.cpuct = cpuct
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.P = {}
        self.Vs = {}  # Valid moves cache

    def get_action_prob(self, state: State, temp=1):
        for _ in range(self.num_simulations):
            self.search(state)
            
        s = str(state)
        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.action_size)]
        
        if sum(counts) == 0:
            # Fallback se non espande: usiamo una distribuzione uniforme sulle mosse legali.
            # Questo evita che np.random.choice fallisca dopo (somma 0 invece che 1).
            valid_moves = self.game.get_valid_moves(state)
            if np.sum(valid_moves) > 0:
                probs = (valid_moves / np.sum(valid_moves)).tolist()
                return probs
            else:
                return [1.0 / self.game.action_size] * self.game.action_size
            
        if temp == 0:
            best_A = np.argmax(counts)
            probs = [0.0] * len(counts)
            probs[best_A] = 1.0
            return probs
            
        counts = [x ** (1. / temp) for x in counts]
        probs = [x / float(sum(counts)) for x in counts]
        
        return probs

    def search(self, state: State):
        s = str(state)
        # Prospettiva è il giocatore corrente a questo nodo
        current_turn = state.get_turn().get_turn()
        
        # Controlliamo la terminalità
        ended = self.game.get_game_ended(state, perspective_turn=current_turn)
        if ended is not None:
            return -ended 
            
        if s not in self.P:
            # Leaf node expand
            planes = self.game.get_canonical_board(state)
            mask = self.game.get_valid_moves(state)
            self.P[s], v = self.net.predict(planes, mask)
            
            self.Vs[s] = mask
            self.Ns[s] = 0
            return -v
            
        valid_moves = self.Vs[s]
        best_u = -float('inf')
        best_a = -1
        
        for a in range(self.game.action_size):
            if valid_moves[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.cpuct * self.P[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.cpuct * self.P[s][a] * math.sqrt(self.Ns[s] + 1e-8)
                    
                if u > best_u:
                    best_u = u
                    best_a = a
                    
        a = best_a
        next_state = self.game.get_next_state(state, a)
        
        v = self.search(next_state)
        
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
            
        self.Ns[s] += 1
        return -v

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        
    def add(self, examples):
        # examples is a list of (canonical_board, pi, z)
        self.buffer.extend(examples)
        if len(self.buffer) > self.capacity:
            trim_amount = len(self.buffer) - self.capacity
            del self.buffer[:trim_amount]
            
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# Trainer Upgrade
class Trainer:
    def __init__(self, game: TablutGame, net: TablutNNet):
        self.game = game
        self.net = net
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.epochs = 1
        
    def train_step(self, examples: List[Tuple[np.ndarray, np.ndarray, float]], batch_size=32):
        self.net.train()
        pi_losses = []
        v_losses = []
        
        for epoch in range(self.epochs):
            # Crea mini-batch
            np.random.shuffle(examples)
            for i in range(0, len(examples), batch_size):
                batch = examples[i:i+batch_size]
                planes, pis, vs = list(zip(*batch))
                
                planes = torch.FloatTensor(np.array(planes)) # (b, 7, 9, 9)
                pis = torch.FloatTensor(np.array(pis))       # (b, 6561)
                vs = torch.FloatTensor(np.array(vs)).unsqueeze(1) # (b, 1)
                
                self.optimizer.zero_grad()
                out_pi, out_v = self.net(planes)
                
                log_pi = F.log_softmax(out_pi, dim=1)
                
                l_pi = -torch.sum(pis * log_pi) / pis.size()[0]
                l_v = F.mse_loss(out_v, vs)
                
                total_loss = l_pi + l_v
                total_loss.backward()
                self.optimizer.step()
                
                pi_losses.append(l_pi.item())
                v_losses.append(l_v.item())
                
        return np.mean(pi_losses), np.mean(v_losses)

    def save_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save({'state_dict': self.net.state_dict()}, filepath)

    def load_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            self.net.load_state_dict(checkpoint['state_dict'])
            print("Checkpoint Caricato!")

# Esercizio con Replay Buffer e noise
def execute_episode(game: TablutGame, net: TablutNNet, mcts_simulations: int = 10):
    mcts = MCTS(game, net, num_simulations=mcts_simulations)
    train_examples = []
    
    try:
        from state_tablut import StateTablut
        state = StateTablut()
    except ImportError:
        return []
    
    step = 0
    while True:
        step += 1
        temp = 1 if step < 15 else 0 
        
        # Dirichlet Noise in root per explorazione locale (Omesso per brevità nel get_action_prob ma idealmente qui)
        pi = mcts.get_action_prob(state, temp=temp)
        
        valid_moves = game.get_valid_moves(state)
        if sum(valid_moves) == 0:
            print("NESSUNA MOSSA VALIDA/STALLO")
            return []
            
        planes = game.get_canonical_board(state)
        # Salva record: piano in ingresso prospettico (e.g. W current), policy e chi è current player
        current_player = state.get_turn().get_turn()
        train_examples.append([planes, pi, current_player])
        
        action = np.random.choice(len(pi), p=pi)
        
        state = game.get_next_state(state, action)
        
        # Check outcome per il TURN CHE HA APPENA MOSSO (o un controllo globale)
        # Lo state adesso ha `turn` skippato all'avversario (o allo stato terminale)
        
        # Per rimettere on-track il `get_game_ended`:
        r = game.get_game_ended(state, perspective_turn="W")  # Prendiamo global perspective su 'W'
        
        if r is not None:
            # Abbiamo un outcome dal POV di 'W'.
            # r = 1 se vince W, -1 se vince B, 0 se pari.
            final_examples = []
            for x in train_examples:
                # x[2] è il marker se stava giocando 'W' o 'B'
                is_white_turn = (x[2] == "W")
                # Il reward che l'utente al tempo 'x[2]' doveva predire:
                # se era White ed ha finto White, z=1. Se era Black ed ha vinto White, z=-1.
                z = r if is_white_turn else -r
                final_examples.append((x[0], x[1], z))
                
            return final_examples

if __name__ == "__main__":
    print("Inizializzando AlphaZero Tablut Pipeline...")
    tablut_game = TablutGame(repeated_moves_allowed=3, cache_size=-1)
    tablut_net = TablutNNet()
    tablut_trainer = Trainer(tablut_game, tablut_net)
    
    # Replay buffer (capacità piccola per test)
    replay_buffer = ReplayBuffer(capacity=1000)
    
    num_episodes = 2
    
    for i in range(num_episodes):
        print(f"--- EPISODIO {i+1}/{num_episodes} ---")
        episode_data = execute_episode(tablut_game, tablut_net, mcts_simulations=40)
        
        if len(episode_data) > 0:
            replay_buffer.add(episode_data)
            print(f"Aggiunti {len(episode_data)} states al Replay Buffer. Size={len(replay_buffer.buffer)}")
        
        # Addestriamo la rete neurale su un mini-batch estratto dal buffer
        if len(replay_buffer.buffer) >= 32:
            print(f"Addestrando la rete su mini-batch estratto...")
            batch = replay_buffer.sample(32)
            pi_loss, v_loss = tablut_trainer.train_step(batch, batch_size=32)
            print(f"Loss Policy: {pi_loss:.4f} | Loss Value: {v_loss:.4f}\n")
        
        # Salva checkpoint periodicamente
        tablut_trainer.save_checkpoint(folder="checkpoints", filename="best_model.pth.tar")

