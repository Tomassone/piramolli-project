import os

# Create base definitions
types_py = """
from typing import Literal, List, Tuple
import numpy as np
from dataclasses import dataclass

State = dict
Action = Tuple[Tuple[int, int], Tuple[int, int]]
Player = Literal[1, -1]
Winner = Literal[1, -1, 0]

ATTACKER = 1
DEFENDER = -1

@dataclass
class EncodingInput:
    white_positions: List[Tuple[int, int]]
    black_positions: List[Tuple[int, int]]
    king_position: Tuple[int, int]
    side_to_move: Player
    move_count: int
    half_move_clock: int
    position_history: List[str]

@dataclass
class TrainingSample:
    state: np.ndarray   # float32, shape [43, 9, 9] (channels-first)
    pi: np.ndarray      # float32, shape [2592]
    z: float            # float32, scalar in {-1.0, 0.0, +1.0}
    player: Player      # +1 = attacker, -1 = defender
"""

config_py = """
from dataclasses import dataclass
import torch

@dataclass
class Config:
    # Optimizer
    weight_decay: float = 0.0001
    warmup_steps: int = 500
    peak_lr: float = 0.002
    min_lr: float = 0.00001
    total_training_steps: int = 102400
    batch_size: int = 512
    
    # Model
    residual_blocks: int = 8
    filters: int = 128
    
    # MCTS & Self-Play
    mcts_simulations: int = 128
    buffer_size: int = 4200000 
    past_selfplay_ratio: float = 0.25
    iterations: int = 100
    parallel_games: int = 1024
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()
"""

encoding_py = """
import numpy as np
from dapan.types import EncodingInput, ATTACKER, DEFENDER

def encode_state(enc_input: EncodingInput) -> np.ndarray:
    x = np.zeros((43, 9, 9), dtype=np.float32)
    turn = enc_input.side_to_move
    
    # For now, without full 8-step history from game, encode step 0 (most recent) in 0-4
    # We will just fill plane 0-4
    white_pos = enc_input.white_positions
    black_pos = enc_input.black_positions
    king_pos = enc_input.king_position
    
    if turn == ATTACKER:
        for r, c in black_pos: x[0, r, c] = 1.0 # Friendly
        for r, c in white_pos: x[1, r, c] = 1.0 # Enemy
    else:
        for r, c in white_pos: x[0, r, c] = 1.0
        for r, c in black_pos: x[1, r, c] = 1.0
        
    if king_pos is not None:
        x[2, king_pos[0], king_pos[1]] = 1.0
        if turn == DEFENDER: x[0, king_pos[0], king_pos[1]] = 1.0
        else: x[1, king_pos[0], king_pos[1]] = 1.0
            
    # Aux
    if turn == ATTACKER: x[40, :, :] = 1.0
    x[41, :, :] = enc_input.move_count / 512.0
    x[42, :, :] = enc_input.half_move_clock / 100.0
    
    return x
"""

action_encoding_py = """
from dapan.types import Action
import numpy as np

def action_to_id(action: Action) -> int:
    r0, c0 = action[0]
    r1, c1 = action[1]
    
    dr = r1 - r0
    dc = c1 - c0
    
    if dr < 0 and dc == 0:
        direction = 0 # N
        distance = -dr
    elif dr > 0 and dc == 0:
        direction = 1 # S
        distance = dr
    elif dr == 0 and dc > 0:
        direction = 2 # E
        distance = dc
    elif dr == 0 and dc < 0:
        direction = 3 # W
        distance = -dc
    else:
        direction = 0; distance = 1
        
    dir_idx = direction * 8 + (distance - 1)
    sq_idx = r0 * 9 + c0
    return sq_idx * 32 + dir_idx

def id_to_action(action_id: int):
    sq_idx = action_id // 32
    dir_idx = action_id % 32
    
    r0 = sq_idx // 9
    c0 = sq_idx % 9
    
    direction = dir_idx // 8
    distance = (dir_idx % 8) + 1
    
    if direction == 0: dr = -distance; dc = 0
    elif direction == 1: dr = distance; dc = 0
    elif direction == 2: dr = 0; dc = distance
    elif direction == 3: dr = 0; dc = -distance
    
    return ((r0, c0), (r0 + dr, c0 + dc))

def get_legal_mask(valid_moves) -> np.ndarray:
    mask = np.zeros(2592, dtype=np.float32)
    for move in valid_moves:
        try:
            mask[action_to_id(move)] = 1.0
        except Exception:
            pass
    return mask
"""

model_py = """
import torch
import torch.nn as torch_nn
import torch.nn.functional as F
from dapan.config import config

class ConvBlock(torch_nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch_nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = torch_nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ResBlock(torch_nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = torch_nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = torch_nn.BatchNorm2d(channels)
        self.conv2 = torch_nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = torch_nn.BatchNorm2d(channels)
    def forward(self, x):
        res = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res
        return F.relu(out)

class TablutNet(torch_nn.Module):
    def __init__(self):
        super().__init__()
        self.in_conv = ConvBlock(43, config.filters)
        self.res_blocks = torch_nn.ModuleList([ResBlock(config.filters) for _ in range(config.residual_blocks)])
        
        # Dual Heads
        # Attacker Policy Head
        self.pol_conv_a = torch_nn.Conv2d(config.filters, 2, 1)
        self.pol_bn_a = torch_nn.BatchNorm2d(2)
        self.pol_fc_a = torch_nn.Linear(2 * 9 * 9, 2592)
        
        # Attacker Value Head
        self.val_conv_a = torch_nn.Conv2d(config.filters, 1, 1)
        self.val_bn_a = torch_nn.BatchNorm2d(1)
        self.val_fc1_a = torch_nn.Linear(9 * 9, 256)
        self.val_fc2_a = torch_nn.Linear(256, 1)
        
        # Defender Policy Head
        self.pol_conv_d = torch_nn.Conv2d(config.filters, 2, 1)
        self.pol_bn_d = torch_nn.BatchNorm2d(2)
        self.pol_fc_d = torch_nn.Linear(2 * 9 * 9, 2592)
        
        # Defender Value Head
        self.val_conv_d = torch_nn.Conv2d(config.filters, 1, 1)
        self.val_bn_d = torch_nn.BatchNorm2d(1)
        self.val_fc1_d = torch_nn.Linear(9 * 9, 256)
        self.val_fc2_d = torch_nn.Linear(256, 1)

    def forward(self, x, players):
        # x is [B, 43, 9, 9]
        # players is [B] containing 1 (attacker) or -1 (defender)
        x = self.in_conv(x)
        for block in self.res_blocks:
            x = block(x)
            
        pol_a = self.pol_fc_a(F.relu(self.pol_bn_a(self.pol_conv_a(x))).view(-1, 2*81))
        val_a_h = F.relu(self.val_fc1_a(F.relu(self.val_bn_a(self.val_conv_a(x))).view(-1, 81)))
        val_a = torch.tanh(self.val_fc2_a(val_a_h))
        
        pol_d = self.pol_fc_d(F.relu(self.pol_bn_d(self.pol_conv_d(x))).view(-1, 2*81))
        val_d_h = F.relu(self.val_fc1_d(F.relu(self.val_bn_d(self.val_conv_d(x))).view(-1, 81)))
        val_d = torch.tanh(self.val_fc2_d(val_d_h))
        
        # Multiplexer based on player
        player_mask_a = (players == 1).float().unsqueeze(1)
        player_mask_d = (players == -1).float().unsqueeze(1)
        
        pol = pol_a * player_mask_a + pol_d * player_mask_d
        val = val_a * player_mask_a + val_d * player_mask_d
        
        return pol, val
"""

replay_buffer_py = """
from collections import deque
import random
from dapan.types import TrainingSample
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, sample: TrainingSample):
        self.buffer.append(sample)
        
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states = np.stack([b.state for b in batch])
        pis = np.stack([b.pi for b in batch])
        zs = np.array([b.z for b in batch], dtype=np.float32)
        players = np.array([b.player for b in batch], dtype=np.float32)
        return states, pis, zs, players
    
    def __len__(self):
        return len(self.buffer)
"""

trainer_py = """
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from dapan.config import config

class Trainer:
    def __init__(self, model):
        self.model = model.to(config.device)
        self.optimizer = AdamW(self.model.parameters(), lr=config.peak_lr, weight_decay=config.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.total_training_steps, eta_min=config.min_lr)
        
    def train_step(self, states, pis, zs, players):
        self.model.train()
        
        states = torch.tensor(states, dtype=torch.float32).to(config.device)
        pis = torch.tensor(pis, dtype=torch.float32).to(config.device)
        zs = torch.tensor(zs, dtype=torch.float32).to(config.device).unsqueeze(1)
        players = torch.tensor(players, dtype=torch.float32).to(config.device)
        
        # C4 Augmentation can be applied here
        
        pol_logits, val_pred = self.model(states, players)
        
        # Loss
        loss_val = F.mse_loss(val_pred, zs)
        
        # pi is probability distribution, pol_logits is raw logits
        # use cross entropy: H(pi, pol) = - sum pi * log_softmax(pol)
        log_probs = F.log_softmax(pol_logits, dim=-1)
        loss_pol = -torch.sum(pis * log_probs) / pis.size(0)
        
        loss = loss_val + loss_pol
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
"""

checkpointing_py = """
import torch
import os

def save_checkpoint(model, optimizer, scheduler, filename):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None
    }, filename)

def load_checkpoint(model, filename, optimizer=None, scheduler=None):
    if os.path.exists(filename):
        cp = torch.load(filename, map_location='cpu')
        model.load_state_dict(cp['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in cp:
            optimizer.load_state_dict(cp['optimizer_state_dict'])
        if scheduler and cp.get('scheduler_state_dict'):
            scheduler.load_state_dict(cp['scheduler_state_dict'])
        return True
    return False
"""

mock_components_py = """
from dapan.types import State, Action, EncodingInput, ATTACKER, DEFENDER, Winner, Player
from typing import Literal
import random
import numpy as np

class MockGame:
    def __init__(self):
        pass
    def get_initial_state(self) -> State:
        return {'side_to_move': ATTACKER, 'move_count': 0, 'half_move_clock': 0, 'terminal': False, 'white_pos': [], 'black_pos': [(0,0)], 'king_pos': (4,4)}
    def get_valid_moves(self, state: State) -> list[Action]:
        return [((0,0), (0,1)), ((4,4), (4,5))]
    def get_next_state(self, state: State, action: Action) -> State:
        return {'side_to_move': -state['side_to_move'], 'move_count': state['move_count']+1, 'half_move_clock': state['half_move_clock']+1, 'terminal': random.random() < 0.1, 'white_pos': [], 'black_pos': [(0,0)], 'king_pos': (4,4)}
    def is_terminal(self, state: State) -> bool:
        return state['terminal'] or state['move_count'] > 20
    def get_winner(self, state: State) -> Winner:
        if not self.is_terminal(state): return 0
        return random.choice([ATTACKER, DEFENDER, 0])
    def state_to_encoding_input(self, state: State) -> EncodingInput:
        return EncodingInput(
            white_positions=state['white_pos'],
            black_positions=state['black_pos'],
            king_position=state['king_pos'],
            side_to_move=state['side_to_move'],
            move_count=state['move_count'],
            half_move_clock=state['half_move_clock'],
            position_history=[]
        )

def get_action_probs(state: State, net, temperature: float) -> np.ndarray:
    game = MockGame()
    moves = game.get_valid_moves(state)
    probs = np.zeros(2592, dtype=np.float32)
    from dapan.action_encoding import action_to_id
    if moves:
        for m in moves: probs[action_to_id(m)] = 1.0 / len(moves)
    else:
        probs[0] = 1.0
    return probs
"""

selfplay_py = """
from dapan.types import TrainingSample, ATTACKER, DEFENDER
from dapan.encoding import encode_state
import numpy as np

def run_episode(game, net, get_action_probs_fn):
    history = []
    state = game.get_initial_state()
    
    while not game.is_terminal(state):
        player = state['side_to_move']
        enc_input = game.state_to_encoding_input(state)
        arr_state = encode_state(enc_input)
        
        pi = get_action_probs_fn(state, net, 1.0)
        
        valid_moves = game.get_valid_moves(state)
        from dapan.action_encoding import action_to_id
        if not valid_moves:
            break
            
        move = valid_moves[np.argmax([pi[action_to_id(m)] for m in valid_moves])] # simple greedy for now in mock
        history.append((arr_state, pi, player, state))
        
        state = game.get_next_state(state, move)
        
    winner = game.get_winner(state)
    samples = []
    
    for arr_state, pi, player, _ in history:
        z = 0.0
        if winner != 0:
            if player == winner: z = 1.0
            else: z = -1.0
        samples.append(TrainingSample(state=arr_state, pi=pi, z=z, player=player))
        
    return samples
"""

train_loop_py = """
from dapan.mock_components import MockGame, get_action_probs
from dapan.model import TablutNet
from dapan.trainer import Trainer
from dapan.replay_buffer import ReplayBuffer
from dapan.selfplay import run_episode
from dapan.checkpointing import save_checkpoint, load_checkpoint
import os
import torch

def main():
    print("Initializing components...")
    model = TablutNet()
    trainer = Trainer(model)
    buffer = ReplayBuffer(capacity=1000)
    game = MockGame()
    
    # Test checkpoint loading
    ckpt_path = "test_cp.pt"
    load_checkpoint(model, ckpt_path, trainer.optimizer, trainer.scheduler)
    
    print("Running Self-Play iterations...")
    for i in range(2):
        samples = run_episode(game, model, get_action_probs)
        for s in samples:
            buffer.add(s)
            
        print(f"Iteration {i+1}: generated {len(samples)} samples. Buffer size: {len(buffer)}")
        
        if len(buffer) >= 4:
            states, pis, zs, players = buffer.sample(4)
            loss = trainer.train_step(states, pis, zs, players)
            print(f"Training step loss: {loss:.4f}")
            
    print("Saving checkpoint...")
    save_checkpoint(model, trainer.optimizer, trainer.scheduler, ckpt_path)
    
    # Test valid loading
    success = load_checkpoint(model, ckpt_path)
    print(f"Checkpoint reload successful: {success}")
    print("End-to-End Pipeline test passed!")

if __name__ == "__main__":
    main()
"""

files = {
    'dapan/types.py': types_py,
    'dapan/config.py': config_py,
    'dapan/encoding.py': encoding_py,
    'dapan/action_encoding.py': action_encoding_py,
    'dapan/model.py': model_py,
    'dapan/replay_buffer.py': replay_buffer_py,
    'dapan/trainer.py': trainer_py,
    'dapan/checkpointing.py': checkpointing_py,
    'dapan/mock_components.py': mock_components_py,
    'dapan/selfplay.py': selfplay_py,
    'dapan/train_loop.py': train_loop_py
}

for fp, contents in files.items():
    with open(fp, 'w') as f:
        f.write(contents.strip() + '\n')
