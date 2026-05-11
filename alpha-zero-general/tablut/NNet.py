import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from NeuralNet import NeuralNet


class TablutNet(nn.Module):
    """
    PyTorch neural network for Tablut.
    
    Architecture:
    - Input: 9x9x28 board representation (8 history frames × 3 channels + global info)
    - Shared feature extraction with residual blocks
    - Separate policy head (same for both players in canonical form)
    - Independent value heads for White and Black (asymmetric game)
    """
    
    def __init__(self, board_height=9, board_width=9, num_channels=28, action_size=6561):
        super(TablutNet, self).__init__()
        self.board_height = board_height
        self.board_width = board_width
        self.num_channels = num_channels
        self.action_size = action_size
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(num_channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        
        # Residual blocks for shared feature extraction
        self.res_blocks = nn.ModuleList([
            self._make_residual_block(128, 128) for _ in range(3)
        ])
        
        # ─── Policy Head ───
        self.policy_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * board_height * board_width, action_size)
        
        # ─── White Value Head (defender/king) ───
        self.value_white_conv = nn.Conv2d(128, 3, kernel_size=1)
        self.value_white_bn = nn.BatchNorm2d(3)
        self.value_white_fc1 = nn.Linear(3 * board_height * board_width, 64)
        self.value_white_fc2 = nn.Linear(64, 1)
        
        # ─── Black Value Head (attacker) ───
        # Independent layers to capture asymmetric game dynamics
        self.value_black_conv = nn.Conv2d(128, 3, kernel_size=1)
        self.value_black_bn = nn.BatchNorm2d(3)
        self.value_black_fc1 = nn.Linear(3 * board_height * board_width, 64)
        self.value_black_fc2 = nn.Linear(64, 1)
    
    def _make_residual_block(self, in_channels, out_channels):
        """Create a residual block with conv layers and batch norm."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: Tensor of shape (batch_size, num_channels, board_height, board_width)
        
        Returns:
            Tuple of (policy_white, value_white, policy_black, value_black)
        """
        # ─── Shared Feature Extraction ───
        x = self.conv1(state)
        x = self.bn1(x)
        x = torch.relu(x)
        
        # Residual blocks
        for res_block in self.res_blocks:
            residual = x
            x = res_block(x)
            x = residual + x
            x = torch.relu(x)
        
        # ─── Policy Head (shared, canonical form) ───
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = torch.relu(policy)
        policy = policy.reshape(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy_logits = policy
        # Non serve softmax qui, lo applichiamo durante la previsione per stabilità numerica
        #policy_white = torch.softmax(policy_logits, dim=1)
        #policy_black = torch.softmax(policy_logits, dim=1)
        
        # ─── White Value Head ───
        value_w = self.value_white_conv(x)
        value_w = self.value_white_bn(value_w)
        value_w = torch.relu(value_w)
        value_w = value_w.reshape(value_w.size(0), -1)
        value_w = torch.relu(self.value_white_fc1(value_w))
        value_w = torch.tanh(self.value_white_fc2(value_w))
        
        # ─── Black Value Head (independent) ───
        value_b = self.value_black_conv(x)
        value_b = self.value_black_bn(value_b)
        value_b = torch.relu(value_b)
        value_b = value_b.reshape(value_b.size(0), -1)
        value_b = torch.relu(self.value_black_fc1(value_b))
        value_b = torch.tanh(self.value_black_fc2(value_b))
        
        return policy_white, value_w, policy_black, value_b


class NNetWrapper(NeuralNet):
    """
    Neural network wrapper for Tablut game.
    
    This wrapper handles:
    - Model initialization and management
    - Training on examples from self-play
    - Prediction for MCTS search
    - Checkpoint persistence
    
    The network is trained from White's canonical perspective, but maintains
    separate value estimates for each side due to the asymmetric nature of
    Tablut (White = defender/king, Black = attacker).
    """

    def __init__(self, game):
        self.game = game
        self.board_height, self.board_width = game.getBoardSize()
        self.action_size = game.getActionSize()
        
        # Initialize model
        self.model = TablutNet(
            board_height=self.board_height,
            board_width=self.board_width,
            num_channels=28,  # From TablutGame.encode_state: 8 history × 3 + 4 info planes
            action_size=self.action_size
        )
        
        # Use CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Optimizer and loss functions
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()

    def train(self, examples):
        """
        Train the neural network on examples from self-play.
        
        Args:
            examples: List of tuples (board, pi, v) where:
                - board: numpy array of board state (canonical form, 9x9x28)
                - pi: numpy array of policy vector (length action_size)
                - v: float, value of the board (-1 to 1)
        """
        if not examples:
            return 0.0
        
        self.model.train()
        
        # Convert examples to tensors
        boards = []
        pis = []
        vs = []
        
        for board, pi, v in examples:
            # board is shape (9, 9, 28) from encode_state
            boards.append(torch.FloatTensor(board).to(self.device))
            pis.append(torch.FloatTensor(pi).to(self.device))
            vs.append(torch.FloatTensor([v]).to(self.device))
        
        # Stack into batches
        boards = torch.stack(boards)  # (batch, 9, 9, 28)
        pis = torch.stack(pis)        # (batch, action_size)
        vs = torch.stack(vs)          # (batch, 1)
        
        # Permute to (batch, channels, height, width) for Conv2d
        boards = boards.permute(0, 3, 1, 2)
        
        # Forward pass
        policy_w, value_w, policy_b, value_b = self.model(boards)
        
        # Compute losses from white's canonical perspective
        # The board is encoded from white's perspective, so we train on white's values
        policy_loss = self.policy_loss_fn(policy_w, pis)
        value_loss = self.value_loss_fn(value_w, vs)
        
        # Combined loss
        total_loss = policy_loss + value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()

    def predict(self, board):
        """
        Predict policy and value for a given board state.
        
        Args:
            board: numpy array of board state (canonical form, shape 9x9x28)
        
        Returns:
            pi: numpy array of policy probabilities (shape action_size)
            v: float, value of the board from white's perspective (-1 to 1)
        """
        self.model.eval()
        
        with torch.no_grad():
            # Convert board to tensor and add batch dimension
            board_tensor = torch.FloatTensor(board).to(self.device)
            board_tensor = board_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 28, 9, 9)
            
            # Forward pass
            policy_w, value_w, policy_b, value_b = self.model(board_tensor)
            
            # Return white's perspective (canonical form is from white's view)
            pi = policy_w[0].cpu().numpy()
            v = value_w[0].item()
        
        return pi, v

    def save_checkpoint(self, folder, filename):
        """
        Save the neural network checkpoint.
        
        Args:
            folder: Directory to save the checkpoint
            filename: Name of the checkpoint file
        """
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        filepath = os.path.join(folder, filename)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, folder, filename):
        """
        Load a neural network checkpoint.
        
        Args:
            folder: Directory containing the checkpoint
            filename: Name of the checkpoint file
        """
        filepath = os.path.join(folder, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {filepath}")
