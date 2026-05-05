import torch
import torch.nn as torch_nn
import torch.nn.functional as F
from persona3.config import config

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
