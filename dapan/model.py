try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = object

from config import ACTION_SIZE, NUM_CHANNELS


class TablutNet(nn.Module if torch is not None else object):
    def __init__(self, channels: int = NUM_CHANNELS, action_size: int = ACTION_SIZE):
        if torch is None:
            raise RuntimeError('PyTorch non installato.')
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 8, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 9 * 9, action_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(64, 4, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 9 * 9, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        h = self.body(x)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value