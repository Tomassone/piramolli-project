import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from persona3.config import config

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
