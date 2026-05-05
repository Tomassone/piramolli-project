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
