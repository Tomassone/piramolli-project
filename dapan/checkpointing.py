from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model, optimizer, step: int) -> str:
        if torch is None:
            raise RuntimeError('PyTorch non installato.')
        path = self.checkpoint_dir / f'checkpoint_step_{step}.pt'
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        return str(path)
