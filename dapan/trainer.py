try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    F = None


class Trainer:
    def __init__(self, model, learning_rate: float, weight_decay: float, device: str = 'cpu'):
        if torch is None:
            raise RuntimeError('PyTorch non installato.')
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.device = device

    def train_step(self, states, target_pi, target_z):
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        target_pi = torch.tensor(target_pi, dtype=torch.float32, device=self.device)
        target_z = torch.tensor(target_z, dtype=torch.float32, device=self.device)

        self.model.train()
        pred_logits, pred_value = self.model(states)
        log_probs = F.log_softmax(pred_logits, dim=1)
        policy_loss = -(target_pi * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(pred_value, target_z)
        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'loss': float(loss.detach().cpu()),
            'policy_loss': float(policy_loss.detach().cpu()),
            'value_loss': float(value_loss.detach().cpu()),
        }
