from pathlib import Path
import json

from checkpointing import CheckpointManager
from config import TrainConfig
from mock_components import MockGame, MockMCTS, MockSelfPlay
from model import TablutNet, torch
from replay_buffer import ReplayBuffer
from trainer import Trainer


def main():
    cfg = TrainConfig()
    game = MockGame(seed=7)
    mcts = MockMCTS(seed=11, legal_actions_per_state=10)
    selfplay = MockSelfPlay(game, mcts, seed=19)
    buffer = ReplayBuffer(capacity=cfg.replay_buffer_capacity)

    samples = selfplay.make_dataset(cfg.num_mock_samples)
    buffer.extend(samples)

    states, pis, zs = buffer.sample_numpy_batch(min(cfg.batch_size, len(buffer)))
    summary = {
        'buffer_size': len(buffer),
        'states_shape': tuple(states.shape),
        'pis_shape': tuple(pis.shape),
        'zs_shape': tuple(zs.shape),
        'pi_sum_example': float(pis[0].sum()),
        'z_examples': [float(z) for z in zs[:8, 0]],
        'torch_available': torch is not None,
    }

    base_dir = Path.cwd()
    artifacts_dir = base_dir / 'artifacts'
    artifacts_dir.mkdir(exist_ok=True)
    with open(artifacts_dir / 'data_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    if torch is None:
        print('PyTorch non disponibile: dataset e buffer creati correttamente.')
        print(json.dumps(summary, indent=2))
        return

    model = TablutNet()
    trainer = Trainer(model, cfg.learning_rate, cfg.weight_decay, cfg.device)
    ckpt = CheckpointManager(str(base_dir / cfg.checkpoint_dir))

    history = []
    step = 0
    for epoch in range(cfg.epochs):
        for _ in range(cfg.steps_per_epoch):
            states, pis, zs = buffer.sample_numpy_batch(cfg.batch_size)
            metrics = trainer.train_step(states, pis, zs)
            step += 1
            history.append({'step': step, 'epoch': epoch, **metrics})
        ckpt.save(model, trainer.optimizer, step)

    with open(artifacts_dir / 'train_history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    print('Training completato.')
    print(json.dumps(history[-1], indent=2))


if __name__ == '__main__':
    main()
