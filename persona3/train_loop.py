from persona3.mock_components import MockGame, get_action_probs
from persona3.model import TablutNet
from persona3.trainer import Trainer
from persona3.replay_buffer import ReplayBuffer
from persona3.selfplay import run_episode
from persona3.checkpointing import save_checkpoint, load_checkpoint
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
