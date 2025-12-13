"""
Main script to train AlphaZero on Knight Capture game.
"""
import torch
import argparse
from model import AlphaZeroNet
from train import AlphaZeroTrainer
from config import Config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train AlphaZero on Knight Capture game')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint to resume training from')
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    config = Config()
    
    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = AlphaZeroNet(
        board_size=config.board_size,
        num_channels=config.num_channels,
        num_residual_blocks=config.num_residual_blocks
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = AlphaZeroTrainer(
        model=model,
        num_simulations=config.num_simulations,
        num_games=config.num_games,
        num_iterations=config.num_iterations,
        num_epochs=config.num_epochs,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        c_puct=config.c_puct,
        temperature=config.temperature,
        device=str(device),
        save_dir=config.save_dir,
        mcts_batch_size=config.mcts_batch_size
    )
    
    # Set max game length from config
    trainer.max_game_length = config.max_game_length
    
    # Update replay buffer size
    from train import ReplayBuffer
    trainer.replay_buffer = ReplayBuffer(max_size=config.buffer_size)
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Start training
    trainer.train()
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()
