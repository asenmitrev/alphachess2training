"""
Configuration file for AlphaZero training.
"""
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for AlphaZero training."""
    # Game settings
    board_size: int = 3
    
    # Model settings
    num_channels: int = 64  # Increased for better capacity
    num_residual_blocks: int = 4  # Deeper network
    
    # MCTS settings
    num_simulations: int = 200  # More simulations for better play
    c_puct: float = 1.0
    mcts_batch_size: int = 32  # Batch size for MCTS neural network evaluations
    
    # Training settings
    num_iterations: int = 100
    num_games: int = 200  # More games per iteration
    num_epochs: int = 20  # More training epochs
    batch_size: int = 64  # Large batch size for GPU utilization
    learning_rate: float = 0.002
    temperature: float = 1.0
    num_workers: int = 8  # Parallel workers for game generation
    
    # Replay buffer settings
    buffer_size: int = 50000  # Larger buffer
    
    # Device settings
    device: str = 'cuda'  # 'cuda' or 'cpu'
    
    # Checkpoint settings
    save_dir: str = 'checkpoints'
    save_interval: int = 10

