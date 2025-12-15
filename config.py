"""
Configuration file for AlphaZero training.
"""
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for AlphaZero training."""
    # Game settings
    board_size: int = 5
    
    # Model settings
    num_channels: int = 128  # Increased for better capacity
    num_residual_blocks: int = 2   # Deeper network
    
    # MCTS settings
    num_simulations: int = 100  # More simulations for better play
    c_puct: float = 2.0  # Increased for more exploration
    mcts_batch_size: int = 1024  # Batch size for MCTS neural network evaluations (increased for better GPU utilization)
    max_game_length: int = 1000  # Maximum moves per game (games longer than this end in draw)
    
    # Training settings
    num_iterations: int = 1000
    num_games: int = 200  # More games per iteration
    num_epochs: int = 10  # Single pass to prevent overfitting
    batch_size: int = 1024  # Large batch size for GPU utilization (increased to use more VRAM)
    learning_rate: float = 0.001
    weight_decay: float = 1e-4  # L2 regularization to prevent overfitting
    temperature: float = 1.5  # Increased temperature for more exploration during self-play
    num_workers: int = 8 # reallel workers for game generation
    
    # Replay buffer settings
    buffer_size: int = 500000  # Larger buffer
    
    # Device settings
    device: str = 'cuda'  # 'cuda' or 'cpu'
    
    # Checkpoint settings
    save_dir: str = 'checkpoints'
    save_interval: int = 10

