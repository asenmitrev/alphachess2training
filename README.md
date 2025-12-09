# AlphaZero for TicTacToe

This project implements AlphaZero algorithm to train a neural network to play TicTacToe on a 3x3 board.

## Overview

AlphaZero is a reinforcement learning algorithm that combines:

- **Monte Carlo Tree Search (MCTS)**: For game tree exploration
- **Deep Neural Networks**: For position evaluation and move prediction
- **Self-Play**: For generating training data

## Project Structure

```
alpatoe/
├── game.py          # TicTacToe game implementation
├── model.py         # Neural network architecture
├── mcts.py          # Monte Carlo Tree Search implementation
├── train.py         # Training loop and self-play logic
├── config.py        # Configuration settings
├── main.py          # Main training script
├── requirements.txt # Python dependencies
└── README.md        # This file
```

## Installation

1. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Verify installation:**
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

## Usage

### Training

To start training AlphaZero:

```bash
python main.py
```

The training process will:

1. Generate self-play games using MCTS
2. Store game states in a replay buffer
3. Train the neural network on the collected data
4. Save checkpoints every 10 iterations

### Configuration

You can modify training parameters in `config.py`:

- `num_simulations`: Number of MCTS simulations per move (default: 100)
- `num_games`: Number of self-play games per iteration (default: 100)
- `num_iterations`: Total number of training iterations (default: 100)
- `num_epochs`: Training epochs per iteration (default: 10)
- `batch_size`: Batch size for training (default: 32)
- `learning_rate`: Learning rate (default: 0.001)
- `device`: 'cuda' or 'cpu' (default: 'cuda')

### Playing Against the AI

After training, you can play against the trained model:

```bash
# Play against the latest checkpoint (you play as X, go first)
python play.py

# Play against a specific checkpoint
python play.py --checkpoint checkpoints/model_iter_50.pth

# Play as O (AI goes first)
python play.py --play-o

# Use more MCTS simulations for stronger play
python play.py --simulations 400

# Use CPU instead of GPU
python play.py --device cpu
```

**Game Controls:**

- Enter moves as `row col` (e.g., `1 1` for center)
- Rows and columns are 0-indexed (0, 1, 2)
- Press Ctrl+C to cancel a game

## Testing the Game

You can test the game implementation:

```python
from game import TicTacToe

game = TicTacToe()
print(game)
game.make_move(0, 0)  # X plays at top-left
print(game)
game.make_move(1, 1)  # O plays at center
print(game)
```

## How It Works

1. **Self-Play**: The current model plays against itself, using MCTS to select moves
2. **Data Collection**: Each game state, along with the MCTS policy and final outcome, is stored
3. **Training**: The neural network learns to predict:
   - **Policy**: Probability distribution over moves
   - **Value**: Expected game outcome from current position
4. **Iteration**: The improved model is used for the next round of self-play

## Model Architecture

The neural network consists of:

- **Input**: 1 channel representing the canonical board state
- **Convolutional layers**: Extract spatial features
- **Residual blocks**: Deep feature extraction
- **Policy head**: Outputs move probabilities
- **Value head**: Outputs position evaluation

## Checkpoints

Model checkpoints are saved in the `checkpoints/` directory. Each checkpoint contains:

- Model weights
- Optimizer state
- Training iteration number

## Notes

- Training on CPU is slower but works fine for TicTacToe
- For faster training, use GPU by setting `device='cuda'` in config.py
- TicTacToe is a simple game, so training should converge relatively quickly
- The model learns optimal play through self-play without any human knowledge

## License

This project is provided as-is for educational purposes.
