"""
Training loop for AlphaZero.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
from collections import deque
import random
from typing import List, Tuple, Dict, Any
from game import KingCapture, Player
from model import AlphaZeroNet
from mcts import MCTS
import os
import time
import logging
import queue

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def run_batched_games(model: AlphaZeroNet, num_games: int, num_simulations: int, 
                     c_puct: float, temperature: float, device: str,
                     max_game_length: int = 400) -> List[Tuple]:
    """
    Run a batch of games using sequential MCTS.
    This function is used by both the main process and worker processes.
    """
    # Initialize MCTS
    mcts = MCTS(model, num_simulations, c_puct, device)
    
    examples = []
    
    # Initialize all games
    games = [KingCapture() for _ in range(num_games)]
    game_histories = [[] for _ in range(num_games)]
    active_indices = list(range(num_games))
    games_completed = 0  # Track sequential count of completed games
    
    # Loop until all games are finished
    while active_indices:
        # Get current states for active games
        current_games = [games[i] for i in active_indices]
        
        # Batched MCTS search - parallelized across all games
        policies = mcts.search_batch(current_games)
        
        # Make moves for all active games
        finished_indices = []
        
        for idx_in_batch, policy in enumerate(policies):
            game_idx = active_indices[idx_in_batch]
            game = games[game_idx]
            
            # Select action
            valid_moves = game.get_valid_moves()
            valid_actions = [game.move_to_action(piece_idx, r, c) for piece_idx, r, c in valid_moves]
            
            if temperature > 0:
                # Sample
                valid_policy = policy[valid_actions]
                valid_policy = valid_policy ** (1.0 / temperature)
                valid_policy = valid_policy / valid_policy.sum()
                action = np.random.choice(valid_actions, p=valid_policy)
            else:
                # Greedy
                action = valid_actions[np.argmax(policy[valid_actions])]
            
            # Store history
            state = game.get_canonical_state()
            
            # Flip policy to canonical view if Black
            # MCTS returns policy in real board coordinates, but we train on canonical coordinates
            store_policy = policy
            if game.current_player == Player.BLACK:
                store_policy = game.flip_policy(policy)
            
            game_histories[game_idx].append((state, store_policy))
            
            # Apply move
            piece_idx, row, col = game.action_to_move(action)
            game.make_move(piece_idx, row, col)
            
            # Check move limit
            max_moves_reached = False
            if len(game.move_history) >= max_game_length:
                game.game_over = True
                game.winner = None  # Draw
                max_moves_reached = True
            
            if game.game_over:
                finished_indices.append(idx_in_batch)
                
                # Process result
                result = game.get_result(Player.WHITE)
                # If max moves reached, mark as loss for both players
                if max_moves_reached:
                    result = -1.0  # Loss for both players
                elif result is None:
                    result = 0.0
                
                # Log game completion
                games_completed += 1
                num_moves = len(game.move_history)
                winner = game.winner.name if game.winner else "Draw"
                remaining_games = len(active_indices) - len(finished_indices)
                logging.info(f"Game {games_completed}/{num_games} finished: {winner} won after {num_moves} moves ({remaining_games} games remaining)")
                
                # Store examples
                for i, (hist_state, hist_policy) in enumerate(game_histories[game_idx]):
                    # If max moves reached, both players get loss (-1.0)
                    if max_moves_reached:
                        value = -1.0  # Loss for both players
                    else:
                        # Value alternates based on whose turn it was
                        # Even indices: White's turn, Odd indices: Black's turn
                        if i % 2 == 0:
                            value = result  # White's perspective
                        else:
                            value = -result  # Black's perspective (negate for zero-sum)
                    
                    examples.append((hist_state, hist_policy, value))
        
        # Remove finished games from active list (in reverse order to keep indices valid)
        for idx_in_batch in sorted(finished_indices, reverse=True):
            active_indices.pop(idx_in_batch)
            
    return examples


def worker_self_play_queue(rank: int, model_state: Dict, config: Dict, work_queue: mp.Queue, result_queue: mp.Queue):
    """
    Worker function for parallel self-play with work-stealing queue.
    Workers pull games from the shared queue until it's empty.
    """
    # Configure logging for worker process
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    
    # Set random seed for this worker to ensure diversity
    seed = int(time.time()) + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Reconstruct model
    device = config['device']
    model = AlphaZeroNet(
        board_size=config['board_size'],
        num_channels=config['num_channels'],
        num_residual_blocks=config['num_residual_blocks']
    )
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    
    # Initialize MCTS
    mcts = MCTS(model, config['num_simulations'], config['c_puct'], device)
    
    all_examples = []
    games_processed = 0
    
    # Pull games from queue until empty
    while True:
        try:
            # Try to get a game from the queue (non-blocking with timeout)
            try:
                game_id = work_queue.get(timeout=0.1)
            except queue.Empty:
                # Queue is empty, we're done
                break
            
            # Process this single game
            try:
                examples, game_info = _run_single_game(
                    model=model,
                    mcts=mcts,
                    num_simulations=config['num_simulations'],
                    c_puct=config['c_puct'],
                    temperature=config['temperature'],
                    max_game_length=config.get('max_game_length', 400)
                )
                
                all_examples.extend(examples)
                games_processed += 1
                
                # Log game completion
                remaining = work_queue.qsize()
                winner = game_info['winner']
                num_moves = game_info['num_moves']
                logging.info(f"Worker {rank}: Completed {games_processed} games (game ID {game_id + 1}) - {winner} won after {num_moves} moves ({remaining} games remaining in queue)")
            except Exception as e:
                logging.error(f"Worker {rank} error processing game {game_id}: {e}", exc_info=True)
            finally:
                # Mark task as done even if processing failed
                work_queue.task_done()
            
        except KeyboardInterrupt:
            logging.info(f"Worker {rank} interrupted")
            break
        except Exception as e:
            logging.error(f"Worker {rank} unexpected error: {e}", exc_info=True)
            break
    
    # Send results back
    result_queue.put((rank, all_examples))


def _run_single_game(model: AlphaZeroNet, mcts: MCTS, num_simulations: int, 
                     c_puct: float, temperature: float, max_game_length: int) -> Tuple[List[Tuple], Dict]:
    """
    Run a single game and return training examples and game info.
    
    Returns:
        examples: List of (state, policy, value) tuples
        game_info: Dict with 'winner', 'num_moves', 'result'
    """
    game = KingCapture()
    game_history = []
    examples = []
    
    # Play until game ends
    max_moves_reached = False
    while not game.game_over:
        # Check move limit
        if len(game.move_history) >= max_game_length:
            game.game_over = True
            game.winner = None  # Draw
            max_moves_reached = True
            break
        # MCTS search for this game state
        policy = mcts.search(game)
        
        # Select action
        valid_moves = game.get_valid_moves()
        valid_actions = [game.move_to_action(piece_idx, r, c) for piece_idx, r, c in valid_moves]
        
        if temperature > 0:
            # Sample
            valid_policy = policy[valid_actions]
            valid_policy = valid_policy ** (1.0 / temperature)
            valid_policy = valid_policy / valid_policy.sum()
            action = np.random.choice(valid_actions, p=valid_policy)
        else:
            # Greedy
            action = valid_actions[np.argmax(policy[valid_actions])]
        # Store history
        state = game.get_canonical_state()
        
        # Flip policy to canonical view if Black
        store_policy = policy
        if game.current_player == Player.BLACK:
            store_policy = game.flip_policy(policy)
        game_history.append((state, store_policy))
        
        # Apply move
        piece_idx, row, col = game.action_to_move(action)
        game.make_move(piece_idx, row, col)
    
    # Process result
    result = game.get_result(Player.WHITE)
    # If max moves reached, mark as loss for both players
    if max_moves_reached:
        result = -1.0  # Loss for both players
    elif result is None:
        result = 0.0
    
    # Store examples
    for i, (hist_state, hist_policy) in enumerate(game_history):
        # If max moves reached, both players get loss (-1.0)
        if max_moves_reached:
            value = -1.0  # Loss for both players
        else:
            # Value alternates based on whose turn it was
            # Even indices: White's turn, Odd indices: Black's turn
            if i % 2 == 0:
                value = result  # White's perspective
            else:
                value = -result  # Black's perspective (negate for zero-sum)
        
        examples.append((hist_state, hist_policy, value))
    
    # Prepare game info
    winner = game.winner.name if game.winner else "Draw"
    num_moves = len(game.move_history)
    game_info = {
        'winner': winner,
        'num_moves': num_moves,
        'result': result
    }
    
    return examples, game_info


def worker_self_play(rank: int, model_state: Dict, config: Dict, num_games: int) -> List[Tuple]:
    """
    Worker function for parallel self-play (legacy static assignment).
    Kept for backward compatibility but not used with work-stealing.
    """
    # Configure logging for worker process
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    
    # Set random seed for this worker to ensure diversity
    seed = int(time.time()) + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Reconstruct model
    device = config['device']
    model = AlphaZeroNet(
        board_size=config['board_size'],
        num_channels=config['num_channels'],
        num_residual_blocks=config['num_residual_blocks']
    )
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    
    # Run games
    return run_batched_games(
        model=model,
        num_games=num_games,
        num_simulations=config['num_simulations'],
        c_puct=config['c_puct'],
        temperature=config['temperature'],
        device=device,
        max_game_length=config.get('max_game_length', 400)
    )


class ReplayBuffer:
    """Buffer to store self-play games."""
    
    def __init__(self, max_size: int = 50000):
        """
        Initialize replay buffer.
        
        Args:
            max_size: Maximum number of game states to store
        """
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state: np.ndarray, policy: np.ndarray, value: float):
        """
        Add a game state to the buffer.
        
        Args:
            state: Game state (canonical form)
            policy: Policy distribution
            value: Outcome value
        """
        self.buffer.append((state, policy, value))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample a batch of game states."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)


class AlphaZeroTrainer:
    """Trainer for AlphaZero."""
    
    def __init__(
        self,
        model: AlphaZeroNet,
        num_simulations: int = 100,
        num_games: int = 100,
        num_iterations: int = 100,
        num_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        c_puct: float = 1.0,
        temperature: float = 1.0,
        device: str = 'cpu',
        save_dir: str = 'checkpoints',
        num_workers: int = 0
    ):
        """
        Initialize trainer.
        
        Args:
            model: Neural network model
            num_simulations: MCTS simulations per move
            num_games: Number of self-play games per iteration
            num_iterations: Number of training iterations
            num_epochs: Number of training epochs per iteration
            batch_size: Batch size for training
            learning_rate: Learning rate
            c_puct: MCTS exploration constant
            temperature: Temperature for action selection (1.0 = full exploration, 0.0 = greedy)
            device: Device to run on
            save_dir: Directory to save checkpoints
            num_workers: Number of parallel workers for self-play
        """
        self.model = model.to(device)
        self.num_simulations = num_simulations
        self.num_games = num_games
        self.num_iterations = num_iterations
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.c_puct = c_puct
        self.temperature = temperature
        self.device = device
        self.save_dir = save_dir
        self.num_workers = num_workers if num_workers > 0 else 1
        self.max_game_length = 400  # Default, can be overridden
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize optimizer with weight decay (L2 regularization)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Store initial parameters for debugging
        self.initial_params = {name: param.data.clone() for name, param in self.model.named_parameters()}
        
        # Initialize MCTS (for single-process use if needed)
        self.mcts = MCTS(self.model, num_simulations, c_puct, device)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Start iteration
        self.start_iteration = 0
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a checkpoint to resume training.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_iteration = checkpoint['iteration']
        
        print(f"Resuming from iteration {self.start_iteration}")
    
    def self_play(self) -> List[Tuple]:
        """
        Generate self-play games using multiprocessing with work-stealing queue.
        Workers dynamically pull games from a shared queue, allowing fast workers
        to pick up remaining games from slow workers.
        """
        examples = []
        print(f"Generating {self.num_games} games using {self.num_workers} workers with work-stealing...")
        start_time = time.time()
        
        if self.num_workers <= 1:
            # Single process
            examples = run_batched_games(
                self.model, self.num_games, self.num_simulations, 
                self.c_puct, self.temperature, self.device,
                max_game_length=getattr(self, 'max_game_length', 400)
            )
        else:
            # Multiprocessing with work-stealing queue
            # We need to use 'spawn' for CUDA support
            ctx = mp.get_context('spawn')
            
            # Create shared queues for work distribution
            manager = ctx.Manager()
            work_queue = manager.Queue()
            result_queue = manager.Queue()
            
            # Put all games into the work queue
            for game_id in range(self.num_games):
                work_queue.put(game_id)
            
            # Config dict needed to reconstruct model in worker
            worker_config = {
                'board_size': self.model.board_size,
                'num_channels': self.model.num_channels,
                'num_residual_blocks': self.model.num_residual_blocks,
                'num_simulations': self.num_simulations,
                'c_puct': self.c_puct,
                'temperature': self.temperature,
                'device': self.device,
                'max_game_length': getattr(self, 'max_game_length', 400)
            }
            
            # Get current model state
            model_state = self.model.state_dict()
            
            # Launch workers
            processes = []
            for rank in range(self.num_workers):
                p = ctx.Process(
                    target=worker_self_play_queue,
                    args=(rank, model_state, worker_config, work_queue, result_queue)
                )
                p.start()
                processes.append(p)
            
            # Collect results from all workers
            worker_results = {}
            for _ in range(self.num_workers):
                rank, worker_examples = result_queue.get()
                worker_results[rank] = worker_examples
            
            # Wait for all processes to finish
            for p in processes:
                p.join()
            
            # Combine all examples
            for rank in sorted(worker_results.keys()):
                examples.extend(worker_results[rank])
                    
        elapsed = time.time() - start_time
        print(f"Generated {self.num_games} games in {elapsed:.2f}s "
              f"({self.num_games/elapsed:.1f} games/sec)")
        
        return examples
    
    def train(self):
        """Main training loop."""
        # Store a test state to track model changes
        test_state = None
        
        for iteration in range(self.start_iteration, self.num_iterations):
            print(f"\n=== Iteration {iteration + 1}/{self.num_iterations} ===")
            
            # Generate self-play games
            print("Generating self-play games...")
            examples = self.self_play()
            
            # Store a test state from first batch if available
            if test_state is None and examples:
                test_state = examples[0][0]  # First state from first example
            
            # Check model predictions on test state to track learning
            if test_state is not None:
                self.model.eval()
                with torch.no_grad():
                    test_tensor = torch.FloatTensor(test_state).unsqueeze(0).unsqueeze(0).to(self.device)
                    test_policy, test_value = self.model(test_tensor)
                    print(f"  Test state - Policy max: {test_policy.max().item():.4f}, Value: {test_value.item():.4f}")
                self.model.train()
            
            # Add to replay buffer
            for state, policy, value in examples:
                self.replay_buffer.add(state, policy, value)
            
            print(f"Replay buffer size: {self.replay_buffer.size()}")
            
            # Train on replay buffer
            if self.replay_buffer.size() >= self.batch_size:
                print("Training model...")
                self._train_epoch()
            
            # Save checkpoint
            if (iteration + 1) % 1 == 0:
                self._save_checkpoint(iteration + 1)
    
    def _train_epoch(self):
        """Train for one epoch with multiple batches."""
        self.model.train()
        
        buffer_size = self.replay_buffer.size()
        num_batches = max(1, buffer_size // self.batch_size)
        
        # Start timing for training epochs
        training_start_time = time.time()
        print(f"Starting training for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_total_loss = 0.0
            
            # Train on multiple batches per epoch
            for batch_idx in range(num_batches):
                batch = self.replay_buffer.sample(self.batch_size)
                
                states = []
                policies = []
                values = []
                
                for state, policy, value in batch:
                    states.append(state)
                    policies.append(policy)
                    values.append(value)
                
                # Convert to tensors
                states_tensor = torch.FloatTensor(np.array(states)).unsqueeze(1).to(self.device)
                policies_tensor = torch.FloatTensor(np.array(policies)).to(self.device)
                values_tensor = torch.FloatTensor(np.array(values)).unsqueeze(1).to(self.device)
                
                # Apply value target smoothing to prevent tanh saturation
                # Smooth targets towards 0: target = target * 0.8 (so ±1.0 becomes ±0.8)
                values_tensor = values_tensor * 0.8
                
                # Add noise to prevent exact memorization (std=0.1)
                values_tensor = values_tensor + torch.randn_like(values_tensor) * 0.1
                # Clamp to valid range
                values_tensor = values_tensor.clamp(-1.0, 1.0)
                
                # Forward pass
                self.optimizer.zero_grad()
                policy_logits, value_pred = self.model(states_tensor)
                
                # Compute losses
                # Policy loss: KL divergence between predicted and MCTS policy
                policy_probs = torch.softmax(policy_logits, dim=1)
                policy_loss = -torch.sum(policies_tensor * torch.log(policy_probs + 1e-8), dim=1).mean()
                
                value_loss = nn.functional.mse_loss(value_pred, values_tensor)
                
                # Debug: Print value statistics for first batch of first epoch
                if epoch == 0 and batch_idx == 0:
                    # Compute manual MSE to verify
                    manual_mse = ((value_pred - values_tensor) ** 2).mean()
                
                # Scale value loss to be comparable to policy loss (value_loss is typically much smaller)
                value_loss_scaled = value_loss * 10.0
                total_loss = policy_loss + value_loss_scaled
                
                # Debug: Check gradients before backward
                if epoch == 0 and batch_idx == 0:
                    # Get a sample parameter to check gradients
                    sample_param = list(self.model.parameters())[0]
                
                # Backward pass
                total_loss.backward()
                
                # Debug: Check gradients after backward
                if epoch == 0 and batch_idx == 0:
                    sample_param = list(self.model.parameters())[0]
                
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Debug: Check if parameters changed
                if epoch == 0 and batch_idx == 0:
                    sample_param = list(self.model.parameters())[0]
                
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_total_loss += total_loss.item()
            
            # Average losses
            epoch_policy_loss /= num_batches
            epoch_value_loss /= num_batches
            epoch_total_loss /= num_batches
            
            if epoch % 5 == 0 or epoch == self.num_epochs - 1:
                # Check if parameters have changed from initial
                if epoch == 0:
                    sample_param_name = list(self.model.named_parameters())[0][0]
                    sample_param = dict(self.model.named_parameters())[sample_param_name]
                    initial_param = self.initial_params[sample_param_name]
                    param_change = (sample_param.data - initial_param).abs().mean().item()
                    print(f"  Epoch {epoch + 1}/{self.num_epochs} ({num_batches} batches): "
                          f"Policy Loss: {epoch_policy_loss:.4f}, "
                          f"Value Loss: {epoch_value_loss:.8f}, "
                          f"Total Loss: {epoch_total_loss:.4f}, "
                          f"Param change: {param_change:.8f}")
                else:
                    print(f"  Epoch {epoch + 1}/{self.num_epochs} ({num_batches} batches): "
                          f"Policy Loss: {epoch_policy_loss:.4f}, "
                          f"Value Loss: {epoch_value_loss:.8f}, "
                          f"Total Loss: {epoch_total_loss:.4f}")
        
        self.model.eval()
        
        # Calculate and print training time
        training_elapsed = time.time() - training_start_time
        print(f"\n✓ Completed {self.num_epochs} epochs in {training_elapsed:.2f} seconds "
              f"({training_elapsed/60:.2f} minutes)")
        print(f"  Average time per epoch: {training_elapsed/self.num_epochs:.2f} seconds")
    
    def _save_checkpoint(self, iteration: int):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.save_dir, f'model_iter_{iteration}.pth')
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

