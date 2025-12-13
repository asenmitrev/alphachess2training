"""
Monte Carlo Tree Search implementation for AlphaZero.
Includes both sequential MCTS and parallel MCTS with centralized inference server.
"""
import numpy as np
import torch
import torch.multiprocessing as mp
from typing import List, Tuple, Optional, Dict
from game import KingCapture, Player
import math
import queue
import time


class Node:
    """Node in the MCTS tree."""
    
    def __init__(self, game: KingCapture, parent: Optional['Node'] = None, action: Optional[int] = None):
        """
        Initialize a node.
        
        Args:
            game: Game state at this node
            parent: Parent node
            action: Action that led to this node
        """
        self.game = game
        self.parent = parent
        self.action = action
        
        self.children = {}  # action -> Node
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = 0.0  # Prior probability from neural network
    
    def is_expanded(self) -> bool:
        """Check if node has been expanded."""
        return len(self.children) > 0
    
    def get_value(self) -> float:
        """Get average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def select_child(self, c_puct: float = 1.0) -> 'Node':
        """
        Select child using UCB formula.
        
        Args:
            c_puct: Exploration constant
        """
        best_score = float('-inf')
        best_child = None
        
        for action, child in self.children.items():
            # PUCT formula: Q + U
            u = c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
            score = (-child.get_value()) + u
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def expand(self, policy: np.ndarray):
        """
        Expand node by creating children for all valid actions.
        
        Args:
            policy: Policy distribution over actions (from neural network)
        """
        valid_moves = self.game.get_valid_moves()
        
        for piece_idx, row, col in valid_moves:
            action = self.game.move_to_action(piece_idx, row, col)
            new_game = self.game.copy()
            new_game.make_move(piece_idx, row, col)
            child = Node(new_game, parent=self, action=action)
            child.prior = policy[action]
            self.children[action] = child
    
    def backpropagate(self, value: float):
        """
        Backpropagate value up the tree.
        
        Args:
            value: Value to propagate (from current player's perspective)
        """
        self.value_sum += value
        self.visit_count += 1
        
        if self.parent is not None:
            self.parent.backpropagate(-value)


class MCTS:
    """Monte Carlo Tree Search for AlphaZero (sequential version)."""
    
    def __init__(self, model: torch.nn.Module, num_simulations: int = 100, c_puct: float = 1.0, 
                 device: str = 'cpu'):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device
        self.model.eval()
    
    def search(self, game: KingCapture) -> np.ndarray:
        """Perform MCTS search and return improved policy."""
        root = Node(game)
        
        with torch.no_grad():
            state = self._game_to_tensor(game)
            policy_logits, value = self.model(state)
            policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
            
            if game.current_player == Player.BLACK:
                policy = game.flip_policy(policy)
            
            action_mask = game.get_action_mask()
            policy = policy * action_mask
            policy = policy / (policy.sum() + 1e-8)
        
        root.expand(policy)
        
        for sim in range(self.num_simulations):
            node = root
            
            while node.is_expanded() and not node.game.game_over:
                node = node.select_child(self.c_puct)
            
            if node.game.game_over:
                result = node.game.get_result(node.game.current_player)
                value = result if result is not None else 0.0
                node.backpropagate(value)
            else:
                self._evaluate(node)
        
        action_size = 2 * game.BOARD_SIZE * game.BOARD_SIZE
        visit_counts = np.zeros(action_size)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count
        
        if visit_counts.sum() > 0:
            policy = visit_counts / visit_counts.sum()
        else:
            action_mask = game.get_action_mask()
            policy = action_mask.astype(float)
            policy = policy / policy.sum()
        
        return policy
    
    def search_batch(self, games: List[KingCapture]) -> List[np.ndarray]:
        """Perform MCTS search on multiple games sequentially."""
        return [self.search(game) for game in games]
    
    def _evaluate(self, node: Node):
        """Evaluate a single node using the neural network."""
        tensor = self._game_to_tensor(node.game)
        
        with torch.no_grad():
            policy_logits, values = self.model(tensor)
            policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
            value = values.item()
        
        if node.game.current_player == Player.BLACK:
            policy = node.game.flip_policy(policy)
        
        action_mask = node.game.get_action_mask()
        policy = policy * action_mask
        policy = policy / (policy.sum() + 1e-8)
        
        node.expand(policy)
        node.backpropagate(value)
    
    def _game_to_tensor(self, game: KingCapture) -> torch.Tensor:
        """Convert game state to tensor for neural network."""
        state = game.get_canonical_state()
        tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)


# ============================================================================
# Inference Server Architecture for Parallel MCTS
# ============================================================================

def inference_server(model_state: Dict, config: Dict, request_queue: mp.Queue, 
                     response_queues: Dict[int, mp.Queue], shutdown_event: mp.Event):
    """
    Centralized inference server that batches GPU requests from all workers.
    
    This process owns the GPU and batches evaluation requests from workers
    for maximum throughput.
    
    Args:
        model_state: State dict of the neural network
        config: Configuration dict with model params
        request_queue: Queue to receive evaluation requests from workers
        response_queues: Dict mapping worker_id -> response queue
        shutdown_event: Event to signal shutdown
    """
    import logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    
    # Reconstruct model on GPU
    from model import AlphaZeroNet
    device = config['device']
    model = AlphaZeroNet(
        board_size=config['board_size'],
        num_channels=config['num_channels'],
        num_residual_blocks=config['num_residual_blocks']
    )
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    
    batch_size = config.get('batch_size', 256)
    batch_timeout = config.get('batch_timeout', 0.001)  # 1ms timeout to collect batch
    
    logging.info(f"Inference server started on {device}, batch_size={batch_size}")
    
    pending_requests = []
    
    while not shutdown_event.is_set():
        # Collect requests into a batch
        try:
            # Wait for first request with timeout
            request = request_queue.get(timeout=0.1)
            pending_requests.append(request)
            
            # Collect more requests up to batch_size or timeout
            batch_start = time.time()
            while len(pending_requests) < batch_size:
                try:
                    elapsed = time.time() - batch_start
                    remaining_timeout = max(0.0001, batch_timeout - elapsed)
                    request = request_queue.get(timeout=remaining_timeout)
                    pending_requests.append(request)
                except queue.Empty:
                    break
            
        except queue.Empty:
            continue
        
        if not pending_requests:
            continue
        
        # Process batch
        try:
            # Extract states and metadata
            states = []
            worker_ids = []
            request_ids = []
            is_black_list = []
            
            for req in pending_requests:
                worker_id, request_id, state, is_black = req
                states.append(state)
                worker_ids.append(worker_id)
                request_ids.append(request_id)
                is_black_list.append(is_black)
            
            # Batch evaluate
            batch_tensor = torch.FloatTensor(np.array(states)).unsqueeze(1).to(device)
            
            with torch.no_grad():
                policy_logits, values = model(batch_tensor)
                policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
                values_np = values.cpu().numpy().flatten()
            
            # Send responses back to workers
            for idx, (worker_id, request_id) in enumerate(zip(worker_ids, request_ids)):
                policy = policies[idx]
                value = float(values_np[idx])
                response_queues[worker_id].put((request_id, policy, value))
            
        except Exception as e:
            logging.error(f"Inference server error: {e}")
            # Send error responses
            for req in pending_requests:
                worker_id, request_id, _, _ = req
                # Send uniform policy and 0 value as fallback
                policy = np.ones(50) / 50  # Uniform over action space
                response_queues[worker_id].put((request_id, policy, 0.0))
        
        pending_requests = []
    
    logging.info("Inference server shutting down")


def worker_process(rank: int, config: Dict, work_queue: mp.Queue, result_queue: mp.Queue,
                   request_queue: mp.Queue, response_queue: mp.Queue, shutdown_event: mp.Event):
    """
    Worker process that runs MCTS games using the inference server.
    
    Args:
        rank: Worker ID
        config: Configuration dict
        work_queue: Queue to get game IDs from
        result_queue: Queue to send training examples to
        request_queue: Queue to send inference requests to server
        response_queue: Queue to receive inference responses from server
        shutdown_event: Event to signal shutdown
    """
    import logging
    import random
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    
    # Set random seed
    seed = int(time.time() * 1000) % (2**32) + rank
    np.random.seed(seed)
    random.seed(seed)
    
    num_simulations = config['num_simulations']
    c_puct = config['c_puct']
    temperature = config['temperature']
    max_game_length = config.get('max_game_length', 400)
    
    all_examples = []
    games_processed = 0
    request_counter = 0
    
    def evaluate_state(state: np.ndarray, is_black: bool) -> Tuple[np.ndarray, float]:
        """Send state to inference server and wait for response."""
        nonlocal request_counter
        request_id = request_counter
        request_counter += 1
        
        request_queue.put((rank, request_id, state, is_black))
        
        # Wait for response
        while True:
            try:
                resp_id, policy, value = response_queue.get(timeout=1.0)
                if resp_id == request_id:
                    return policy, value
                # Wrong response, put it back (shouldn't happen)
                response_queue.put((resp_id, policy, value))
            except queue.Empty:
                if shutdown_event.is_set():
                    return np.ones(50) / 50, 0.0
    
    def run_mcts(game: KingCapture) -> np.ndarray:
        """Run MCTS for a single position using inference server."""
        root = Node(game.copy())
        
        # Get initial policy for root
        state = game.get_canonical_state()
        is_black = game.current_player == Player.BLACK
        policy, _ = evaluate_state(state, is_black)
        
        if is_black:
            policy = game.flip_policy(policy)
        
        action_mask = game.get_action_mask()
        policy = policy * action_mask
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy = policy / policy_sum
        
        root.expand(policy)
        
        # Run simulations
        for sim in range(num_simulations):
            node = root
            
            # Selection
            while node.is_expanded() and not node.game.game_over:
                node = node.select_child(c_puct)
            
            # Terminal check
            if node.game.game_over:
                result = node.game.get_result(node.game.current_player)
                value = result if result is not None else 0.0
                node.backpropagate(value)
            else:
                # Evaluate leaf
                state = node.game.get_canonical_state()
                is_black = node.game.current_player == Player.BLACK
                policy, value = evaluate_state(state, is_black)
                
                if is_black:
                    policy = node.game.flip_policy(policy)
                
                action_mask = node.game.get_action_mask()
                policy = policy * action_mask
                policy_sum = policy.sum()
                if policy_sum > 0:
                    policy = policy / policy_sum
                
                node.expand(policy)
                node.backpropagate(value)
        
        # Extract policy from visit counts
        action_size = 2 * game.BOARD_SIZE * game.BOARD_SIZE
        visit_counts = np.zeros(action_size)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count
        
        if visit_counts.sum() > 0:
            return visit_counts / visit_counts.sum()
        else:
            action_mask = game.get_action_mask()
            return action_mask.astype(float) / action_mask.sum()
    
    # Main loop: pull games from queue
    while not shutdown_event.is_set():
        try:
            game_id = work_queue.get(timeout=0.1)
        except queue.Empty:
            break
        
        try:
            # Play one game
            game = KingCapture()
            game_history = []
            
            while not game.game_over:
                if len(game.move_history) >= max_game_length:
                    game.game_over = True
                    game.winner = None
                    break
                
                # Run MCTS
                policy = run_mcts(game)
                
                # Select action
                valid_moves = game.get_valid_moves()
                valid_actions = [game.move_to_action(p, r, c) for p, r, c in valid_moves]
                
                if temperature > 0:
                    valid_policy = policy[valid_actions]
                    valid_policy = valid_policy ** (1.0 / temperature)
                    valid_policy = valid_policy / (valid_policy.sum() + 1e-8)
                    action = np.random.choice(valid_actions, p=valid_policy)
                else:
                    action = valid_actions[np.argmax(policy[valid_actions])]
                
                # Store history
                state = game.get_canonical_state()
                store_policy = policy
                if game.current_player == Player.BLACK:
                    store_policy = game.flip_policy(policy)
                
                game_history.append((state, store_policy))
                
                # Make move
                piece_idx, row, col = game.action_to_move(action)
                game.make_move(piece_idx, row, col)
            
            # Process result
            max_moves_reached = len(game.move_history) >= max_game_length
            result = game.get_result(Player.WHITE)
            if max_moves_reached:
                result = -1.0
            elif result is None:
                result = 0.0
            
            # Create training examples
            for i, (hist_state, hist_policy) in enumerate(game_history):
                if max_moves_reached:
                    value = -1.0
                else:
                    value = result if i % 2 == 0 else -result
                all_examples.append((hist_state, hist_policy, value))
            
            games_processed += 1
            
            if games_processed % 10 == 0:
                winner = game.winner.name if game.winner else "Draw"
                logging.info(f"Worker {rank}: {games_processed} games done (last: {winner} in {len(game.move_history)} moves)")
            
            work_queue.task_done()
            
        except Exception as e:
            logging.error(f"Worker {rank} error: {e}")
            work_queue.task_done()
    
    # Send results
    result_queue.put((rank, all_examples))
    logging.info(f"Worker {rank} finished: {games_processed} games, {len(all_examples)} examples")


class ParallelMCTSRunner:
    """
    Manages parallel MCTS game generation with a centralized inference server.
    
    Architecture:
    - One inference server process owns the GPU and batches requests
    - Multiple worker processes run MCTS trees on CPU
    - Workers send states to evaluate, server batches and returns results
    
    This maximizes GPU utilization while allowing parallel tree traversal.
    """
    
    def __init__(self, model: torch.nn.Module, num_simulations: int = 100,
                 c_puct: float = 1.0, temperature: float = 1.0,
                 device: str = 'cuda', num_workers: int = 4,
                 batch_size: int = 256, batch_timeout: float = 0.001):
        """
        Initialize the parallel MCTS runner.
        
        Args:
            model: Neural network model
            num_simulations: MCTS simulations per move
            c_puct: MCTS exploration constant
            temperature: Action selection temperature
            device: Device for neural network (should be 'cuda')
            num_workers: Number of parallel worker processes
            batch_size: Maximum batch size for inference server
            batch_timeout: How long to wait to collect a batch (seconds)
        """
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.device = device
        self.num_workers = max(1, num_workers)
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
    
    def run_games(self, num_games: int, max_game_length: int = 400) -> List[Tuple]:
        """
        Run multiple games in parallel using inference server architecture.
        
        Args:
            num_games: Number of games to generate
            max_game_length: Maximum moves per game
            
        Returns:
            List of (state, policy, value) training examples
        """
        import logging
        
        ctx = mp.get_context('spawn')
        
        # Create queues
        work_queue = ctx.JoinableQueue()
        result_queue = ctx.Queue()
        request_queue = ctx.Queue()
        response_queues = {i: ctx.Queue() for i in range(self.num_workers)}
        shutdown_event = ctx.Event()
        
        # Put all games into work queue
        for game_id in range(num_games):
            work_queue.put(game_id)
        
        # Config for workers
        config = {
            'board_size': self.model.board_size,
            'num_channels': self.model.num_channels,
            'num_residual_blocks': self.model.num_residual_blocks,
            'num_simulations': self.num_simulations,
            'c_puct': self.c_puct,
            'temperature': self.temperature,
            'device': self.device,
            'max_game_length': max_game_length,
            'batch_size': self.batch_size,
            'batch_timeout': self.batch_timeout,
        }
        
        model_state = self.model.state_dict()
        
        # Start inference server
        server_process = ctx.Process(
            target=inference_server,
            args=(model_state, config, request_queue, response_queues, shutdown_event)
        )
        server_process.start()
        
        # Give server time to initialize
        time.sleep(0.5)
        
        # Start workers
        workers = []
        for rank in range(self.num_workers):
            p = ctx.Process(
                target=worker_process,
                args=(rank, config, work_queue, result_queue, 
                      request_queue, response_queues[rank], shutdown_event)
            )
            p.start()
            workers.append(p)
        
        # Collect results
        all_examples = []
        for _ in range(self.num_workers):
            rank, examples = result_queue.get()
            all_examples.extend(examples)
            logging.info(f"Collected {len(examples)} examples from worker {rank}")
        
        # Shutdown
        shutdown_event.set()
        
        for p in workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        
        server_process.join(timeout=5)
        if server_process.is_alive():
            server_process.terminate()
        
        return all_examples


# Backward compatibility alias
BatchedMCTS = MCTS
ParallelGameRunner = ParallelMCTSRunner
