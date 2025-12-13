"""
Monte Carlo Tree Search implementation for AlphaZero.
Includes both sequential and batched MCTS for efficient GPU utilization.
"""
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
from game import KingCapture, Player
import math


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
        
        # For batched MCTS: track if this node is pending evaluation
        self.pending_evaluation = False
    
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
        best_action = None
        best_child = None
        
        for action, child in self.children.items():
            # PUCT formula: Q + U
            #
            # IMPORTANT SIGN NOTE:
            # - `child.get_value()` is from the *child node's* `current_player` perspective.
            # - At the parent, the player-to-move is the opponent of the child.
            # Therefore, from the *parent* perspective, Q(s,a) = -V(child).
            #
            # U = c_puct * P * sqrt(N_parent) / (1 + N_child)
            u = c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
            score = (-child.get_value()) + u
            
            if score > best_score:
                best_score = score
                best_action = action
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
                  1.0 for win, -1.0 for loss, 0.0 for draw
        """
        # Add real value
        self.value_sum += value
        self.visit_count += 1
        
        if self.parent is not None:
            # Value is from current player's perspective
            # For parent (opponent), negate the value (zero-sum game)
            self.parent.backpropagate(-value)


class MCTS:
    """Monte Carlo Tree Search for AlphaZero."""
    
    def __init__(self, model: torch.nn.Module, num_simulations: int = 100, c_puct: float = 1.0, 
                 device: str = 'cpu'):
        """
        Initialize MCTS.
        
        Args:
            model: Neural network model
            num_simulations: Number of MCTS simulations per move
            c_puct: Exploration constant
            device: Device to run model on
        """
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device
        self.model.eval()
    
    def search(self, game: KingCapture) -> np.ndarray:
        """
        Perform MCTS search and return improved policy.
        
        Args:
            game: Current game state
        
        Returns:
            Policy distribution over actions
        """
        root = Node(game)
        
        # Get initial policy from neural network
        with torch.no_grad():
            state = self._game_to_tensor(game)
            policy_logits, value = self.model(state)
            policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
            
            # If playing as Black, flip policy from canonical view to real board view
            if game.current_player == Player.BLACK:
                policy = game.flip_policy(policy)
            
            # Mask invalid actions
            action_mask = game.get_action_mask()
            policy = policy * action_mask
            policy = policy / (policy.sum() + 1e-8)  # Renormalize
        
        root.expand(policy)
        
        # Sequential MCTS simulations
        for sim in range(self.num_simulations):
            node = root
            
            # Selection: traverse to leaf
            while node.is_expanded() and not node.game.game_over:
                node = node.select_child(self.c_puct)
            
            # Check if terminal
            if node.game.game_over:
                result = node.game.get_result(node.game.current_player)
                value = result if result is not None else 0.0
                node.backpropagate(value)
            else:
                # Evaluate and expand this leaf node
                self._evaluate(node)
        
        # Extract visit counts as policy
        action_size = 2 * game.BOARD_SIZE * game.BOARD_SIZE
        visit_counts = np.zeros(action_size)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count
        
        # Normalize to get policy distribution
        if visit_counts.sum() > 0:
            policy = visit_counts / visit_counts.sum()
        else:
            # Fallback to uniform over valid moves
            action_mask = game.get_action_mask()
            policy = action_mask.astype(float)
            policy = policy / policy.sum()
        
        return policy
    
    def search_batch(self, games: List[KingCapture]) -> List[np.ndarray]:
        """
        Perform MCTS search on multiple games sequentially.
        
        Args:
            games: List of game states to search
            
        Returns:
            List of policy distributions, one per game
        """
        return [self.search(game) for game in games]
    
    def _evaluate(self, node: Node):
        """
        Evaluate a single node using the neural network.
        
        Args:
            node: Node to evaluate
        """
        # Convert to tensor
        tensor = self._game_to_tensor(node.game)
        
        # Evaluate
        with torch.no_grad():
            policy_logits, values = self.model(tensor)
            policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
            value = values.item()
        
        # Flip policy if Black
        if node.game.current_player == Player.BLACK:
            policy = node.game.flip_policy(policy)
        
        # Mask invalid actions
        action_mask = node.game.get_action_mask()
        policy = policy * action_mask
        policy = policy / (policy.sum() + 1e-8)
        
        node.expand(policy)
        node.backpropagate(value)
    
    def _game_to_tensor(self, game: KingCapture) -> torch.Tensor:
        """Convert game state to tensor for neural network."""
        state = game.get_canonical_state()
        # Add batch and channel dimensions: (1, 1, board_size, board_size)
        tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)


class BatchedMCTS:
    """
    Batched Monte Carlo Tree Search for efficient GPU utilization.
    
    This implementation runs MCTS for multiple games simultaneously,
    collecting all leaf nodes that need neural network evaluation and
    processing them in a single batched GPU call.
    """
    
    def __init__(self, model: torch.nn.Module, num_simulations: int = 100, 
                 c_puct: float = 1.0, device: str = 'cpu', batch_size: int = 256):
        """
        Initialize Batched MCTS.
        
        Args:
            model: Neural network model
            num_simulations: Number of MCTS simulations per move
            c_puct: Exploration constant
            device: Device to run model on
            batch_size: Maximum batch size for neural network evaluation
        """
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device
        self.batch_size = batch_size
        self.model.eval()
    
    def search_batch(self, games: List[KingCapture]) -> List[np.ndarray]:
        """
        Perform MCTS search on multiple games with batched neural network evaluation.
        
        This is the main entry point for efficient parallel game generation.
        All games are processed together, and neural network calls are batched.
        
        Args:
            games: List of game states to search
            
        Returns:
            List of policy distributions, one per game
        """
        if not games:
            return []
        
        num_games = len(games)
        
        # Initialize roots for all games
        roots = [Node(game.copy()) for game in games]
        
        # First, expand all roots with batched neural network evaluation
        self._batch_expand_roots(roots, games)
        
        # Run simulations with batched evaluation
        for sim in range(self.num_simulations):
            # For each game, traverse to a leaf node
            leaves_to_evaluate = []
            terminal_nodes = []
            
            for game_idx, root in enumerate(roots):
                node = root
                
                # Selection: traverse to leaf
                while node.is_expanded() and not node.game.game_over:
                    node = node.select_child(self.c_puct)
                
                # Check if terminal
                if node.game.game_over:
                    terminal_nodes.append((game_idx, node))
                else:
                    leaves_to_evaluate.append((game_idx, node))
            
            # Handle terminal nodes immediately
            for game_idx, node in terminal_nodes:
                result = node.game.get_result(node.game.current_player)
                value = result if result is not None else 0.0
                node.backpropagate(value)
            
            # Batch evaluate all leaves
            if leaves_to_evaluate:
                self._batch_evaluate(leaves_to_evaluate)
        
        # Extract policies from visit counts
        policies = []
        action_size = 2 * games[0].BOARD_SIZE * games[0].BOARD_SIZE
        
        for game_idx, (root, game) in enumerate(zip(roots, games)):
            visit_counts = np.zeros(action_size)
            for action, child in root.children.items():
                visit_counts[action] = child.visit_count
            
            # Normalize to get policy distribution
            if visit_counts.sum() > 0:
                policy = visit_counts / visit_counts.sum()
            else:
                # Fallback to uniform over valid moves
                action_mask = game.get_action_mask()
                policy = action_mask.astype(float)
                policy = policy / (policy.sum() + 1e-8)
            
            policies.append(policy)
        
        return policies
    
    def _batch_expand_roots(self, roots: List[Node], games: List[KingCapture]):
        """
        Expand root nodes with a batched neural network call.
        
        Args:
            roots: List of root nodes to expand
            games: Original game states (for reference)
        """
        if not roots:
            return
        
        # Prepare batch of states
        states = []
        for root in roots:
            state = root.game.get_canonical_state()
            states.append(state)
        
        # Stack into batch tensor
        batch_tensor = torch.FloatTensor(np.array(states)).unsqueeze(1).to(self.device)
        
        # Batch evaluate
        with torch.no_grad():
            policy_logits, values = self.model(batch_tensor)
            policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
        
        # Expand each root with its policy
        for idx, (root, game) in enumerate(zip(roots, games)):
            policy = policies[idx]
            
            # Flip policy if Black
            if game.current_player == Player.BLACK:
                policy = game.flip_policy(policy)
            
            # Mask invalid actions
            action_mask = game.get_action_mask()
            policy = policy * action_mask
            policy = policy / (policy.sum() + 1e-8)
            
            root.expand(policy)
    
    def _batch_evaluate(self, leaves: List[Tuple[int, Node]]):
        """
        Batch evaluate multiple leaf nodes and expand/backpropagate them.
        
        Args:
            leaves: List of (game_idx, node) tuples to evaluate
        """
        if not leaves:
            return
        
        # Prepare batch of states
        states = []
        for game_idx, node in leaves:
            state = node.game.get_canonical_state()
            states.append(state)
        
        # Stack into batch tensor
        batch_tensor = torch.FloatTensor(np.array(states)).unsqueeze(1).to(self.device)
        
        # Batch evaluate
        with torch.no_grad():
            policy_logits, values = self.model(batch_tensor)
            policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
            values_np = values.cpu().numpy().flatten()
        
        # Expand and backpropagate each node
        for idx, (game_idx, node) in enumerate(leaves):
            policy = policies[idx]
            value = float(values_np[idx])
            
            # Flip policy if Black
            if node.game.current_player == Player.BLACK:
                policy = node.game.flip_policy(policy)
            
            # Mask invalid actions
            action_mask = node.game.get_action_mask()
            policy = policy * action_mask
            policy_sum = policy.sum()
            if policy_sum > 0:
                policy = policy / policy_sum
            
            node.expand(policy)
            node.backpropagate(value)


class ParallelGameRunner:
    """
    Runs multiple games in parallel with batched MCTS.
    
    This is the main class for efficient game generation.
    It manages multiple games simultaneously and uses batched
    neural network evaluation for maximum GPU throughput.
    """
    
    def __init__(self, model: torch.nn.Module, num_simulations: int = 100,
                 c_puct: float = 1.0, temperature: float = 1.0,
                 device: str = 'cpu', batch_size: int = 256):
        """
        Initialize the parallel game runner.
        
        Args:
            model: Neural network model
            num_simulations: MCTS simulations per move
            c_puct: MCTS exploration constant
            temperature: Action selection temperature
            device: Device for neural network
            batch_size: Batch size for neural network calls
        """
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.device = device
        self.batch_size = batch_size
        self.mcts = BatchedMCTS(model, num_simulations, c_puct, device, batch_size)
    
    def run_games(self, num_games: int, max_game_length: int = 400,
                  log_interval: int = 10) -> List[Tuple]:
        """
        Run multiple games in parallel and return training examples.
        
        Args:
            num_games: Number of games to run
            max_game_length: Maximum moves per game
            log_interval: How often to log progress
            
        Returns:
            List of (state, policy, value) training examples
        """
        import logging
        
        # Initialize all games
        games = [KingCapture() for _ in range(num_games)]
        game_histories = [[] for _ in range(num_games)]
        active_indices = list(range(num_games))
        
        examples = []
        move_count = 0
        
        # Play until all games are finished
        while active_indices:
            move_count += 1
            
            # Get current games
            current_games = [games[i] for i in active_indices]
            
            # Batched MCTS search for all active games
            policies = self.mcts.search_batch(current_games)
            
            # Process each game
            finished_indices = []
            
            for idx_in_batch, policy in enumerate(policies):
                game_idx = active_indices[idx_in_batch]
                game = games[game_idx]
                
                # Select action
                valid_moves = game.get_valid_moves()
                valid_actions = [game.move_to_action(piece_idx, r, c) 
                               for piece_idx, r, c in valid_moves]
                
                if self.temperature > 0:
                    # Sample with temperature
                    valid_policy = policy[valid_actions]
                    valid_policy = valid_policy ** (1.0 / self.temperature)
                    valid_policy = valid_policy / (valid_policy.sum() + 1e-8)
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
                
                game_histories[game_idx].append((state, store_policy))
                
                # Apply move
                piece_idx, row, col = game.action_to_move(action)
                game.make_move(piece_idx, row, col)
                
                # Check move limit
                max_moves_reached = False
                if len(game.move_history) >= max_game_length:
                    game.game_over = True
                    game.winner = None
                    max_moves_reached = True
                
                if game.game_over:
                    finished_indices.append(idx_in_batch)
                    
                    # Process result
                    result = game.get_result(Player.WHITE)
                    if max_moves_reached:
                        result = -1.0
                    elif result is None:
                        result = 0.0
                    
                    # Log completion
                    num_moves = len(game.move_history)
                    winner = game.winner.name if game.winner else "Draw"
                    remaining = len(active_indices) - len(finished_indices)
                    
                    if (num_games - remaining) % log_interval == 0 or remaining == 0:
                        logging.info(f"Games completed: {num_games - remaining}/{num_games} "
                                   f"(last: {winner} in {num_moves} moves)")
                    
                    # Store training examples
                    for i, (hist_state, hist_policy) in enumerate(game_histories[game_idx]):
                        if max_moves_reached:
                            value = -1.0
                        else:
                            if i % 2 == 0:
                                value = result
                            else:
                                value = -result
                        
                        examples.append((hist_state, hist_policy, value))
            
            # Remove finished games
            for idx_in_batch in sorted(finished_indices, reverse=True):
                active_indices.pop(idx_in_batch)
        
        return examples
