"""
Monte Carlo Tree Search implementation for AlphaZero.
"""
import numpy as np
import torch
from typing import List, Tuple, Optional
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
    Batched Monte Carlo Tree Search that batches leaf evaluations across multiple trees.
    
    Instead of batching within a single tree (which requires virtual loss and complex
    synchronization), this implementation maintains multiple independent MCTS trees
    (one per game) and batches the neural network evaluations of leaf nodes across trees.
    
    This approach:
    - Makes efficient use of GPU batch processing
    - Avoids virtual loss complexity
    - Is simpler and more predictable than within-tree parallelization
    """
    
    def __init__(self, model: torch.nn.Module, num_simulations: int = 100, c_puct: float = 1.0,
                 device: str = 'cpu'):
        """
        Initialize BatchedMCTS.
        
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
    
    def search_batch(self, games: List[KingCapture]) -> List[np.ndarray]:
        """
        Perform batched MCTS search on multiple games simultaneously.
        
        Leaf node evaluations are batched across all trees for efficient GPU utilization.
        
        Args:
            games: List of game states to search
            
        Returns:
            List of policy distributions, one per game
        """
        if not games:
            return []
        
        num_trees = len(games)
        
        # Initialize roots for all trees
        roots = [Node(game) for game in games]
        
        # Batch evaluate roots to get initial policies
        root_states = []
        for game in games:
            state = game.get_canonical_state()
            root_states.append(state)
        
        root_states_tensor = torch.FloatTensor(np.array(root_states)).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            policy_logits, values = self.model(root_states_tensor)
            policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
        
        # Expand all roots
        for i, (root, game) in enumerate(zip(roots, games)):
            policy = policies[i]
            
            # Flip policy if Black
            if game.current_player == Player.BLACK:
                policy = game.flip_policy(policy)
            
            # Mask and normalize
            action_mask = game.get_action_mask()
            policy = policy * action_mask
            policy_sum = policy.sum()
            if policy_sum > 0:
                policy = policy / policy_sum
            else:
                policy = action_mask.astype(float)
                policy = policy / (policy.sum() + 1e-8)
            
            root.expand(policy)
        
        # Run batched simulations
        for _ in range(self.num_simulations):
            # Selection phase: select leaf node for each tree
            leaves = []
            leaf_indices = []  # Track which trees have non-terminal leaves
            
            for tree_idx, root in enumerate(roots):
                node = root
                
                # Selection: traverse to leaf
                while node.is_expanded() and not node.game.game_over:
                    node = node.select_child(self.c_puct)
                
                # Check if terminal
                if node.game.game_over:
                    # Terminal node: backpropagate immediately
                    result = node.game.get_result(node.game.current_player)
                    value = result if result is not None else 0.0
                    node.backpropagate(value)
                else:
                    # Non-terminal leaf: add to batch
                    leaves.append(node)
                    leaf_indices.append(tree_idx)
            
            # Batch evaluate all non-terminal leaves
            if leaves:
                self._batch_evaluate(leaves)
        
        # Extract policies from all trees
        policies_out = []
        for root, game in zip(roots, games):
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
                policy = policy / (policy.sum() + 1e-8)
            
            policies_out.append(policy)
        
        return policies_out
    
    def _batch_evaluate(self, nodes: List[Node]):
        """
        Batch evaluate multiple leaf nodes using a single neural network forward pass.
        
        Args:
            nodes: List of leaf nodes to evaluate
        """
        if not nodes:
            return
        
        # Collect states for batch evaluation
        states = []
        for node in nodes:
            state = node.game.get_canonical_state()
            states.append(state)
        
        # Batch forward pass
        states_tensor = torch.FloatTensor(np.array(states)).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            policy_logits, values = self.model(states_tensor)
            policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
            values_np = values.cpu().numpy().flatten()
        
        # Expand and backpropagate for each node
        for i, node in enumerate(nodes):
            policy = policies[i]
            value = values_np[i]
            
            # Flip policy if Black
            if node.game.current_player == Player.BLACK:
                policy = node.game.flip_policy(policy)
            
            # Mask and normalize
            action_mask = node.game.get_action_mask()
            policy = policy * action_mask
            policy_sum = policy.sum()
            if policy_sum > 0:
                policy = policy / policy_sum
            else:
                policy = action_mask.astype(float)
                policy = policy / (policy.sum() + 1e-8)
            
            # Expand and backpropagate
            node.expand(policy)
            node.backpropagate(value)
    
    def search(self, game: KingCapture) -> np.ndarray:
        """
        Perform MCTS search on a single game (convenience method).
        
        Args:
            game: Current game state
            
        Returns:
            Policy distribution over actions
        """
        return self.search_batch([game])[0]
    
    def _game_to_tensor(self, game: KingCapture) -> torch.Tensor:
        """Convert game state to tensor for neural network."""
        state = game.get_canonical_state()
        tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)
