"""
Monte Carlo Tree Search implementation for AlphaZero - WASM/Pyodide version.
Removes Torch dependencies and uses callbacks for model inference.
"""
import numpy as np
import math
from typing import List, Tuple, Optional, Callable, Any
from game import TicTacToe, Player

# Define a type for the inference callback
# It takes a list of canonical states (numpy arrays) and returns (policies, values)
# policies: List[np.ndarray], values: List[float]
InferenceCallback = Callable[[List[np.ndarray]], Any]

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class Node:
    """Node in the MCTS tree."""
    
    def __init__(self, game: TicTacToe, parent: Optional['Node'] = None, action: Optional[int] = None):
        self.game = game
        self.parent = parent
        self.action = action
        
        self.children = {}  # action -> Node
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = 0.0
    
    def is_expanded(self) -> bool:
        return len(self.children) > 0
    
    def get_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def select_child(self, c_puct: float = 1.0) -> 'Node':
        best_score = float('-inf')
        best_child = None
        
        for action, child in self.children.items():
            u = c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
            score = child.get_value() + u
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def expand(self, policy: np.ndarray):
        valid_moves = self.game.get_valid_moves()
        
        for row, col in valid_moves:
            action = self.game.move_to_action(row, col)
            new_game = self.game.copy()
            new_game.make_move(row, col)
            child = Node(new_game, parent=self, action=action)
            child.prior = policy[action]
            self.children[action] = child
    
    def backpropagate(self, value: float):
        self.value_sum += value
        self.visit_count += 1
        
        if self.parent is not None:
            self.parent.backpropagate(-value)


class MCTS:
    """Monte Carlo Tree Search for AlphaZero (WASM version)."""
    
    def __init__(self, inference_fn: InferenceCallback, num_simulations: int = 100, c_puct: float = 1.0):
        """
        Initialize MCTS.
        
        Args:
            inference_fn: Async function that takes batch of states and returns (policies, values)
            num_simulations: Number of simulations per move
            c_puct: Exploration constant
        """
        self.inference_fn = inference_fn
        self.num_simulations = num_simulations
        self.c_puct = c_puct
    
    async def search(self, game: TicTacToe) -> np.ndarray:
        """
        Perform MCTS search and return policy.
        """
        root = Node(game)
        
        # Initial inference
        # In this simplified version, we process one by one or small batches.
        # But here we just need the root evaluation first.
        state = game.get_canonical_state()
        # Expect inference_fn to return lists
        policies, values = await self.inference_fn([state])
        policy = softmax(policies[0])
        
        # Mask and normalize
        action_mask = game.get_action_mask()
        policy = policy * action_mask
        policy = policy / (policy.sum() + 1e-8)
        
        root.expand(policy)
        
        for _ in range(self.num_simulations):
            node = root
            
            # Selection
            while node.is_expanded() and not node.game.game_over:
                node = node.select_child(self.c_puct)
            
            # Evaluation
            if not node.game.game_over:
                state = node.game.get_canonical_state()
                policies, values = await self.inference_fn([state])
                policy = softmax(policies[0])
                value = values[0]
                
                # Mask and normalize
                action_mask = node.game.get_action_mask()
                policy = policy * action_mask
                policy = policy / (policy.sum() + 1e-8)
                
                node.expand(policy)
                node.backpropagate(value)
            else:
                result = node.game.get_result(node.game.current_player)
                value = result if result is not None else 0.0
                node.backpropagate(value)
        
        # Extract policy from visit counts
        visit_counts = np.zeros(game.BOARD_SIZE * game.BOARD_SIZE)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count
            
        if visit_counts.sum() > 0:
            policy = visit_counts / visit_counts.sum()
        else:
            action_mask = game.get_action_mask()
            policy = action_mask.astype(float)
            policy = policy / policy.sum()
            
        return policy

