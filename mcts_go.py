"""
Python wrapper for Go MCTS engine using ctypes.
Provides high-performance MCTS search with neural network evaluation callbacks.
"""
import ctypes
import numpy as np
import os
from typing import Tuple, Optional, Callable
import numpy.ctypeslib as npct

# Try to load the shared library
_lib = None
_lib_path = os.path.join(os.path.dirname(__file__), 'libmctsengine.so')
if not os.path.exists(_lib_path):
    # Try alternative paths
    _lib_path = 'libmctsengine.so'
    if not os.path.exists(_lib_path):
        _lib_path = None

if _lib_path:
    try:
        _lib = ctypes.CDLL(_lib_path)
    except OSError as e:
        _lib = None
        import warnings
        warnings.warn(f"Failed to load libmctsengine.so: {e}. Go MCTS will not be available.")

# Define C types
_int8_p = ctypes.POINTER(ctypes.c_int8)
_float32_p = ctypes.POINTER(ctypes.c_float)
_float64_p = ctypes.POINTER(ctypes.c_double)
_uintptr_t = ctypes.c_uint64  # uintptr_t is typically 64-bit

# Define function signatures
# MCTSCreate(state *int8, current_player int8, num_simulations int, cPuct float) -> uintptr_t
if _lib is not None:
    _lib.MCTSCreate.argtypes = [
        _int8_p,           # state
        ctypes.c_int8,     # current_player
        ctypes.c_int,      # num_simulations
        ctypes.c_float,    # cPuct
    ]
    _lib.MCTSCreate.restype = _uintptr_t

    # MCTSDestroy(ctx_id uintptr_t)
    _lib.MCTSDestroy.argtypes = [_uintptr_t]
    _lib.MCTSDestroy.restype = None

    # MCTSSearchStep(ctx_id uintptr_t, canonical_state *float32) -> int
    # Returns: 0 = need root evaluation, 1 = need leaf evaluation, 2 = simulation done (terminal), 3 = all simulations complete
    _lib.MCTSSearchStep.argtypes = [
        _uintptr_t,        # ctx_id
        _float32_p,        # canonical_state (output)
    ]
    _lib.MCTSSearchStep.restype = ctypes.c_int

    # MCTSProvideEvaluation(ctx_id uintptr_t, policy *float64, value float64)
    _lib.MCTSProvideEvaluation.argtypes = [
        _uintptr_t,        # ctx_id
        _float64_p,        # policy
        ctypes.c_double,   # value
    ]
    _lib.MCTSProvideEvaluation.restype = None

    # MCTSGetPolicy(ctx_id uintptr_t, policy *float64)
    _lib.MCTSGetPolicy.argtypes = [
        _uintptr_t,        # ctx_id
        _float64_p,        # policy (output)
    ]
    _lib.MCTSGetPolicy.restype = None


class GoMCTS:
    """Go MCTS engine wrapper."""
    
    def __init__(self, eval_fn: Callable[[np.ndarray], Tuple[np.ndarray, float]], 
                 num_simulations: int = 100, c_puct: float = 1.0):
        """
        Initialize Go MCTS.
        
        Args:
            eval_fn: Function that takes canonical state (5x5 float32) and returns (policy, value)
            num_simulations: Number of MCTS simulations per move
            c_puct: Exploration constant
        """
        if _lib is None:
            raise RuntimeError("Go MCTS library not loaded. Build it with 'make build-mcts'")
        
        self.eval_fn = eval_fn
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.ctx = None
    
    def search(self, state: np.ndarray, current_player: int) -> np.ndarray:
        """
        Perform MCTS search and return improved policy.
        
        Args:
            state: 5x5 int8 numpy array representing board state
            current_player: 1 for WHITE, 2 for BLACK
        
        Returns:
            Policy distribution over actions (50-element float64 array)
        """
        if state.shape != (5, 5):
            raise ValueError(f"State must be 5x5, got {state.shape}")
        
        state_flat = state.flatten().astype(np.int8)
        
        # Create MCTS context
        ctx_id = _lib.MCTSCreate(
            state_flat.ctypes.data_as(_int8_p),
            ctypes.c_int8(current_player),
            ctypes.c_int(self.num_simulations),
            ctypes.c_float(self.c_puct),
        )
        
        if ctx_id == 0:
            raise RuntimeError("Failed to create MCTS context")
        
        try:
            # Allocate buffers
            canonical_state = np.zeros(25, dtype=np.float32)
            policy_buffer = np.zeros(50, dtype=np.float64)
            
            # MCTS search loop
            while True:
                status = _lib.MCTSSearchStep(
                    ctx_id,
                    canonical_state.ctypes.data_as(_float32_p),
                )
                
                if status == -1:
                    raise RuntimeError("Invalid MCTS context ID")
                
                if status == 0:  # Need root evaluation
                    # Reshape to 5x5 for evaluation
                    canonical_5x5 = canonical_state.reshape(5, 5)
                    policy, value = self.eval_fn(canonical_5x5)
                    
                    # Ensure policy is correct shape and type
                    if policy.shape != (50,):
                        raise ValueError(f"Policy must be shape (50,), got {policy.shape}")
                    policy = policy.astype(np.float64)
                    
                    # Provide evaluation
                    _lib.MCTSProvideEvaluation(
                        ctx_id,
                        policy.ctypes.data_as(_float64_p),
                        ctypes.c_double(value),
                    )
                
                elif status == 1:  # Need leaf evaluation
                    # Reshape to 5x5 for evaluation
                    canonical_5x5 = canonical_state.reshape(5, 5)
                    policy, value = self.eval_fn(canonical_5x5)
                    
                    # Ensure policy is correct shape and type
                    if policy.shape != (50,):
                        raise ValueError(f"Policy must be shape (50,), got {policy.shape}")
                    policy = policy.astype(np.float64)
                    
                    # Provide evaluation
                    _lib.MCTSProvideEvaluation(
                        ctx_id,
                        policy.ctypes.data_as(_float64_p),
                        ctypes.c_double(value),
                    )
                
                elif status == 2:  # Simulation done (terminal), continue
                    continue
                
                elif status == 3:  # All simulations complete
                    break
                
                else:
                    raise RuntimeError(f"Unexpected MCTS status: {status}")
            
            # Get final policy
            final_policy = np.zeros(50, dtype=np.float64)
            _lib.MCTSGetPolicy(
                ctx_id,
                final_policy.ctypes.data_as(_float64_p),
            )
            
            return final_policy
        
        finally:
            # Clean up
            _lib.MCTSDestroy(ctx_id)


def search(state: np.ndarray, current_player: int, eval_fn: Callable[[np.ndarray], Tuple[np.ndarray, float]],
           num_simulations: int = 100, c_puct: float = 1.0) -> np.ndarray:
    """
    Convenience function to perform MCTS search.
    
    Args:
        state: 5x5 int8 numpy array representing board state
        current_player: 1 for WHITE, 2 for BLACK
        eval_fn: Function that takes canonical state (5x5 float32) and returns (policy, value)
        num_simulations: Number of MCTS simulations per move
        c_puct: Exploration constant
    
    Returns:
        Policy distribution over actions (50-element float64 array)
    """
    mcts = GoMCTS(eval_fn, num_simulations, c_puct)
    return mcts.search(state, current_player)

