"""
Python wrapper for Go game engine using ctypes.
Provides tensor-based interface for maximum performance.
"""
import ctypes
import numpy as np
import os
from typing import Tuple, Optional
import numpy.ctypeslib as npct

# Try to load the shared library
_lib = None
_lib_path = os.path.join(os.path.dirname(__file__), 'libgameengine.so')
if not os.path.exists(_lib_path):
    # Try alternative paths
    _lib_path = 'libgameengine.so'
    if not os.path.exists(_lib_path):
        _lib_path = None

if _lib_path:
    try:
        _lib = ctypes.CDLL(_lib_path)
    except OSError as e:
        _lib = None
        import warnings
        warnings.warn(f"Failed to load libgameengine.so: {e}. Go engine will not be available.")

# Define C types
_int8_p = ctypes.POINTER(ctypes.c_int8)
_float32_p = ctypes.POINTER(ctypes.c_float)

# Define function signatures
# MakeMove(state *int8, current_player int8, piece_idx int8, row int8, col int8,
#          new_state *int8, game_over *int8, winner *int8) -> int8
_lib.MakeMove.argtypes = [
    _int8_p,      # state
    ctypes.c_int8,  # current_player
    ctypes.c_int8,  # piece_idx
    ctypes.c_int8,  # row
    ctypes.c_int8,  # col
    _int8_p,      # new_state
    _int8_p,      # game_over
    _int8_p,      # winner
]
_lib.MakeMove.restype = ctypes.c_int8

# GetValidMoves(state *int8, current_player int8, mask *int8)
_lib.GetValidMoves.argtypes = [
    _int8_p,      # state
    ctypes.c_int8,  # current_player
    _int8_p,      # mask
]
_lib.GetValidMoves.restype = None

# GetCanonicalState(state *int8, current_player int8, canonical_state *float32)
_lib.GetCanonicalState.argtypes = [
    _int8_p,      # state
    ctypes.c_int8,  # current_player
    _float32_p,   # canonical_state
]
_lib.GetCanonicalState.restype = None


def make_move(
    state: np.ndarray,
    current_player: int,
    piece_idx: int,
    row: int,
    col: int
) -> Tuple[Optional[np.ndarray], bool, Optional[int], bool]:
    """
    Execute a move and return updated state.
    
    Args:
        state: 5x5 int8 numpy array representing board state
        current_player: 1 for WHITE, 2 for BLACK
        piece_idx: 0 for true king, 1 for kinglike
        row: Destination row (0-4)
        col: Destination column (0-4)
    
    Returns:
        Tuple of (new_state, game_over, winner, success)
        - new_state: Updated 5x5 int8 array (None if move invalid)
        - game_over: True if game ended
        - winner: 1 for WHITE, 2 for BLACK, None if no winner
        - success: True if move was valid
    """
    if _lib is None:
        raise RuntimeError("Go engine library not loaded. Build it with 'make build'")
    # Ensure state is correct shape and type
    if state.shape != (5, 5):
        raise ValueError(f"State must be 5x5, got {state.shape}")
    
    state_flat = state.flatten().astype(np.int8)
    new_state_flat = np.zeros(25, dtype=np.int8)
    
    # Create output variables
    game_over = np.array([0], dtype=np.int8)
    winner = np.array([0], dtype=np.int8)
    
    # Get C pointers
    state_ptr = state_flat.ctypes.data_as(_int8_p)
    new_state_ptr = new_state_flat.ctypes.data_as(_int8_p)
    game_over_ptr = game_over.ctypes.data_as(_int8_p)
    winner_ptr = winner.ctypes.data_as(_int8_p)
    
    # Call Go function
    success = _lib.MakeMove(
        state_ptr,
        ctypes.c_int8(current_player),
        ctypes.c_int8(piece_idx),
        ctypes.c_int8(row),
        ctypes.c_int8(col),
        new_state_ptr,
        game_over_ptr,
        winner_ptr,
    )
    
    if success == 0:
        return None, False, None, False
    
    # Reshape to 5x5
    new_state = new_state_flat.reshape(5, 5)
    is_game_over = bool(game_over[0])
    winner_val = int(winner[0]) if is_game_over else None
    
    return new_state, is_game_over, winner_val, True


def get_valid_moves(state: np.ndarray, current_player: int) -> np.ndarray:
    """
    Get action mask for valid moves.
    
    Args:
        state: 5x5 int8 numpy array representing board state
        current_player: 1 for WHITE, 2 for BLACK
    
    Returns:
        50-element boolean numpy array (2 pieces × 25 positions)
    """
    if _lib is None:
        raise RuntimeError("Go engine library not loaded. Build it with 'make build'")
    if state.shape != (5, 5):
        raise ValueError(f"State must be 5x5, got {state.shape}")
    
    state_flat = state.flatten().astype(np.int8)
    mask_flat = np.zeros(50, dtype=np.int8)
    
    # Get C pointers
    state_ptr = state_flat.ctypes.data_as(_int8_p)
    mask_ptr = mask_flat.ctypes.data_as(_int8_p)
    
    # Call Go function
    _lib.GetValidMoves(state_ptr, ctypes.c_int8(current_player), mask_ptr)
    
    # Convert to boolean
    return mask_flat.astype(bool)


def get_canonical_state(state: np.ndarray, current_player: int) -> np.ndarray:
    """
    Convert state to canonical form (from current player's perspective).
    
    Args:
        state: 5x5 int8 numpy array representing board state
        current_player: 1 for WHITE, 2 for BLACK
    
    Returns:
        5x5 float32 numpy array in canonical form
    """
    if _lib is None:
        raise RuntimeError("Go engine library not loaded. Build it with 'make build'")
    if state.shape != (5, 5):
        raise ValueError(f"State must be 5x5, got {state.shape}")
    
    state_flat = state.flatten().astype(np.int8)
    canonical_flat = np.zeros(25, dtype=np.float32)
    
    # Get C pointers
    state_ptr = state_flat.ctypes.data_as(_int8_p)
    canonical_ptr = canonical_flat.ctypes.data_as(_float32_p)
    
    # Call Go function
    _lib.GetCanonicalState(state_ptr, ctypes.c_int8(current_player), canonical_ptr)
    
    # Reshape to 5x5
    return canonical_flat.reshape(5, 5)

