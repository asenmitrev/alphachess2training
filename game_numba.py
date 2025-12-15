"""
Numba-optimized functions for game logic.
Provides JIT-compiled versions of hot paths for better performance.
"""
import numpy as np

# Try to import numba for JIT compilation, fallback if not available
try:
    from numba import jit, types
    from numba.typed import List as NumbaList
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create a no-op decorator if numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# Numba-optimized helper functions for hot paths
if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def _get_valid_moves_numba(board, current_player_int, white_king_pos, white_kinglike_pos, 
                                black_king_pos, black_kinglike_pos, game_over):
        """
        Numba-optimized version of get_valid_moves.
        Returns moves as a numpy array of shape (N, 3) where each row is (piece_idx, row, col).
        """
        BOARD_SIZE = 5
        MAX_MOVES = 16  # Maximum possible moves (2 pieces * 8 directions)
        moves = np.zeros((MAX_MOVES, 3), dtype=np.int32)
        move_count = 0
        
        if game_over:
            return moves[:0]  # Return empty array
        
        KING_MOVES = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)], dtype=np.int32)
        
        # Piece constants
        EMPTY = 0
        WHITE_KINGLIKE = 1
        WHITE_KING = 2
        BLACK_KINGLIKE = 3
        BLACK_KING = 4
        WHITE = 1
        BLACK = 2
        
        # Get piece positions for current player - process directly
        # Process white pieces
        if current_player_int == WHITE:
            # White king
            if white_king_pos[0] >= 0:
                piece_idx = 0
                piece_row = white_king_pos[0]
                piece_col = white_king_pos[1]
                for j in range(len(KING_MOVES)):
                    dr = KING_MOVES[j, 0]
                    dc = KING_MOVES[j, 1]
                    new_row = piece_row + dr
                    new_col = piece_col + dc
                    
                    if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
                        dest_piece = board[new_row, new_col]
                        
                        if dest_piece == EMPTY:
                            moves[move_count, 0] = piece_idx
                            moves[move_count, 1] = new_row
                            moves[move_count, 2] = new_col
                            move_count += 1
                        elif dest_piece == BLACK_KING or dest_piece == BLACK_KINGLIKE:
                            moves[move_count, 0] = piece_idx
                            moves[move_count, 1] = new_row
                            moves[move_count, 2] = new_col
                            move_count += 1
            
            # White kinglike
            if white_kinglike_pos[0] >= 0:
                piece_idx = 1
                piece_row = white_kinglike_pos[0]
                piece_col = white_kinglike_pos[1]
                for j in range(len(KING_MOVES)):
                    dr = KING_MOVES[j, 0]
                    dc = KING_MOVES[j, 1]
                    new_row = piece_row + dr
                    new_col = piece_col + dc
                    
                    if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
                        dest_piece = board[new_row, new_col]
                        
                        if dest_piece == EMPTY:
                            moves[move_count, 0] = piece_idx
                            moves[move_count, 1] = new_row
                            moves[move_count, 2] = new_col
                            move_count += 1
                        elif dest_piece == BLACK_KING or dest_piece == BLACK_KINGLIKE:
                            moves[move_count, 0] = piece_idx
                            moves[move_count, 1] = new_row
                            moves[move_count, 2] = new_col
                            move_count += 1
        else:  # BLACK
            # Black king
            if black_king_pos[0] >= 0:
                piece_idx = 0
                piece_row = black_king_pos[0]
                piece_col = black_king_pos[1]
                for j in range(len(KING_MOVES)):
                    dr = KING_MOVES[j, 0]
                    dc = KING_MOVES[j, 1]
                    new_row = piece_row + dr
                    new_col = piece_col + dc
                    
                    if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
                        dest_piece = board[new_row, new_col]
                        
                        if dest_piece == EMPTY:
                            moves[move_count, 0] = piece_idx
                            moves[move_count, 1] = new_row
                            moves[move_count, 2] = new_col
                            move_count += 1
                        elif dest_piece == WHITE_KING or dest_piece == WHITE_KINGLIKE:
                            moves[move_count, 0] = piece_idx
                            moves[move_count, 1] = new_row
                            moves[move_count, 2] = new_col
                            move_count += 1
            
            # Black kinglike
            if black_kinglike_pos[0] >= 0:
                piece_idx = 1
                piece_row = black_kinglike_pos[0]
                piece_col = black_kinglike_pos[1]
                for j in range(len(KING_MOVES)):
                    dr = KING_MOVES[j, 0]
                    dc = KING_MOVES[j, 1]
                    new_row = piece_row + dr
                    new_col = piece_col + dc
                    
                    if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
                        dest_piece = board[new_row, new_col]
                        
                        if dest_piece == EMPTY:
                            moves[move_count, 0] = piece_idx
                            moves[move_count, 1] = new_row
                            moves[move_count, 2] = new_col
                            move_count += 1
                        elif dest_piece == WHITE_KING or dest_piece == WHITE_KINGLIKE:
                            moves[move_count, 0] = piece_idx
                            moves[move_count, 1] = new_row
                            moves[move_count, 2] = new_col
                            move_count += 1
        
        
        return moves[:move_count]
    
    @jit(nopython=True, cache=True)
    def _get_action_mask_numba(board, current_player_int, white_king_pos, white_kinglike_pos,
                               black_king_pos, black_kinglike_pos, game_over):
        """Numba-optimized version of get_action_mask."""
        BOARD_SIZE = 5
        mask = np.zeros(2 * BOARD_SIZE * BOARD_SIZE, dtype=np.bool_)
        
        if game_over:
            return mask
        
        moves = _get_valid_moves_numba(board, current_player_int, white_king_pos, white_kinglike_pos,
                                       black_king_pos, black_kinglike_pos, game_over)
        
        for i in range(moves.shape[0]):
            piece_idx = moves[i, 0]
            row = moves[i, 1]
            col = moves[i, 2]
            action_idx = piece_idx * BOARD_SIZE * BOARD_SIZE + row * BOARD_SIZE + col
            mask[action_idx] = True
        
        return mask
else:
    # Fallback functions if numba is not available
    def _get_valid_moves_numba(*args, **kwargs):
        raise RuntimeError("Numba not available")
    
    def _get_action_mask_numba(*args, **kwargs):
        raise RuntimeError("Numba not available")

