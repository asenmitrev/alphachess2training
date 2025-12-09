"""
King Capture game implementation for AlphaZero training.
5x5 board with two kings and two kinglikes per player.
Game ends if true king is captured or reaches the end row.

Performance optimizations:
- Uses Numba JIT compilation for hot paths (get_valid_moves, get_action_mask)
- Optimized game state copying to reduce overhead
- Falls back to pure Python if Numba is not available
"""
import numpy as np
from typing import List, Tuple, Optional
from enum import Enum

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


class Piece(Enum):
    """Piece enumeration."""
    EMPTY = 0
    WHITE_KINGLIKE = 1
    WHITE_KING = 2
    BLACK_KINGLIKE = 3
    BLACK_KING = 4


class Player(Enum):
    """Player enumeration."""
    WHITE = 1
    BLACK = 2


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


class KingCapture:
    """King Capture game implementation."""
    
    BOARD_SIZE = 5
    # King move offsets (1 square in any direction)
    KING_MOVES = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    def __init__(self):
        """Initialize board with pieces in starting positions."""
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
        
        # Place white pieces: true king at (4, 2), kinglike at (4, 1)
        self.board[4, 2] = Piece.WHITE_KING.value
        self.board[4, 1] = Piece.WHITE_KINGLIKE.value
        self.white_king_pos = (4, 2)
        self.white_kinglike_pos = (4, 1)
        
        # Place black pieces: true king at (0, 2), kinglike at (0, 1)
        self.board[0, 2] = Piece.BLACK_KING.value
        self.board[0, 1] = Piece.BLACK_KINGLIKE.value
        self.black_king_pos = (0, 2)
        self.black_kinglike_pos = (0, 1)
        
        self.current_player = Player.WHITE
        self.game_over = False
        self.winner = None
        self.move_history = []
    
    def copy(self):
        """Create a deep copy of the game state."""
        # Optimized copy: avoid calling __init__ which does unnecessary initialization
        new_game = object.__new__(KingCapture)
        new_game.board = self.board.copy()  # Use copy() for numpy array
        new_game.white_king_pos = self.white_king_pos
        new_game.white_kinglike_pos = self.white_kinglike_pos
        new_game.black_king_pos = self.black_king_pos
        new_game.black_kinglike_pos = self.black_kinglike_pos
        new_game.current_player = self.current_player
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        new_game.move_history = self.move_history.copy()
        return new_game
    
    def _get_piece_positions(self) -> List[Tuple[int, int, int]]:
        """
        Get all piece positions for current player.
        Returns list of (piece_idx, row, col) tuples.
        piece_idx: 0 = true king, 1 = kinglike
        """
        positions = []
        if self.current_player == Player.WHITE:
            if self.white_king_pos is not None:
                positions.append((0, self.white_king_pos[0], self.white_king_pos[1]))
            if self.white_kinglike_pos is not None:
                positions.append((1, self.white_kinglike_pos[0], self.white_kinglike_pos[1]))
        else:
            if self.black_king_pos is not None:
                positions.append((0, self.black_king_pos[0], self.black_king_pos[1]))
            if self.black_kinglike_pos is not None:
                positions.append((1, self.black_kinglike_pos[0], self.black_kinglike_pos[1]))
        return positions
    
    def get_valid_moves(self) -> List[Tuple[int, int, int]]:
        """
        Get list of valid moves as (piece_idx, row, col) tuples.
        piece_idx: 0 = true king, 1 = kinglike
        """
        if NUMBA_AVAILABLE and not self.game_over:
            # Convert positions to arrays (None -> (-1, -1))
            white_king = np.array(self.white_king_pos if self.white_king_pos else (-1, -1), dtype=np.int32)
            white_kinglike = np.array(self.white_kinglike_pos if self.white_kinglike_pos else (-1, -1), dtype=np.int32)
            black_king = np.array(self.black_king_pos if self.black_king_pos else (-1, -1), dtype=np.int32)
            black_kinglike = np.array(self.black_kinglike_pos if self.black_kinglike_pos else (-1, -1), dtype=np.int32)
            current_player_int = Player.WHITE.value if self.current_player == Player.WHITE else Player.BLACK.value
            
            moves_array = _get_valid_moves_numba(
                self.board, current_player_int, white_king, white_kinglike,
                black_king, black_kinglike, self.game_over
            )
            # Convert numpy array to list of tuples
            return [tuple(move) for move in moves_array]
        
        # Fallback to original implementation
        if self.game_over:
            return []
        
        moves = []
        piece_positions = self._get_piece_positions()
        
        for piece_idx, piece_row, piece_col in piece_positions:
            # Check all possible king moves
            for dr, dc in self.KING_MOVES:
                new_row = piece_row + dr
                new_col = piece_col + dc
                
                # Check if move is within board bounds
                if 0 <= new_row < self.BOARD_SIZE and 0 <= new_col < self.BOARD_SIZE:
                    dest_piece = self.board[new_row, new_col]
                    
                    # Can move to empty square
                    if dest_piece == Piece.EMPTY.value:
                        moves.append((piece_idx, new_row, new_col))
                    # Can capture opponent pieces
                    elif self.current_player == Player.WHITE:
                        if dest_piece == Piece.BLACK_KING.value or dest_piece == Piece.BLACK_KINGLIKE.value:
                            moves.append((piece_idx, new_row, new_col))
                    else:  # BLACK
                        if dest_piece == Piece.WHITE_KING.value or dest_piece == Piece.WHITE_KINGLIKE.value:
                            moves.append((piece_idx, new_row, new_col))
        
        return moves
    
    def make_move(self, piece_idx: int, row: int, col: int) -> bool:
        """
        Make a move with specified piece to the given position.
        
        Args:
            piece_idx: 0 for true king, 1 for kinglike
            row: Destination row
            col: Destination column
        
        Returns:
            True if move was valid, False otherwise
        """
        if self.game_over:
            return False
        
        # Validate move is in valid moves list
        valid_moves = self.get_valid_moves()
        if (piece_idx, row, col) not in valid_moves:
            return False
        
        # Get current piece position
        if self.current_player == Player.WHITE:
            if piece_idx == 0:
                old_pos = self.white_king_pos
                piece_type = Piece.WHITE_KING
            else:
                old_pos = self.white_kinglike_pos
                piece_type = Piece.WHITE_KINGLIKE
        else:  # BLACK
            if piece_idx == 0:
                old_pos = self.black_king_pos
                piece_type = Piece.BLACK_KING
            else:
                old_pos = self.black_kinglike_pos
                piece_type = Piece.BLACK_KINGLIKE
        
        # Check what's at destination
        dest_piece_value = self.board[row, col]
        captured_king = False
        
        # Check if capturing opponent's true king
        if self.current_player == Player.WHITE:
            if dest_piece_value == Piece.BLACK_KING.value:
                captured_king = True
                self.black_king_pos = None
            elif dest_piece_value == Piece.BLACK_KINGLIKE.value:
                self.black_kinglike_pos = None
        else:  # BLACK
            if dest_piece_value == Piece.WHITE_KING.value:
                captured_king = True
                self.white_king_pos = None
            elif dest_piece_value == Piece.WHITE_KINGLIKE.value:
                self.white_kinglike_pos = None
        
        # Clear old position
        self.board[old_pos[0], old_pos[1]] = Piece.EMPTY.value
        
        # Place piece at new position
        self.board[row, col] = piece_type.value
        
        # Update piece position
        if self.current_player == Player.WHITE:
            if piece_idx == 0:
                self.white_king_pos = (row, col)
            else:
                self.white_kinglike_pos = (row, col)
        else:  # BLACK
            if piece_idx == 0:
                self.black_king_pos = (row, col)
            else:
                self.black_kinglike_pos = (row, col)
        
        self.move_history.append((piece_idx, row, col))
        
        # Check win conditions
        # 1. True king captured
        if captured_king:
            self.game_over = True
            self.winner = self.current_player
        # 2. True king reached end row
        elif piece_idx == 0:  # Moving true king
            if self.current_player == Player.WHITE and row == 0:
                # White king reached top (black's starting row)
                self.game_over = True
                self.winner = Player.WHITE
            elif self.current_player == Player.BLACK and row == 4:
                # Black king reached bottom (white's starting row)
                self.game_over = True
                self.winner = Player.BLACK
            else:
                # True king moved but didn't reach end row - switch player
                self.current_player = Player.BLACK if self.current_player == Player.WHITE else Player.WHITE
        else:
            # Kinglike moved - switch player
            self.current_player = Player.BLACK if self.current_player == Player.WHITE else Player.WHITE
        
        return True
    
    def get_result(self, player: Player) -> Optional[float]:
        """
        Get game result from the perspective of the given player.
        Returns 1.0 for win, -1.0 for loss, 0.0 for draw, None if game not over.
        """
        if not self.game_over:
            return None
        
        if self.winner is None:
            return 0.0  # Draw (shouldn't happen)
        
        if self.winner == player:
            return 1.0
        
        return -1.0
    
    def get_state(self) -> np.ndarray:
        """
        Get the current game state as a numpy array.
        Returns a 5x5 array with piece values.
        """
        return self.board.copy()
    
    def get_canonical_state(self) -> np.ndarray:
        """
        Get canonical form of the state (from current player's perspective).
        In canonical form:
        - Current player's true king = 1.0
        - Current player's kinglike = 0.5
        - Opponent's true king = -1.0
        - Opponent's kinglike = -0.5
        - Empty = 0.0
        """
        state = self.get_state().astype(np.float32)
        
        if self.current_player == Player.BLACK:
            # Flip perspective: black becomes positive, white becomes negative
            state = np.where(state == Piece.BLACK_KING.value, 1.0,
                   np.where(state == Piece.BLACK_KINGLIKE.value, 0.5,
                   np.where(state == Piece.WHITE_KING.value, -1.0,
                   np.where(state == Piece.WHITE_KINGLIKE.value, -0.5, 0.0))))
        else:  # WHITE
            # White is positive, black is negative
            state = np.where(state == Piece.WHITE_KING.value, 1.0,
                   np.where(state == Piece.WHITE_KINGLIKE.value, 0.5,
                   np.where(state == Piece.BLACK_KING.value, -1.0,
                   np.where(state == Piece.BLACK_KINGLIKE.value, -0.5, 0.0))))
        
        return state
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get a boolean mask of valid actions.
        Action space: 2 pieces * BOARD_SIZE * BOARD_SIZE = 2 * 25 = 50 actions
        Format: [piece0_actions, piece1_actions] flattened
        """
        if NUMBA_AVAILABLE:
            # Convert positions to arrays (None -> (-1, -1))
            white_king = np.array(self.white_king_pos if self.white_king_pos else (-1, -1), dtype=np.int32)
            white_kinglike = np.array(self.white_kinglike_pos if self.white_kinglike_pos else (-1, -1), dtype=np.int32)
            black_king = np.array(self.black_king_pos if self.black_king_pos else (-1, -1), dtype=np.int32)
            black_kinglike = np.array(self.black_kinglike_pos if self.black_kinglike_pos else (-1, -1), dtype=np.int32)
            current_player_int = Player.WHITE.value if self.current_player == Player.WHITE else Player.BLACK.value
            
            mask = _get_action_mask_numba(
                self.board, current_player_int, white_king, white_kinglike,
                black_king, black_kinglike, self.game_over
            )
            return mask
        
        # Fallback to original implementation
        mask = np.zeros(2 * self.BOARD_SIZE * self.BOARD_SIZE, dtype=bool)
        valid_moves = self.get_valid_moves()
        
        for piece_idx, row, col in valid_moves:
            # Action index: piece_idx * BOARD_SIZE^2 + row * BOARD_SIZE + col
            action_idx = piece_idx * self.BOARD_SIZE * self.BOARD_SIZE + row * self.BOARD_SIZE + col
            mask[action_idx] = True
        
        return mask
    
    def action_to_move(self, action: int) -> Tuple[int, int, int]:
        """
        Convert action index to (piece_idx, row, col) tuple.
        
        Args:
            action: Action index (0 to 49)
        
        Returns:
            (piece_idx, row, col) tuple
        """
        piece_idx = action // (self.BOARD_SIZE * self.BOARD_SIZE)
        remaining = action % (self.BOARD_SIZE * self.BOARD_SIZE)
        row = remaining // self.BOARD_SIZE
        col = remaining % self.BOARD_SIZE
        return (piece_idx, row, col)
    
    def move_to_action(self, piece_idx: int, row: int, col: int) -> int:
        """
        Convert (piece_idx, row, col) tuple to action index.
        
        Args:
            piece_idx: 0 for true king, 1 for kinglike
            row: Row
            col: Column
        
        Returns:
            Action index
        """
        return piece_idx * self.BOARD_SIZE * self.BOARD_SIZE + row * self.BOARD_SIZE + col
    
    def __str__(self):
        """String representation of the board."""
        symbols = {
            Piece.EMPTY.value: '.',
            Piece.WHITE_KINGLIKE.value: 'w',
            Piece.WHITE_KING.value: 'W',
            Piece.BLACK_KINGLIKE.value: 'b',
            Piece.BLACK_KING.value: 'B'
        }
        lines = []
        for row in range(self.BOARD_SIZE):
            line = ' '.join(symbols[self.board[row, col]] for col in range(self.BOARD_SIZE))
            lines.append(line)
        return '\n'.join(lines)
