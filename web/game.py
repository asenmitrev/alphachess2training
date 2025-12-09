"""
Knight Capture game implementation for AlphaZero training.
5x5 board with two knights (white and black), goal is to capture the opponent's knight.
"""
import numpy as np
from typing import List, Tuple, Optional
from enum import Enum
import json


class Player(Enum):
    """Player enumeration."""
    EMPTY = 0
    WHITE_KNIGHT = 1
    BLACK_KNIGHT = 2


class KnightCapture:
    """Knight Capture game implementation."""
    
    BOARD_SIZE = 5
    # Knight move offsets (L-shaped moves)
    KNIGHT_MOVES = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1)
    ]
    
    def __init__(self):
        """Initialize board with knights in starting positions."""
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
        # Place white knight at (0, 0) and black knight at (4, 4)
        self.board[0, 0] = Player.WHITE_KNIGHT.value
        self.board[4, 4] = Player.BLACK_KNIGHT.value
        self.white_knight_pos = (0, 0)
        self.black_knight_pos = (4, 4)
        self.current_player = Player.WHITE_KNIGHT
        self.game_over = False
        self.winner = None
        self.move_history = []
    
    def copy(self):
        """Create a deep copy of the game state."""
        new_game = KnightCapture()
        new_game.board = self.board.copy()
        new_game.white_knight_pos = self.white_knight_pos
        new_game.black_knight_pos = self.black_knight_pos
        new_game.current_player = self.current_player
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        new_game.move_history = self.move_history.copy()
        return new_game
    
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Get list of valid knight moves as (row, col) tuples."""
        if self.game_over:
            return []
        
        # Get current knight position
        if self.current_player == Player.WHITE_KNIGHT:
            knight_pos = self.white_knight_pos
        else:
            knight_pos = self.black_knight_pos
        
        moves = []
        knight_row, knight_col = knight_pos
        
        # Check all possible knight moves
        for dr, dc in self.KNIGHT_MOVES:
            new_row = knight_row + dr
            new_col = knight_col + dc
            
            # Check if move is within board bounds
            if 0 <= new_row < self.BOARD_SIZE and 0 <= new_col < self.BOARD_SIZE:
                # Check if destination is empty or contains opponent knight
                if self.board[new_row, new_col] == Player.EMPTY.value:
                    moves.append((new_row, new_col))
                elif self.current_player == Player.WHITE_KNIGHT and \
                     self.board[new_row, new_col] == Player.BLACK_KNIGHT.value:
                    # Can capture opponent knight
                    moves.append((new_row, new_col))
                elif self.current_player == Player.BLACK_KNIGHT and \
                     self.board[new_row, new_col] == Player.WHITE_KNIGHT.value:
                    # Can capture opponent knight
                    moves.append((new_row, new_col))
        
        return moves
    
    async def make_move(self, row: int, col: int) -> bool:
        """
        Make a move at the specified position by sending POST request to /move endpoint.
        Returns True if move was valid, False otherwise.
        """
        if self.game_over:
            return False
        
        # Validate move is in valid moves list
        valid_moves = self.get_valid_moves()
        if (row, col) not in valid_moves:
            return False
        
        # Prepare payload for POST request
        board_list = self.board.tolist()
        payload = {
            "board": board_list,
            "playerMove": {
                "x": col,  # x is column
                "y": row   # y is row
            }
        }
        
        try:
            # Use js.fetch from Pyodide to make POST request
            import js
            response = await js.fetch("/move", {
                "method": "POST",
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": json.dumps(payload)
            })
            
            if not response.ok:
                return False
            
            # Parse response and update board state
            result = await response.json()
            
            # Update board from response
            if "board" in result:
                self.board = np.array(result["board"], dtype=np.int8)
            
            # Update knight positions
            white_pos = None
            black_pos = None
            for r in range(self.BOARD_SIZE):
                for c in range(self.BOARD_SIZE):
                    if self.board[r, c] == Player.WHITE_KNIGHT.value:
                        white_pos = (r, c)
                    elif self.board[r, c] == Player.BLACK_KNIGHT.value:
                        black_pos = (r, c)
            
            if white_pos:
                self.white_knight_pos = white_pos
            if black_pos:
                self.black_knight_pos = black_pos
            
            self.move_history.append((row, col))
            
            # Check for win (capture) - verify if opponent knight still exists
            white_exists = any(self.board[r, c] == Player.WHITE_KNIGHT.value 
                              for r in range(self.BOARD_SIZE) 
                              for c in range(self.BOARD_SIZE))
            black_exists = any(self.board[r, c] == Player.BLACK_KNIGHT.value 
                              for r in range(self.BOARD_SIZE) 
                              for c in range(self.BOARD_SIZE))
            
            if not white_exists:
                self.game_over = True
                self.winner = Player.BLACK_KNIGHT
            elif not black_exists:
                self.game_over = True
                self.winner = Player.WHITE_KNIGHT
            
            # Switch player if game not over
            if not self.game_over:
                self.current_player = Player.BLACK_KNIGHT if self.current_player == Player.WHITE_KNIGHT else Player.WHITE_KNIGHT
            
            return True
            
        except Exception as e:
            print(f"Error making move: {e}")
            return False
    
    def _check_capture(self) -> bool:
        """Check if a knight has been captured."""
        white_exists = any(self.board[r, c] == Player.WHITE_KNIGHT.value 
                          for r in range(self.BOARD_SIZE) 
                          for c in range(self.BOARD_SIZE))
        black_exists = any(self.board[r, c] == Player.BLACK_KNIGHT.value 
                          for r in range(self.BOARD_SIZE) 
                          for c in range(self.BOARD_SIZE))
        
        if not white_exists:
            self.game_over = True
            self.winner = Player.BLACK_KNIGHT
            return True
        elif not black_exists:
            self.game_over = True
            self.winner = Player.WHITE_KNIGHT
            return True
        
        return False
    
    def get_result(self, player: Player) -> Optional[float]:
        """
        Get game result from the perspective of the given player.
        Returns 1.0 for win, -1.0 for loss, 0.0 for draw, None if game not over.
        Note: Using -1.0/0.0/1.0 instead of 0.0/0.5/1.0 for easier backpropagation.
        """
        if not self.game_over:
            return None
        
        if self.winner is None:
            return 0.0  # Draw (shouldn't happen in knight capture)
        
        if self.winner == player:
            return 1.0
        
        return -1.0
    
    def get_state(self) -> np.ndarray:
        """
        Get the current game state as a numpy array.
        Returns a 5x5 array with values: 0 (empty), 1 (white knight), 2 (black knight)
        """
        return self.board.copy()
    
    def get_canonical_state(self) -> np.ndarray:
        """
        Get canonical form of the state (from current player's perspective).
        In canonical form, current player is always 1, opponent is -1.
        """
        state = self.get_state()
        if self.current_player == Player.BLACK_KNIGHT:
            # Flip white and black
            state = np.where(state == Player.WHITE_KNIGHT.value, -1,
                   np.where(state == Player.BLACK_KNIGHT.value, 1, 0))
        else:
            # White is 1, Black is -1
            state = np.where(state == Player.WHITE_KNIGHT.value, 1,
                   np.where(state == Player.BLACK_KNIGHT.value, -1, 0))
        return state
    
    def get_action_mask(self) -> np.ndarray:
        """Get a boolean mask of valid actions (flattened board positions)."""
        mask = np.zeros(self.BOARD_SIZE * self.BOARD_SIZE, dtype=bool)
        valid_moves = self.get_valid_moves()
        for row, col in valid_moves:
            idx = row * self.BOARD_SIZE + col
            mask[idx] = True
        return mask
    
    def action_to_move(self, action: int) -> Tuple[int, int]:
        """Convert action index to (row, col) tuple."""
        row = action // self.BOARD_SIZE
        col = action % self.BOARD_SIZE
        return (row, col)
    
    def move_to_action(self, row: int, col: int) -> int:
        """Convert (row, col) tuple to action index."""
        return row * self.BOARD_SIZE + col
    
    def __str__(self):
        """String representation of the board."""
        symbols = {
            Player.EMPTY.value: '.', 
            Player.WHITE_KNIGHT.value: 'W', 
            Player.BLACK_KNIGHT.value: 'B'
        }
        lines = []
        for row in range(self.BOARD_SIZE):
            line = ' '.join(symbols[self.board[row, col]] for col in range(self.BOARD_SIZE))
            lines.append(line)
        return '\n'.join(lines)

