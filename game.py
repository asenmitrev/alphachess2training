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
import requests
import logging
from typing import List, Tuple, Optional
from enum import Enum

logger = logging.getLogger(__name__)


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


class KingCapture:
    """King Capture game implementation."""
    
    BOARD_SIZE = 5
    SERVER_URL = "http://localhost:8085/gameTester"
    USE_SERVER = True  # Set to False to use local implementation
    
    # Shared HTTP session for connection pooling (one per process)
    _http_session = None
    
    # King move offsets (1 square in any direction)
    KING_MOVES = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    # Precomputed moves in server format
    _SERVER_KING_MOVES = [
        {"type": "absolute", "x": 0, "y": 1},
        {"type": "absolute", "x": 1, "y": 0},
        {"type": "absolute", "x": 1, "y": 1},
        {"type": "absolute", "x": -1, "y": -1},
        {"type": "absolute", "x": 0, "y": -1},
        {"type": "absolute", "x": -1, "y": 0},
        {"type": "absolute", "x": -1, "y": 1},
        {"type": "absolute", "x": 1, "y": -1}
    ]
    
    @classmethod
    def _get_session(cls):
        """Get or create HTTP session for connection pooling."""
        if cls._http_session is None:
            cls._http_session = requests.Session()
            # Configure session for better performance
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=1,  # Single connection pool
                pool_maxsize=10,      # Max connections in pool
                max_retries=0        # No retries (fail fast)
            )
            cls._http_session.mount('http://', adapter)
            cls._http_session.mount('https://', adapter)
        return cls._http_session
    
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
    
    def _state_to_server_json(self, piece_idx: int, target_row: int, target_col: int) -> dict:
        """
        Convert game state to server JSON format.
        
        Args:
            piece_idx: Index of the piece being moved (0=king, 1=kinglike)
            target_row: Target row for the move
            target_col: Target column for the move
        
        Returns:
            Dictionary in server-expected format
        """
        pieces = []
        
        # Add black king
        if self.black_king_pos is not None:
            pieces.append({
                "icon": "blackShroom.png",
                "moves": self._SERVER_KING_MOVES.copy(),
                "color": "black",
                "x": int(self.black_king_pos[1]),  # col -> x
                "y": int(self.black_king_pos[0]),  # row -> y
                "afterThisPieceTaken": "state.won = 1",
                "afterThisPieceMoves": f"if(this.y === {self.BOARD_SIZE - 1}){{this.value = 2000;state.won = 2}}"
            })
        
        # Add black kinglike
        if self.black_kinglike_pos is not None:
            pieces.append({
                "icon": "blackKing.png",
                "moves": self._SERVER_KING_MOVES.copy(),
                "color": "black",
                "x": int(self.black_kinglike_pos[1]),
                "y": int(self.black_kinglike_pos[0])
            })
        
        # Add white king
        if self.white_king_pos is not None:
            pieces.append({
                "icon": "whiteKing.png",
                "moves": self._SERVER_KING_MOVES.copy(),
                "color": "white",
                "x": int(self.white_king_pos[1]),
                "y": int(self.white_king_pos[0]),
                "afterThisPieceTaken": "state.won = 2",
                "afterThisPieceMoves": "if(this.y === 0){this.value = 2000;state.won = 1}",
                "value": 2.5,
                "posValue": 2
            })
        
        # Add white kinglike
        if self.white_kinglike_pos is not None:
            pieces.append({
                "icon": "whiteKing.png",
                "moves": self._SERVER_KING_MOVES.copy(),
                "color": "white",
                "x": int(self.white_kinglike_pos[1]),
                "y": int(self.white_kinglike_pos[0]),
                "value": 2.5,
                "posValue": 2
            })
        
        # Create board cells
        board = []
        for y in range(self.BOARD_SIZE):
            for x in range(self.BOARD_SIZE):
                board.append({
                    "x": x,
                    "y": y,
                    "allowedMove": False,
                    "light": False
                })
        
        # Get the piece being moved
        if self.current_player == Player.WHITE:
            piece_pos = self.white_king_pos if piece_idx == 0 else self.white_kinglike_pos
        else:
            piece_pos = self.black_king_pos if piece_idx == 0 else self.black_kinglike_pos
        
        return {
            "state": {
                "pieces": pieces,
                "board": board,
                "turn": "white" if self.current_player == Player.WHITE else "black"
            },
            "pieceAt": {
                "x": int(piece_pos[1]),  # col -> x
                "y": int(piece_pos[0])   # row -> y
            },
            "playerMove": {
                "x": int(target_col),
                "y": int(target_row)
            }
        }
    
    def _check_win_conditions(self, piece_idx: int, previous_player: Player):
        """
        Check win conditions after a move based on current board state.
        
        Args:
            piece_idx: Index of piece that was moved (0=king, 1=kinglike)
            previous_player: Player who just made the move
        """
        # Check if a king was captured
        if previous_player == Player.WHITE:
            if self.black_king_pos is None:
                # White captured black king
                self.game_over = True
                self.winner = Player.WHITE
                return
            # Check if white king reached end row (only if true king was moved)
            if piece_idx == 0 and self.white_king_pos is not None:
                if self.white_king_pos[0] == 0:  # White king reached top row (row 0)
                    self.game_over = True
                    self.winner = Player.WHITE
                    return
        else:  # BLACK
            if self.white_king_pos is None:
                # Black captured white king
                self.game_over = True
                self.winner = Player.BLACK
                return
            # Check if black king reached end row (only if true king was moved)
            if piece_idx == 0 and self.black_king_pos is not None:
                if self.black_king_pos[0] == 4:  # Black king reached bottom row (row 4)
                    self.game_over = True
                    self.winner = Player.BLACK
                    return
    
    def _parse_server_response(self, response: dict, piece_idx: int = None, target_row: int = None, target_col: int = None):
        """
        Parse server response and update game state.
        
        Args:
            response: Server response dictionary containing the new game state
            piece_idx: Index of piece that was moved (for win condition fallback)
            target_row: Target row of the move (for piece identification)
            target_col: Target column of the move (for piece identification)
        """
        state = response.get("state", response)
        
        # Store previous player and positions before updating turn
        previous_player = self.current_player
        prev_white_king_pos = self.white_king_pos
        prev_white_kinglike_pos = self.white_kinglike_pos
        
        # Reset board
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
        self.white_king_pos = None
        self.white_kinglike_pos = None
        self.black_king_pos = None
        self.black_kinglike_pos = None
        
        # Process pieces from server response
        # Track which pieces we've assigned to detect duplicates
        white_king_found = False
        white_kinglike_found = False
        black_king_found = False
        black_kinglike_found = False
        
        for piece in state["pieces"]:
            x, y = piece["x"], piece["y"]
            color = piece["color"]
            icon = piece.get("icon", "")
            
            # Identify king vs kinglike:
            # - For black: "blackShroom.png" = king, "blackKing.png" = kinglike (server preserves icon)
            # - For white: server strips "afterThisPieceTaken", so we track by position matching
            if color == "white":
                # Skip white pieces here - we'll process them separately after collecting all pieces
                continue
            else:  # black
                # Use icon to identify: blackShroom.png = king, blackKing.png = kinglike
                is_king = icon == "blackShroom.png"
                if is_king:
                    if black_king_found:
                        print(f"Warning: Multiple black kings found in server response at ({x}, {y})")
                    self.board[y, x] = Piece.BLACK_KING.value
                    self.black_king_pos = (y, x)
                    black_king_found = True
                else:
                    if black_kinglike_found:
                        print(f"Warning: Multiple black kinglikes found in server response at ({x}, {y})")
                    self.board[y, x] = Piece.BLACK_KINGLIKE.value
                    self.black_kinglike_pos = (y, x)
                    black_kinglike_found = True
        
        # Process white pieces (server strips afterThisPieceTaken, so track by position matching)
        white_pieces = [(piece["x"], piece["y"], piece.get("icon", ""), piece) 
                       for piece in state["pieces"] if piece["color"] == "white"]
        
        # Identify white pieces using the move information
        # If white just moved, we know which piece moved and where
        if previous_player == Player.WHITE and piece_idx is not None and target_row is not None and target_col is not None:
            # Find the piece at the target position (this is the piece that moved)
            moved_piece_pos = None
            for x, y, icon, piece_data in white_pieces:
                if y == target_row and x == target_col:
                    moved_piece_pos = (x, y)
                    # This is the piece that moved - identify it by piece_idx
                    if piece_idx == 0:  # King moved
                        if white_king_found:
                            print(f"Warning: Multiple white kings found in server response at ({x}, {y})")
                        self.board[y, x] = Piece.WHITE_KING.value
                        self.white_king_pos = (y, x)
                        white_king_found = True
                    else:  # Kinglike moved
                        if white_kinglike_found:
                            print(f"Warning: Multiple white kinglikes found in server response at ({x}, {y})")
                        self.board[y, x] = Piece.WHITE_KINGLIKE.value
                        self.white_kinglike_pos = (y, x)
                        white_kinglike_found = True
                    break
            
            # Process the other white piece (the one that didn't move)
            for x, y, icon, piece_data in white_pieces:
                if (y, x) != (target_row, target_col):  # This is the piece that didn't move
                    if not white_king_found:  # If king not found yet, this must be it
                        self.board[y, x] = Piece.WHITE_KING.value
                        self.white_king_pos = (y, x)
                        white_king_found = True
                    elif not white_kinglike_found:  # Otherwise it's the kinglike
                        self.board[y, x] = Piece.WHITE_KINGLIKE.value
                        self.white_kinglike_pos = (y, x)
                        white_kinglike_found = True
        else:
            # Fallback: Use position matching with previous positions
            # This handles cases where black moved or we don't have move info
            # First pass: try to match pieces by their previous positions
            unmatched_pieces = []
            for x, y, icon, piece_data in white_pieces:
                matched = False
                # Try to match by previous position
                if prev_white_king_pos is not None and (y, x) == prev_white_king_pos:
                    if white_king_found:
                        print(f"Warning: Multiple white kings found in server response at ({x}, {y})")
                    self.board[y, x] = Piece.WHITE_KING.value
                    self.white_king_pos = (y, x)
                    white_king_found = True
                    matched = True
                elif prev_white_kinglike_pos is not None and (y, x) == prev_white_kinglike_pos:
                    if white_kinglike_found:
                        print(f"Warning: Multiple white kinglikes found in server response at ({x}, {y})")
                    self.board[y, x] = Piece.WHITE_KINGLIKE.value
                    self.white_kinglike_pos = (y, x)
                    white_kinglike_found = True
                    matched = True
                
                if not matched:
                    unmatched_pieces.append((x, y, icon, piece_data))
            
            # Second pass: assign unmatched pieces (these moved or are new)
            # Assign based on what's missing, prioritizing king
            for x, y, icon, piece_data in unmatched_pieces:
                if not white_king_found:
                    self.board[y, x] = Piece.WHITE_KING.value
                    self.white_king_pos = (y, x)
                    white_king_found = True
                elif not white_kinglike_found:
                    self.board[y, x] = Piece.WHITE_KINGLIKE.value
                    self.white_kinglike_pos = (y, x)
                    white_kinglike_found = True
                else:
                    print(f"Warning: Extra white piece found at ({x}, {y}) but both king and kinglike already assigned")
        
        # Update current player turn (server response has already advanced the turn)
        if "turn" in state:
            self.current_player = Player.WHITE if state["turn"] == "white" else Player.BLACK
        else:
            # Fallback: advance turn manually
            self.current_player = Player.BLACK if previous_player == Player.WHITE else Player.WHITE

        # Check for game over (won field in state)
        if "won" in state:
            self.game_over = True
            self.winner = Player.WHITE if state["won"] == 1 else Player.BLACK
        else:
            # Fallback: Use the old method if piece_idx is available
            # (This handles edge cases where position-based check might miss something)
            self._check_win_conditions(piece_idx, previous_player)
    
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
        
        # Use server if enabled
        if self.USE_SERVER:
            return self._make_move_via_server(piece_idx, row, col)
        
        return self._make_move_local(piece_idx, row, col)
    
    def _make_move_via_server(self, piece_idx: int, row: int, col: int) -> bool:
        """
        Make a move by calling the game server.
        
        Args:
            piece_idx: 0 for true king, 1 for kinglike
            row: Destination row
            col: Destination column
        
        Returns:
            True if move was successful, False otherwise
        """
        try:
            # Convert state to server format
            request_data = self._state_to_server_json(piece_idx, row, col)
            
            # Make HTTP request to server using connection pooling
            session = self._get_session()
            response = session.post(
                self.SERVER_URL,
                json=request_data,
                timeout=1.0  # Reduced timeout since we're using localhost
            )
            response.raise_for_status()
            
            # Parse response and update state (pass target position for piece identification)
            server_response = response.json()
            
            self._parse_server_response(server_response, piece_idx=piece_idx, target_row=row, target_col=col)
            
            # Add move to history
            self.move_history.append((piece_idx, row, col))
            
            return True
            
        except requests.RequestException as e:
            # If server fails, fall back to local implementation
            print(f"Server request failed: {e}. Falling back to local implementation.")
            return self._make_move_local(piece_idx, row, col)
    
    def _make_move_local(self, piece_idx: int, row: int, col: int) -> bool:
        """
        Make a move using local implementation (fallback or when server is disabled).
        
        Args:
            piece_idx: 0 for true king, 1 for kinglike
            row: Destination row
            col: Destination column
        
        Returns:
            True if move was valid, False otherwise
        """
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
            winner_name = "White" if self.winner == Player.WHITE else "Black"
            logger.info(f"GAME OVER: {winner_name} wins by capturing opponent's king. Moves: {len(self.move_history)}")
        # 2. True king reached end row
        elif piece_idx == 0:  # Moving true king
            if self.current_player == Player.WHITE and row == 0:
                # White king reached top (black's starting row)
                self.game_over = True
                self.winner = Player.WHITE
                logger.info(f"GAME OVER: White wins - king reached end row (row 0). Moves: {len(self.move_history)}")
            elif self.current_player == Player.BLACK and row == 4:
                # Black king reached bottom (white's starting row)
                self.game_over = True
                self.winner = Player.BLACK
                logger.info(f"GAME OVER: Black wins - king reached end row (row 4). Moves: {len(self.move_history)}")

        # IMPORTANT: Always advance turn after a move, even if the game ended.
        # MCTS/backprop assumes `current_player` is the side-to-move at this state.
        # `winner` captures the side that won the terminal move.
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
        
        Also flips the board spatially if current player is BLACK,
        so that current player always plays from bottom (rows increasing).
        """
        state = self.get_state().astype(np.float32)
        
        if self.current_player == Player.BLACK:
            # Flip perspective: black becomes positive, white becomes negative
            state = np.where(state == Piece.BLACK_KING.value, 1.0,
                   np.where(state == Piece.BLACK_KINGLIKE.value, 0.5,
                   np.where(state == Piece.WHITE_KING.value, -1.0,
                   np.where(state == Piece.WHITE_KINGLIKE.value, -0.5, 0.0))))
            # Flip spatially: row 0 becomes row 4
            state = np.flip(state, axis=0).copy()
        else:  # WHITE
            # White is positive, black is negative
            state = np.where(state == Piece.WHITE_KING.value, 1.0,
                   np.where(state == Piece.WHITE_KINGLIKE.value, 0.5,
                   np.where(state == Piece.BLACK_KING.value, -1.0,
                   np.where(state == Piece.BLACK_KINGLIKE.value, -0.5, 0.0))))
        
        return state
    
    def flip_policy(self, policy: np.ndarray) -> np.ndarray:
        """
        Flip the policy spatially (rows 0<->4, 1<->3).
        Used to convert network output (canonical view) back to real board actions,
        or vice versa.
        """
        flipped_policy = np.zeros_like(policy)
        
        # Action space: 2 pieces * 25 positions
        for piece_idx in range(2):
            for row in range(self.BOARD_SIZE):
                for col in range(self.BOARD_SIZE):
                    action = piece_idx * self.BOARD_SIZE * self.BOARD_SIZE + row * self.BOARD_SIZE + col
                    flipped_row = self.BOARD_SIZE - 1 - row
                    flipped_action = piece_idx * self.BOARD_SIZE * self.BOARD_SIZE + flipped_row * self.BOARD_SIZE + col
                    
                    flipped_policy[flipped_action] = policy[action]
                    
        return flipped_policy

    
    def get_action_mask(self) -> np.ndarray:
        """
        Get a boolean mask of valid actions.
        Action space: 2 pieces * BOARD_SIZE * BOARD_SIZE = 2 * 25 = 50 actions
        Format: [piece0_actions, piece1_actions] flattened
        """

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
