"""
Python implementation of the JavaScript game engine.
Provides direct function calls instead of HTTP requests for better performance.

Optimizations:
- O(1) position lookups using dictionaries instead of O(n) linear searches
- Cached frequently accessed values to reduce dictionary lookups
- Shallow copy instead of deep copy for state rollback
- Reduced redundant checks and function calls
"""
from typing import Dict, List, Optional, Tuple, Callable, Any


# ============================================================================
# Helper Functions - Optimized with O(1) lookups
# ============================================================================

def _build_square_map(board: List[Dict]) -> Dict[Tuple[int, int], Dict]:
    """Build a position -> square mapping for O(1) lookups."""
    return {(sq.get("x", 0), sq.get("y", 0)): sq for sq in board}


def _build_piece_map(pieces: List[Dict]) -> Dict[Tuple[int, int], Dict]:
    """Build a position -> piece mapping for O(1) lookups."""
    return {(p.get("x", 0), p.get("y", 0)): p for p in pieces if p.get("x") is not None and p.get("y") is not None}


def findSquareByXY(board: List[Dict], x: int, y: int, square_map: Optional[Dict[Tuple[int, int], Dict]] = None) -> Optional[Dict]:
    """Find a square in the board by its x, y coordinates. Uses map if provided for O(1) lookup."""
    if square_map is not None:
        return square_map.get((x, y))
    # Fallback to linear search if map not provided
    for square in board:
        if square.get("x") == x and square.get("y") == y:
            return square
    return None


def pieceFromSquare(square: Dict, pieces: List[Dict], piece_map: Optional[Dict[Tuple[int, int], Dict]] = None) -> Optional[Dict]:
    """Find the piece at a given square's position. Uses map if provided for O(1) lookup."""
    if not square:
        return None
    x = square.get("x")
    y = square.get("y")
    if piece_map is not None:
        return piece_map.get((x, y))
    # Fallback to linear search
    for piece in pieces:
        if piece.get("x") == x and piece.get("y") == y:
            return piece
    return None


def findPieceByXY(pieces: List[Dict], x: int, y: int, piece_map: Optional[Dict[Tuple[int, int], int]] = None) -> int:
    """Find the index of a piece by its x, y coordinates. Returns -1 if not found."""
    if piece_map is not None:
        return piece_map.get((x, y), -1)
    # Fallback to linear search
    for i, piece in enumerate(pieces):
        if piece.get("x") == x and piece.get("y") == y:
            return i
    return -1


def closeLights(board: List[Dict], flag: str = "light") -> None:
    """Reset all square flags (light/allowedMove/etc) to False."""
    if board:
        for square in board:
            square[flag] = False


def giveOppositeColor(color: str) -> str:
    """Return the opposite color. Optimized with direct lookup."""
    # Use dictionary lookup instead of if/elif chain
    _color_map = {"white": "black", "black": "white"}
    return _color_map.get(color, color)


# ============================================================================
# Move Validation (lightBoardFE)
# ============================================================================

def blockableSpecialFunction(
    state: Dict,
    powerX: int,
    powerY: int,
    x: int,
    y: int,
    move: Dict,
    limit: int,
    flag: str,
    secondFlag: Optional[str],
    missedSquareX: int,
    missedSquareY: int,
    minimal: bool,
    operatedPiece: Dict,
    offsetX: int,
    offsetY: int,
    square_map: Optional[Dict[Tuple[int, int], Dict]] = None,
    piece_map: Optional[Dict[Tuple[int, int], Dict]] = None
) -> None:
    """
    Recursive function to handle blockable moves (like rooks, bishops, queens).
    Marks squares as valid moves along a line until blocked.
    Optimized with position maps for O(1) lookups.
    """
    if not flag:
        flag = "light"
    
    if limit == 0:
        return
    
    if missedSquareX is None:
        missedSquareX = 0
    if missedSquareY is None:
        missedSquareY = 0
    
    target_x = powerX + x
    target_y = powerY + y
    square = findSquareByXY(state["board"], target_x, target_y, square_map)
    if not square:
        return
    
    piece = pieceFromSquare(square, state["pieces"], piece_map)
    
    # Cache frequently accessed values
    piece_color = piece.get("color") if piece else None
    operated_color = operatedPiece.get("color")
    friendly_pieces = move.get("friendlyPieces", False)
    impotent = move.get("impotent", False)
    
    # Determine direction (cached)
    directionX = -1 if powerX < 0 else (1 if powerX > 0 else 0)
    directionY = -1 if powerY < 0 else (1 if powerY > 0 else 0)
    
    if not piece:
        # Empty square - mark as valid and continue
        square[flag] = True
        blockableSpecialFunction(
            state=state,
            powerX=powerX + directionX + missedSquareX,
            powerY=powerY + directionY + missedSquareY,
            x=x,
            y=y,
            move=move,
            limit=limit - 1,
            flag=flag,
            secondFlag=secondFlag,
            missedSquareX=missedSquareX,
            missedSquareY=missedSquareY,
            minimal=minimal,
            operatedPiece=operatedPiece,
            offsetX=offsetX,
            offsetY=offsetY,
            square_map=square_map,
            piece_map=piece_map
        )
    elif ((piece_color != operated_color and not friendly_pieces) or
          (piece_color == operated_color and friendly_pieces)) and minimal and not impotent:
        # Can capture this piece in minimal mode
        square[flag] = True
    elif piece and not impotent and not minimal:
        # Check if we can capture or if we should mark as blocked
        # Use map for O(1) lookup instead of linear search
        selected_x = x - offsetX
        selected_y = y - offsetY
        selectedPiece = piece_map.get((selected_x, selected_y)) if piece_map else None
        
        if not selectedPiece:
            # Fallback to linear search if map not available
            for p in state["pieces"]:
                if p.get("x") == selected_x and p.get("y") == selected_y:
                    selectedPiece = p
                    break
        
        if selectedPiece:
            selected_color = selectedPiece.get("color")
            if not ((selected_color == piece_color and not friendly_pieces) or
                    (selected_color != piece_color and friendly_pieces)):
                if secondFlag:
                    square[secondFlag] = True
                square[flag] = True
        
        if secondFlag:
            square[flag] = True
            move_x = move.get("x", 0)
            move_y = move.get("y", 0)
            blockableSpecialFunction(
                state=state,
                powerX=move_x + directionX + missedSquareX,
                powerY=move_y + directionY + missedSquareY,
                x=x,
                y=y,
                move=move,
                limit=limit - 1,
                flag=secondFlag,
                secondFlag=secondFlag,
                missedSquareX=missedSquareX,
                missedSquareY=missedSquareY,
                minimal=minimal,
                operatedPiece=operatedPiece,
                offsetX=offsetX,
                offsetY=offsetY,
                square_map=square_map,
                piece_map=piece_map
            )


def lightBoardFE(
    piece: Dict,
    state: Dict,
    flag: str = "light",
    blockedFlag: Optional[str] = None,
    minimal: bool = False
) -> None:
    """
    Mark valid moves for a piece on the board.
    Sets the 'light' flag (or custom flag) on squares that are valid destinations.
    Optimized with position maps for O(1) lookups.
    """
    if not flag:
        flag = "light"
    
    closeLights(state["board"], flag)
    
    if not piece:
        return
    
    # Build position maps once for all lookups
    square_map = _build_square_map(state["board"])
    piece_map = _build_piece_map(state["pieces"])
    
    # Cache piece values
    piece_x = piece.get("x", 0)
    piece_y = piece.get("y", 0)
    piece_color = piece.get("color")
    piece_icon = piece.get("icon")
    
    # Get moves list
    moves = list(piece.get("moves", []))
    
    # Handle conditional moves if present
    if "conditionalMoves" in piece:
        conditional_moves = piece["conditionalMoves"]
        if callable(conditional_moves):
            temp_moves = conditional_moves(state)
            if temp_moves:
                moves.extend(temp_moves)
    
    # Process each move
    for move in moves:
        move_type = move.get("type")
        impotent = move.get("impotent", False)
        friendly_pieces = move.get("friendlyPieces", False)
        
        if move_type == "absolute":
            # Absolute move: move to relative position
            target_x = piece_x + move.get("x", 0)
            target_y = piece_y + move.get("y", 0)
            square = findSquareByXY(state["board"], target_x, target_y, square_map)
            
            if square:
                inner_piece = pieceFromSquare(square, state["pieces"], piece_map)
                if inner_piece:
                    # Square has a piece
                    inner_color = inner_piece.get("color")
                    if inner_color != piece_color and not impotent:
                        # Enemy piece - can capture
                        check_for_enemies = (inner_color != piece_color and not friendly_pieces and not impotent)
                        check_for_friends = (inner_color == piece_color and friendly_pieces and not impotent)
                        if (check_for_friends or check_for_enemies) and not impotent:
                            square[flag] = True
                    else:
                        # Friendly piece or impotent move - mark as blocked
                        if blockedFlag:
                            square[blockedFlag] = True
                else:
                    # Empty square - valid move
                    square[flag] = True
        
        elif move_type == "allMine":
            # Mark all squares with pieces of same color (except this piece)
            # Use piece_map for faster iteration
            for pos, inner_piece in piece_map.items():
                if (inner_piece.get("color") == piece_color and 
                    inner_piece.get("icon") != piece_icon):
                    square = square_map.get(pos)
                    if square:
                        square[flag] = True
        
        elif move_type == "takeMove":
            # Can only move to this square if capturing
            target_x = piece_x + move.get("x", 0)
            target_y = piece_y + move.get("y", 0)
            square = findSquareByXY(state["board"], target_x, target_y, square_map)
            
            if square:
                inner_piece = pieceFromSquare(square, state["pieces"], piece_map)
                if inner_piece:
                    inner_color = inner_piece.get("color")
                    check_for_enemies = (inner_color != piece_color and not friendly_pieces)
                    check_for_friends = (inner_color == piece_color and friendly_pieces)
                    if (check_for_friends or check_for_enemies) and not impotent:
                        square[flag] = True
                else:
                    # Empty square - mark as blocked
                    if blockedFlag:
                        square[blockedFlag] = True
        
        elif move_type == "blockable":
            # Blockable move (like rook, bishop, queen)
            if move.get("repeat"):
                limit = move.get("limit", 100)
                offsetX = move.get("offsetX", 0)
                offsetY = move.get("offsetY", 0)
                blockableSpecialFunction(
                    state=state,
                    powerX=move.get("x", 0),
                    powerY=move.get("y", 0),
                    x=piece_x + offsetX,
                    y=piece_y + offsetY,
                    move=move,
                    limit=limit,
                    flag=flag,
                    secondFlag=blockedFlag,
                    missedSquareX=move.get("missedSquareX", 0),
                    missedSquareY=move.get("missedSquareY", 0),
                    minimal=minimal,
                    operatedPiece=piece,
                    offsetX=offsetX,
                    offsetY=offsetY,
                    square_map=square_map,
                    piece_map=piece_map
                )


# ============================================================================
# Piece Selection
# ============================================================================

def selectPiece(playerMove: Dict, state: Dict) -> None:
    """
    Select a piece at the given coordinates.
    Sets state.pieceSelected and lights valid moves.
    Optimized with position map for O(1) lookup.
    """
    x = playerMove.get("x")
    y = playerMove.get("y")
    
    # Use piece map for O(1) lookup
    piece_map = _build_piece_map(state["pieces"])
    piece = piece_map.get((x, y))
    
    if not piece:
        # Fallback to linear search if map lookup fails
        for p in state["pieces"]:
            if p.get("x") == x and p.get("y") == y:
                piece = p
                break
    
    if not piece:
        closeLights(state["board"])
        state["pieceSelected"] = None
        return
    
    # Check if it's the correct player's turn
    turn = state.get("turn")
    if piece.get("color") != turn:
        closeLights(state["board"])
        state["pieceSelected"] = None
        return
    
    # Select the piece
    state["pieceSelected"] = piece
    
    # Light valid moves
    lightBoardFE(piece, state, flag="light", blockedFlag=None, minimal=False)


# ============================================================================
# Move Execution
# ============================================================================

def playerMove(
    playerMove: Dict,
    state: Dict,
    alwaysLight: bool = False,
    selectedForced: Optional[Dict] = None,
    specialFlag: Optional[str] = None
) -> bool:
    """
    Execute a player move.
    
    Args:
        playerMove: Dict with 'x' and 'y' keys for destination
        state: Game state dict (will be modified in-place)
        alwaysLight: If True, skip lighting check (move is always valid)
        selectedForced: Force a specific piece to move (overrides state.pieceSelected)
        specialFlag: Flag name to check instead of 'light' (e.g., 'allowedMove')
    
    Returns:
        True if move was successful, False otherwise
    
    Optimizations:
    - Uses position map for O(1) piece lookups
    - Manual state tracking instead of expensive deepcopy
    - Caches frequently accessed values
    """
    light = specialFlag or "light"
    x = playerMove.get("x")
    y = playerMove.get("y")
    
    operatedPiece = selectedForced if selectedForced else state.get("pieceSelected")
    if not operatedPiece:
        return False
    
    # Build position maps for fast lookups
    square_map = _build_square_map(state["board"])
    piece_map = _build_piece_map(state["pieces"])
    
    # Find destination square
    square = findSquareByXY(state["board"], x, y, square_map)
    if not square:
        return False
    
    # Check if square is lighted (valid move)
    if not square.get(light) and not alwaysLight:
        return False
    
    # Find pieces at destination using map
    operated_color = operatedPiece.get("color")
    piece_at_dest = piece_map.get((x, y))
    
    enemyPiece = None
    friendlyPiece = None
    if piece_at_dest:
        if piece_at_dest.get("color") != operated_color:
            enemyPiece = piece_at_dest
        else:
            friendlyPiece = piece_at_dest
    
    # Cache old positions for rollback
    friendlyPieceOldX = friendlyPiece.get("x") if friendlyPiece else None
    friendlyPieceOldY = friendlyPiece.get("y") if friendlyPiece else None
    oldX = operatedPiece.get("x")
    oldY = operatedPiece.get("y")
    
    # Manual state backup (much faster than deepcopy)
    # Only backup what we might need to restore
    old_piece_selected = state.get("pieceSelected")
    old_move = state.get("oldMove")
    pieces_backup = [(p.get("x"), p.get("y")) for p in state["pieces"]]
    
    # Move piece
    operatedPiece["x"] = x
    operatedPiece["y"] = y
    
    # Update piece map for callbacks
    if (oldX, oldY) in piece_map:
        del piece_map[(oldX, oldY)]
    piece_map[(x, y)] = operatedPiece
    
    continueTurn = True
    
    # Check afterPlayerMove callbacks
    pieces = state["pieces"]
    for i in range(len(pieces) - 1, -1, -1):
        piece = pieces[i]
        if "afterPlayerMove" in piece:
            callback = piece["afterPlayerMove"]
            if callable(callback):
                if callback(state, playerMove, {"x": oldX, "y": oldY}):
                    continueTurn = False
                    break
    
    # Check friendlyPieceInteraction
    if continueTurn and operatedPiece.get("friendlyPieceInteraction"):
        callback = operatedPiece["friendlyPieceInteraction"]
        if callable(callback):
            if callback(state, friendlyPiece, {"x": oldX, "y": oldY}):
                if friendlyPiece:
                    friendlyPiece["x"] = friendlyPieceOldX
                    friendlyPiece["y"] = friendlyPieceOldY
                continueTurn = False
    
    if not continueTurn:
        # Rollback: restore piece positions
        operatedPiece["x"] = oldX
        operatedPiece["y"] = oldY
        if friendlyPiece:
            friendlyPiece["x"] = friendlyPieceOldX
            friendlyPiece["y"] = friendlyPieceOldY
        # Restore piece positions from backup
        for i, (backup_x, backup_y) in enumerate(pieces_backup):
            if i < len(pieces):
                pieces[i]["x"] = backup_x
                pieces[i]["y"] = backup_y
        state["pieceSelected"] = old_piece_selected
        state["oldMove"] = old_move
        return False
    
    # Handle enemy piece capture
    if enemyPiece:
        # Check afterThisPieceTaken callback
        if "afterThisPieceTaken" in enemyPiece:
            callback = enemyPiece["afterThisPieceTaken"]
            if callable(callback):
                if callback(state):
                    # Capture was prevented
                    operatedPiece["x"] = oldX
                    operatedPiece["y"] = oldY
                    return False
        
        # Check afterEnemyPieceTaken callback
        if operatedPiece.get("afterEnemyPieceTaken"):
            callback = operatedPiece["afterEnemyPieceTaken"]
            if callable(callback):
                callback(enemyPiece, state)
        
        # Remove enemy piece
        enemyPiece["x"] = None
        enemyPiece["y"] = None
        state["pieces"].remove(enemyPiece)
        # Update piece map
        if (x, y) in piece_map:
            del piece_map[(x, y)]
    
    # Check afterPieceMove callback
    if operatedPiece.get("afterPieceMove"):
        callback = operatedPiece["afterPieceMove"]
        if callable(callback):
            continueTurn = callback(state, playerMove, {"x": oldX, "y": oldY})
            if not continueTurn:
                operatedPiece["x"] = oldX
                operatedPiece["y"] = oldY
                return False
    
    # Update state
    state["oldMove"] = {
        "oldX": oldX,
        "oldY": oldY,
        "currentY": operatedPiece.get("y"),
        "currentX": operatedPiece.get("x")
    }
    
    state["pieceSelected"] = None
    closeLights(state["board"])
    
    # Check afterThisPieceMoves callback (for win conditions)
    if operatedPiece.get("afterThisPieceMoves"):
        callback = operatedPiece["afterThisPieceMoves"]
        if callable(callback):
            callback(state)
    
    return True


# ============================================================================
# Utility Functions
# ============================================================================

def changeTurn(state: Dict) -> None:
    """Change the turn to the opposite player. Optimized with direct lookup."""
    turn = state.get("turn")
    if turn == "white":
        state["turn"] = "black"
    elif turn == "black":
        state["turn"] = "white"


def checkTurn(state: Dict, playerRef: Any) -> bool:
    """
    Check if it's the given player's turn.
    Returns True if it's their turn, False otherwise.
    """
    if ((state.get("turn") == "white" and state.get("white") == playerRef) or
        (state.get("turn") == "black" and state.get("black") == playerRef)):
        return True
    return False

