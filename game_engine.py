"""
Python implementation of the JavaScript game engine.
Provides direct function calls instead of HTTP requests for better performance.
"""
from typing import Dict, List, Optional, Tuple, Callable, Any


# ============================================================================
# Helper Functions
# ============================================================================

def findSquareByXY(board: List[Dict], x: int, y: int) -> Optional[Dict]:
    """Find a square in the board by its x, y coordinates."""
    for square in board:
        if square.get("x") == x and square.get("y") == y:
            return square
    return None


def pieceFromSquare(square: Dict, pieces: List[Dict]) -> Optional[Dict]:
    """Find the piece at a given square's position."""
    if not square:
        return None
    x = square.get("x")
    y = square.get("y")
    for piece in pieces:
        if piece.get("x") == x and piece.get("y") == y:
            return piece
    return None


def findPieceByXY(pieces: List[Dict], x: int, y: int) -> int:
    """Find the index of a piece by its x, y coordinates. Returns -1 if not found."""
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
    """Return the opposite color."""
    if color == "white":
        return "black"
    elif color == "black":
        return "white"
    return color


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
    offsetY: int
) -> None:
    """
    Recursive function to handle blockable moves (like rooks, bishops, queens).
    Marks squares as valid moves along a line until blocked.
    """
    if not flag:
        flag = "light"
    
    if limit == 0:
        return
    
    if missedSquareX is None:
        missedSquareX = 0
    if missedSquareY is None:
        missedSquareY = 0
    
    square = findSquareByXY(state["board"], powerX + x, powerY + y)
    if not square:
        return
    
    piece = pieceFromSquare(square, state["pieces"])
    
    # Determine direction
    directionX = 0
    if powerX < 0:
        directionX = -1
    elif powerX > 0:
        directionX = 1
    
    directionY = 0
    if powerY < 0:
        directionY = -1
    elif powerY > 0:
        directionY = 1
    
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
            offsetY=offsetY
        )
    elif ((piece.get("color") != operatedPiece.get("color") and not move.get("friendlyPieces")) or
          (piece.get("color") == operatedPiece.get("color") and move.get("friendlyPieces"))) and minimal and not move.get("impotent"):
        # Can capture this piece in minimal mode
        square[flag] = True
    elif piece and not move.get("impotent") and not minimal:
        # Check if we can capture or if we should mark as blocked
        selectedPiece = None
        for p in state["pieces"]:
            if p.get("x") == x - offsetX and p.get("y") == y - offsetY:
                selectedPiece = p
                break
        
        if selectedPiece:
            if not ((selectedPiece.get("color") == piece.get("color") and not move.get("friendlyPieces")) or
                    (selectedPiece.get("color") != piece.get("color") and move.get("friendlyPieces"))):
                if secondFlag:
                    square[secondFlag] = True
                square[flag] = True
        
        if secondFlag:
            square[flag] = True
            blockableSpecialFunction(
                state=state,
                powerX=move.get("x", 0) + directionX + missedSquareX,
                powerY=move.get("y", 0) + directionY + missedSquareY,
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
                offsetY=offsetY
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
    """
    if not flag:
        flag = "light"
    
    closeLights(state["board"], flag)
    
    if not piece:
        return
    
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
        
        if move_type == "absolute":
            # Absolute move: move to relative position
            target_x = piece.get("x") + move.get("x", 0)
            target_y = piece.get("y") + move.get("y", 0)
            square = findSquareByXY(state["board"], target_x, target_y)
            
            if square:
                inner_piece = pieceFromSquare(square, state["pieces"])
                if inner_piece:
                    # Square has a piece
                    if inner_piece.get("color") != piece.get("color") and not move.get("impotent"):
                        # Enemy piece - can capture
                        check_for_enemies = (inner_piece.get("color") != piece.get("color") and 
                                          not move.get("friendlyPieces") and 
                                          not move.get("impotent"))
                        check_for_friends = (inner_piece.get("color") == piece.get("color") and 
                                            move.get("friendlyPieces") and 
                                            not move.get("impotent"))
                        if (check_for_friends or check_for_enemies) and not move.get("impotent"):
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
            for square in state["board"]:
                inner_piece = pieceFromSquare(square, state["pieces"])
                if inner_piece:
                    if (inner_piece.get("color") == piece.get("color") and 
                        inner_piece.get("icon") != piece.get("icon")):
                        square[flag] = True
        
        elif move_type == "takeMove":
            # Can only move to this square if capturing
            target_x = piece.get("x") + move.get("x", 0)
            target_y = piece.get("y") + move.get("y", 0)
            square = findSquareByXY(state["board"], target_x, target_y)
            
            if square:
                inner_piece = pieceFromSquare(square, state["pieces"])
                if inner_piece:
                    check_for_enemies = (inner_piece.get("color") != piece.get("color") and 
                                       not move.get("friendlyPieces"))
                    check_for_friends = (inner_piece.get("color") == piece.get("color") and 
                                       move.get("friendlyPieces"))
                    if (check_for_friends or check_for_enemies) and not move.get("impotent"):
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
                    x=piece.get("x") + offsetX,
                    y=piece.get("y") + offsetY,
                    move=move,
                    limit=limit,
                    flag=flag,
                    secondFlag=blockedFlag,
                    missedSquareX=move.get("missedSquareX", 0),
                    missedSquareY=move.get("missedSquareY", 0),
                    minimal=minimal,
                    operatedPiece=piece,
                    offsetX=offsetX,
                    offsetY=offsetY
                )


# ============================================================================
# Piece Selection
# ============================================================================

def selectPiece(playerMove: Dict, state: Dict) -> None:
    """
    Select a piece at the given coordinates.
    Sets state.pieceSelected and lights valid moves.
    """
    x = playerMove.get("x")
    y = playerMove.get("y")
    
    # Find piece at this position
    piece = None
    for p in state["pieces"]:
        if p.get("x") == x and p.get("y") == y:
            piece = p
            break
    
    if not piece:
        closeLights(state["board"])
        state["pieceSelected"] = None
        return
    
    # Check if it's the correct player's turn
    if piece.get("color") != state.get("turn"):
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
    """
    light = specialFlag or "light"
    x = playerMove.get("x")
    y = playerMove.get("y")
    
    operatedPiece = selectedForced if selectedForced else state.get("pieceSelected")
    if not operatedPiece:
        return False
    
    # Find destination square
    square = findSquareByXY(state["board"], x, y)
    if not square:
        return False
    
    # Check if square is lighted (valid move)
    if not square.get(light) and not alwaysLight:
        return False
    
    # Find pieces at destination
    enemyPiece = None
    friendlyPiece = None
    
    for p in state["pieces"]:
        if p.get("x") == x and p.get("y") == y:
            if p.get("color") != operatedPiece.get("color"):
                enemyPiece = p
            else:
                friendlyPiece = p
            break
    
    friendlyPieceOldX = friendlyPiece.get("x") if friendlyPiece else None
    friendlyPieceOldY = friendlyPiece.get("y") if friendlyPiece else None
    
    oldX = operatedPiece.get("x")
    oldY = operatedPiece.get("y")
    
    # Save old state for rollback
    import copy
    oldState = copy.deepcopy(state)
    
    # Move piece
    operatedPiece["x"] = x
    operatedPiece["y"] = y
    
    continueTurn = True
    
    # Check afterPlayerMove callbacks
    for i in range(len(state["pieces"]) - 1, -1, -1):
        piece = state["pieces"][i]
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
        # Rollback state
        state.clear()
        state.update(oldState)
        if friendlyPiece:
            friendlyPiece["x"] = friendlyPieceOldX
            friendlyPiece["y"] = friendlyPieceOldY
        operatedPiece["x"] = oldX
        operatedPiece["y"] = oldY
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
    """Change the turn to the opposite player."""
    if state.get("turn") == "white":
        state["turn"] = "black"
    else:
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

