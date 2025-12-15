package main

/*
#include <stdint.h>
*/
import "C"
import (
	"unsafe"
)

// Piece constants
const (
	EMPTY          = int8(0)
	WHITE_KINGLIKE = int8(1)
	WHITE_KING     = int8(2)
	BLACK_KINGLIKE = int8(3)
	BLACK_KING     = int8(4)
)

// Player constants
const (
	WHITE = int8(1)
	BLACK = int8(2)
)

// Winner constants
const (
	NO_WINNER = int8(0)
	WHITE_WIN = int8(1)
	BLACK_WIN = int8(2)
)

const BOARD_SIZE = 5
const ACTION_SIZE = 50 // 2 pieces * 25 positions

// King move offsets (8 directions)
var kingMoves = [8][2]int8{
	{-1, -1}, {-1, 0}, {-1, 1},
	{0, -1}, {0, 1},
	{1, -1}, {1, 0}, {1, 1},
}

// getPiecePositions returns positions of pieces for current player
// Returns: (king_row, king_col, kinglike_row, kinglike_col, king_exists, kinglike_exists)
func getPiecePositions(state *[25]int8, currentPlayer int8) (int8, int8, int8, int8, bool, bool) {
	var kingRow, kingCol, kinglikeRow, kinglikeCol int8
	var kingExists, kinglikeExists bool

	kingPiece := WHITE_KING
	kinglikePiece := WHITE_KINGLIKE
	if currentPlayer == BLACK {
		kingPiece = BLACK_KING
		kinglikePiece = BLACK_KINGLIKE
	}

	for i := int8(0); i < BOARD_SIZE; i++ {
		for j := int8(0); j < BOARD_SIZE; j++ {
			idx := i*BOARD_SIZE + j
			piece := state[idx]
			if piece == kingPiece {
				kingRow = i
				kingCol = j
				kingExists = true
			} else if piece == kinglikePiece {
				kinglikeRow = i
				kinglikeCol = j
				kinglikeExists = true
			}
		}
	}

	return kingRow, kingCol, kinglikeRow, kinglikeCol, kingExists, kinglikeExists
}

// isValidPosition checks if coordinates are within board bounds
func isValidPosition(row, col int8) bool {
	return row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE
}

// isEnemyPiece checks if a piece belongs to the opponent
func isEnemyPiece(piece int8, currentPlayer int8) bool {
	if currentPlayer == WHITE {
		return piece == BLACK_KING || piece == BLACK_KINGLIKE
	}
	return piece == WHITE_KING || piece == WHITE_KINGLIKE
}

// isFriendlyPiece checks if a piece belongs to current player
func isFriendlyPiece(piece int8, currentPlayer int8) bool {
	if currentPlayer == WHITE {
		return piece == WHITE_KING || piece == WHITE_KINGLIKE
	}
	return piece == BLACK_KING || piece == BLACK_KINGLIKE
}

// checkWinConditions checks if game is over and who won
// Returns: (game_over, winner)
func checkWinConditions(state *[25]int8, currentPlayer int8, pieceIdx int8, newRow int8) (bool, int8) {
	// Check if opponent's king was captured
	var opponentKing int8
	if currentPlayer == WHITE {
		opponentKing = BLACK_KING
	} else {
		opponentKing = WHITE_KING
	}

	opponentKingExists := false
	for i := int8(0); i < BOARD_SIZE*BOARD_SIZE; i++ {
		if state[i] == opponentKing {
			opponentKingExists = true
			break
		}
	}

	if !opponentKingExists {
		// Current player captured opponent's king
		if currentPlayer == WHITE {
			return true, WHITE_WIN
		}
		return true, BLACK_WIN
	}

	// Check if true king reached end row (only if true king was moved)
	if pieceIdx == 0 {
		if currentPlayer == WHITE && newRow == 0 {
			// White king reached top row
			return true, WHITE_WIN
		} else if currentPlayer == BLACK && newRow == BOARD_SIZE-1 {
			// Black king reached bottom row
			return true, BLACK_WIN
		}
	}

	return false, NO_WINNER
}

//export MakeMove
func MakeMove(
	state *C.int8_t,
	currentPlayer C.int8_t,
	pieceIdx C.int8_t,
	row C.int8_t,
	col C.int8_t,
	newState *C.int8_t,
	gameOver *C.int8_t,
	winner *C.int8_t,
) C.int8_t {
	// Convert C pointers to Go arrays
	stateGo := (*[25]int8)(unsafe.Pointer(state))
	newStateGo := (*[25]int8)(unsafe.Pointer(newState))

	// Copy state to newState
	for i := 0; i < 25; i++ {
		newStateGo[i] = stateGo[i]
	}

	player := int8(currentPlayer)
	pidx := int8(pieceIdx)
	targetRow := int8(row)
	targetCol := int8(col)

	// Validate inputs
	if player != WHITE && player != BLACK {
		return 0
	}
	if pidx != 0 && pidx != 1 {
		return 0
	}
	if !isValidPosition(targetRow, targetCol) {
		return 0
	}

	// Get piece positions
	kingRow, kingCol, kinglikeRow, kinglikeCol, kingExists, kinglikeExists := getPiecePositions(stateGo, player)

	// Determine which piece to move
	var pieceRow, pieceCol int8
	var pieceType int8
	if pidx == 0 {
		if !kingExists {
			return 0
		}
		pieceRow = kingRow
		pieceCol = kingCol
		if player == WHITE {
			pieceType = WHITE_KING
		} else {
			pieceType = BLACK_KING
		}
	} else {
		if !kinglikeExists {
			return 0
		}
		pieceRow = kinglikeRow
		pieceCol = kinglikeCol
		if player == WHITE {
			pieceType = WHITE_KINGLIKE
		} else {
			pieceType = BLACK_KINGLIKE
		}
	}

	// Check if move is valid (king can move 1 square in any direction)
	validMove := false
	for _, move := range kingMoves {
		if pieceRow+move[0] == targetRow && pieceCol+move[1] == targetCol {
			validMove = true
			break
		}
	}

	if !validMove {
		return 0
	}

	// Check destination
	destIdx := targetRow*BOARD_SIZE + targetCol
	destPiece := newStateGo[destIdx]

	// Can't move to square with friendly piece
	if isFriendlyPiece(destPiece, player) {
		return 0
	}

	// Move piece
	oldIdx := pieceRow*BOARD_SIZE + pieceCol
	newStateGo[oldIdx] = EMPTY
	newStateGo[destIdx] = pieceType

	// Check win conditions
	over, win := checkWinConditions(newStateGo, player, pidx, targetRow)
	if over {
		*gameOver = C.int8_t(1)
		*winner = C.int8_t(win)
	} else {
		*gameOver = C.int8_t(0)
		*winner = C.int8_t(NO_WINNER)
	}

	return 1 // Success
}

//export GetValidMoves
func GetValidMoves(
	state *C.int8_t,
	currentPlayer C.int8_t,
	mask *C.int8_t,
) {
	stateGo := (*[25]int8)(unsafe.Pointer(state))
	maskGo := (*[50]int8)(unsafe.Pointer(mask))

	// Initialize mask to zeros
	for i := 0; i < ACTION_SIZE; i++ {
		maskGo[i] = 0
	}

	player := int8(currentPlayer)
	if player != WHITE && player != BLACK {
		return
	}

	// Get piece positions
	kingRow, kingCol, kinglikeRow, kinglikeCol, kingExists, kinglikeExists := getPiecePositions(stateGo, player)

	// Check moves for king (piece_idx = 0)
	if kingExists {
		for _, move := range kingMoves {
			newRow := kingRow + move[0]
			newCol := kingCol + move[1]

			if !isValidPosition(newRow, newCol) {
				continue
			}

			destIdx := newRow*BOARD_SIZE + newCol
			destPiece := stateGo[destIdx]

			// Can move to empty square or capture enemy piece
			if destPiece == EMPTY || isEnemyPiece(destPiece, player) {
				actionIdx := int(0*BOARD_SIZE*BOARD_SIZE + newRow*BOARD_SIZE + newCol)
				if actionIdx < ACTION_SIZE {
					maskGo[actionIdx] = 1
				}
			}
		}
	}

	// Check moves for kinglike (piece_idx = 1)
	if kinglikeExists {
		for _, move := range kingMoves {
			newRow := kinglikeRow + move[0]
			newCol := kinglikeCol + move[1]

			if !isValidPosition(newRow, newCol) {
				continue
			}

			destIdx := newRow*BOARD_SIZE + newCol
			destPiece := stateGo[destIdx]

			// Can move to empty square or capture enemy piece
			if destPiece == EMPTY || isEnemyPiece(destPiece, player) {
				actionIdx := int(1*BOARD_SIZE*BOARD_SIZE + newRow*BOARD_SIZE + newCol)
				if actionIdx < ACTION_SIZE {
					maskGo[actionIdx] = 1
				}
			}
		}
	}
}

//export GetCanonicalState
func GetCanonicalState(
	state *C.int8_t,
	currentPlayer C.int8_t,
	canonicalState *C.float,
) {
	stateGo := (*[25]int8)(unsafe.Pointer(state))
	canonicalGo := (*[25]float32)(unsafe.Pointer(canonicalState))

	player := int8(currentPlayer)

	// Convert to canonical form
	for i := int8(0); i < BOARD_SIZE; i++ {
		for j := int8(0); j < BOARD_SIZE; j++ {
			idx := i*BOARD_SIZE + j
			piece := stateGo[idx]

			var canonicalRow, canonicalCol int8
			if player == BLACK {
				// Flip spatially: row 0 becomes row 4
				canonicalRow = BOARD_SIZE - 1 - i
				canonicalCol = j
			} else {
				canonicalRow = i
				canonicalCol = j
			}
			canonicalIdx := canonicalRow*BOARD_SIZE + canonicalCol

			var value float32
			if player == WHITE {
				// White is positive, black is negative
				switch piece {
				case WHITE_KING:
					value = 1.0
				case WHITE_KINGLIKE:
					value = 0.5
				case BLACK_KING:
					value = -1.0
				case BLACK_KINGLIKE:
					value = -0.5
				default:
					value = 0.0
				}
			} else {
				// Black is positive, white is negative
				switch piece {
				case BLACK_KING:
					value = 1.0
				case BLACK_KINGLIKE:
					value = 0.5
				case WHITE_KING:
					value = -1.0
				case WHITE_KINGLIKE:
					value = -0.5
				default:
					value = 0.0
				}
			}

			canonicalGo[canonicalIdx] = value
		}
	}
}

func main() {
	// Empty main - this is a shared library
}

