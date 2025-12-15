package main

/*
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
*/
import "C"
import (
	"math"
	"sync"
	"unsafe"
)

// MCTSNode represents a node in the MCTS tree
type MCTSNode struct {
	state         [25]int8
	currentPlayer int8
	gameOver      bool
	winner        int8

	parent   *MCTSNode
	action   int8 // Action that led to this node
	children map[int8]*MCTSNode

	visitCount int
	valueSum   float64
	prior      float64
}

// MCTSContext holds the MCTS search state for iterative search
type MCTSContext struct {
	root           *MCTSNode
	numSimulations int
	cPuct          float64
	simulation     int
	pendingNode    *MCTSNode // Node waiting for evaluation
}

// Global context map for storing MCTS contexts
var (
	contextMap   = make(map[uintptr]*MCTSContext)
	contextMapMu sync.Mutex
	contextID    uintptr = 1
)

// checkWinConditionsFromState checks if game is over from state only
func checkWinConditionsFromState(state *[25]int8, currentPlayer int8) (bool, int8) {
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

	// Check if true king reached end row
	var kingPiece int8
	if currentPlayer == WHITE {
		kingPiece = WHITE_KING
	} else {
		kingPiece = BLACK_KING
	}

	for i := int8(0); i < BOARD_SIZE; i++ {
		for j := int8(0); j < BOARD_SIZE; j++ {
			idx := i*BOARD_SIZE + j
			if state[idx] == kingPiece {
				if currentPlayer == WHITE && i == 0 {
					// White king reached top row
					return true, WHITE_WIN
				} else if currentPlayer == BLACK && i == BOARD_SIZE-1 {
					// Black king reached bottom row
					return true, BLACK_WIN
				}
			}
		}
	}

	return false, NO_WINNER
}

// NewMCTSNode creates a new MCTS node
func NewMCTSNode(state [25]int8, currentPlayer int8, parent *MCTSNode, action int8) *MCTSNode {
	// Check if game is over
	gameOver, winner := checkWinConditionsFromState(&state, currentPlayer)

	return &MCTSNode{
		state:         state,
		currentPlayer: currentPlayer,
		gameOver:      gameOver,
		winner:        winner,
		parent:        parent,
		action:        action,
		children:      make(map[int8]*MCTSNode),
		visitCount:    0,
		valueSum:      0.0,
		prior:          0.0,
	}
}

// IsExpanded checks if node has been expanded
func (n *MCTSNode) IsExpanded() bool {
	return len(n.children) > 0
}

// GetValue returns the average value of this node
func (n *MCTSNode) GetValue() float64 {
	if n.visitCount == 0 {
		return 0.0
	}
	return n.valueSum / float64(n.visitCount)
}

// SelectChild selects a child using PUCT formula
func (n *MCTSNode) SelectChild(cPuct float64) *MCTSNode {
	bestScore := math.Inf(-1)
	var bestChild *MCTSNode

	for _, child := range n.children {
		// PUCT formula: Q + U
		// Q(s,a) = -V(child) from parent's perspective (zero-sum)
		q := -child.GetValue()

		// U = c_puct * P * sqrt(N_parent) / (1 + N_child)
		u := cPuct * child.prior * math.Sqrt(float64(n.visitCount)) / (1.0 + float64(child.visitCount))

		score := q + u

		if score > bestScore {
			bestScore = score
			bestChild = child
		}
	}

	return bestChild
}

// Expand expands the node by creating children for all valid actions
func (n *MCTSNode) Expand(policy []float64) {
	// Get valid moves
	var mask [50]int8
	getValidMovesGo(&n.state, n.currentPlayer, &mask)

	// Create children for all valid moves
	for action := int8(0); action < ACTION_SIZE; action++ {
		if mask[action] == 0 {
			continue
		}

		// Convert action to move
		pieceIdx := action / (BOARD_SIZE * BOARD_SIZE)
		remaining := action % (BOARD_SIZE * BOARD_SIZE)
		row := remaining / BOARD_SIZE
		col := remaining % BOARD_SIZE

		// Make move
		var newState [25]int8
		var gameOver int8
		var winner int8
		success := makeMoveGo(&n.state, n.currentPlayer, pieceIdx, row, col, &newState, &gameOver, &winner)

		if success == 0 {
			continue
		}

		// Determine next player
		nextPlayer := int8(1) // WHITE
		if n.currentPlayer == 1 {
			nextPlayer = 2 // BLACK
		}

		// Create child node
		child := NewMCTSNode(newState, nextPlayer, n, action)
		child.prior = policy[action]
		n.children[action] = child
	}
}

// Backpropagate propagates value up the tree
func (n *MCTSNode) Backpropagate(value float64) {
	n.valueSum += value
	n.visitCount++

	if n.parent != nil {
		// Value is from current player's perspective
		// For parent (opponent), negate the value (zero-sum game)
		n.parent.Backpropagate(-value)
	}
}

// GetCanonicalStateForEval gets canonical state for neural network evaluation
func (n *MCTSNode) GetCanonicalStateForEval() [25]float32 {
	var canonical [25]float32
	getCanonicalStateGo(&n.state, n.currentPlayer, &canonical)
	return canonical
}

// GetResult returns game result from current player's perspective
func (n *MCTSNode) GetResult() float64 {
	if !n.gameOver {
		return 0.0
	}

	if n.winner == 0 {
		return 0.0 // Draw
	}

	// Check if current player won
	if n.currentPlayer == n.winner {
		return 1.0
	}

	return -1.0
}

// FlipPolicy flips policy spatially (for Black player)
func FlipPolicy(policy []float64) []float64 {
	flipped := make([]float64, ACTION_SIZE)

	for pieceIdx := 0; pieceIdx < 2; pieceIdx++ {
		for row := 0; row < BOARD_SIZE; row++ {
			for col := 0; col < BOARD_SIZE; col++ {
				action := pieceIdx*BOARD_SIZE*BOARD_SIZE + row*BOARD_SIZE + col
				flippedRow := BOARD_SIZE - 1 - row
				flippedAction := pieceIdx*BOARD_SIZE*BOARD_SIZE + flippedRow*BOARD_SIZE + col
				flipped[flippedAction] = policy[action]
			}
		}
	}

	return flipped
}

//export MCTSCreate
func MCTSCreate(state *C.int8_t, currentPlayer C.int8_t, numSimulations C.int, cPuct C.float) C.uintptr_t {
	stateGo := (*[25]int8)(unsafe.Pointer(state))
	player := int8(currentPlayer)

	// Create root node
	root := NewMCTSNode(*stateGo, player, nil, -1)

	// Create context
	ctx := &MCTSContext{
		root:           root,
		numSimulations: int(numSimulations),
		cPuct:          float64(cPuct),
		simulation:     0,
		pendingNode:    nil,
	}

	// Store context in map and return ID
	contextMapMu.Lock()
	id := contextID
	contextID++
	contextMap[id] = ctx
	contextMapMu.Unlock()

	return C.uintptr_t(id)
}

//export MCTSDestroy
func MCTSDestroy(ctxID C.uintptr_t) {
	// Remove context from map
	contextMapMu.Lock()
	delete(contextMap, uintptr(ctxID))
	contextMapMu.Unlock()
}

// getContext safely retrieves a context by ID
func getContext(ctxID C.uintptr_t) *MCTSContext {
	contextMapMu.Lock()
	defer contextMapMu.Unlock()
	return contextMap[uintptr(ctxID)]
}

//export MCTSSearchStep
// Returns: 0 = need root evaluation, 1 = need leaf evaluation, 2 = simulation done (terminal), 3 = all simulations complete
func MCTSSearchStep(ctxID C.uintptr_t, canonicalState *C.float) C.int {
	ctx := getContext(ctxID)
	if ctx == nil {
		return -1 // Error: invalid context ID
	}

	// If root not expanded, need root evaluation
	if !ctx.root.IsExpanded() && !ctx.root.gameOver {
		canonical := ctx.root.GetCanonicalStateForEval()
		stateGo := (*[25]float32)(unsafe.Pointer(canonicalState))
		for i := 0; i < 25; i++ {
			stateGo[i] = canonical[i]
		}
		ctx.pendingNode = ctx.root
		return 0 // Need root evaluation
	}

	// Check if all simulations done
	if ctx.simulation >= ctx.numSimulations {
		return 3 // All simulations complete
	}

	// Do one simulation step
	node := ctx.root

	// Selection: traverse to leaf
	for node.IsExpanded() && !node.gameOver {
		node = node.SelectChild(ctx.cPuct)
	}

	// Check if terminal
	if node.gameOver {
		result := node.GetResult()
		node.Backpropagate(result)
		ctx.simulation++
		return 2 // Simulation done (terminal), continue
	}

	// Leaf needs evaluation
	canonical := node.GetCanonicalStateForEval()
	stateGo := (*[25]float32)(unsafe.Pointer(canonicalState))
	for i := 0; i < 25; i++ {
		stateGo[i] = canonical[i]
	}
	ctx.pendingNode = node
	return 1 // Need leaf evaluation
}

//export MCTSProvideEvaluation
func MCTSProvideEvaluation(
	ctxID C.uintptr_t,
	policy *C.float,
	value C.float,
) {
	ctx := getContext(ctxID)
	if ctx == nil {
		return // Error: invalid context ID
	}
	policyGo := (*[50]float64)(unsafe.Pointer(policy))

	if ctx.pendingNode == nil {
		return
	}

	node := ctx.pendingNode
	val := float64(value)

	// Extract policy
	policySlice := make([]float64, ACTION_SIZE)
	for i := 0; i < ACTION_SIZE; i++ {
		policySlice[i] = policyGo[i]
	}

	// Flip policy if Black
	if node.currentPlayer == 2 { // BLACK
		policySlice = FlipPolicy(policySlice)
	}

	// Mask invalid actions
	var mask [50]int8
	getValidMovesGo(&node.state, node.currentPlayer, &mask)

	// Normalize policy
	sum := 0.0
	for i := 0; i < ACTION_SIZE; i++ {
		if mask[i] == 0 {
			policySlice[i] = 0.0
		}
		sum += policySlice[i]
	}

	if sum > 0 {
		for i := 0; i < ACTION_SIZE; i++ {
			policySlice[i] /= sum
		}
	} else {
		// Fallback to uniform
		validCount := 0
		for i := 0; i < ACTION_SIZE; i++ {
			if mask[i] != 0 {
				validCount++
			}
		}
		if validCount > 0 {
			uniform := 1.0 / float64(validCount)
			for i := 0; i < ACTION_SIZE; i++ {
				if mask[i] != 0 {
					policySlice[i] = uniform
				}
			}
		}
	}

	// Expand node
	node.Expand(policySlice)
	node.Backpropagate(val)

	// Clear pending node
	ctx.pendingNode = nil
	ctx.simulation++
}

//export MCTSGetPolicy
func MCTSGetPolicy(ctxID C.uintptr_t, policy *C.float) {
	ctx := getContext(ctxID)
	if ctx == nil {
		return // Error: invalid context ID
	}
	policyGo := (*[50]float64)(unsafe.Pointer(policy))

	root := ctx.root

	// Extract visit counts as policy
	visitCounts := make([]float64, ACTION_SIZE)
	for action, child := range root.children {
		visitCounts[action] = float64(child.visitCount)
	}

	// Normalize to get policy distribution
	sum := 0.0
	for i := 0; i < ACTION_SIZE; i++ {
		sum += visitCounts[i]
	}

	if sum > 0 {
		for i := 0; i < ACTION_SIZE; i++ {
			policyGo[i] = float64(visitCounts[i] / sum)
		}
	} else {
		// Fallback to uniform over valid moves
		var mask [50]int8
		getValidMovesGo(&root.state, root.currentPlayer, &mask)
		validCount := 0
		for i := 0; i < ACTION_SIZE; i++ {
			if mask[i] != 0 {
				validCount++
			}
		}
		if validCount > 0 {
			uniform := 1.0 / float64(validCount)
			for i := 0; i < ACTION_SIZE; i++ {
				if mask[i] != 0 {
					policyGo[i] = uniform
				}
			}
		}
	}
}

// Internal Go wrappers for game engine functions
func makeMoveGo(state *[25]int8, currentPlayer int8, pieceIdx int8, row int8, col int8,
	newState *[25]int8, gameOver *int8, winner *int8) int8 {
	result := MakeMove(
		(*C.int8_t)(unsafe.Pointer(state)),
		C.int8_t(currentPlayer),
		C.int8_t(pieceIdx),
		C.int8_t(row),
		C.int8_t(col),
		(*C.int8_t)(unsafe.Pointer(newState)),
		(*C.int8_t)(unsafe.Pointer(gameOver)),
		(*C.int8_t)(unsafe.Pointer(winner)),
	)
	return int8(result)
}

func getValidMovesGo(state *[25]int8, currentPlayer int8, mask *[50]int8) {
	GetValidMoves(
		(*C.int8_t)(unsafe.Pointer(state)),
		C.int8_t(currentPlayer),
		(*C.int8_t)(unsafe.Pointer(mask)),
	)
}

func getCanonicalStateGo(state *[25]int8, currentPlayer int8, canonical *[25]float32) {
	GetCanonicalState(
		(*C.int8_t)(unsafe.Pointer(state)),
		C.int8_t(currentPlayer),
		(*C.float)(unsafe.Pointer(canonical)),
	)
}
