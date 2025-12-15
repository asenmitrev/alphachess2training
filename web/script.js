// Main thread script for King Capture game

let worker = null;
let ortSession = null;
let boardLocked = true;
let selectedPiece = null; // { pieceIdx, row, col }
let validMoves = []; // List of valid moves for selected piece
let playerSide = 'white'; // 'white' or 'black'

const BOARD_SIZE = 5;
const ACTION_SIZE = 2 * BOARD_SIZE * BOARD_SIZE; // 50 actions

// Piece types (must match game.py)
const Piece = {
    EMPTY: 0,
    WHITE_KINGLIKE: 1,
    WHITE_KING: 2,
    BLACK_KINGLIKE: 3,
    BLACK_KING: 4
};

const loadingEl = document.getElementById('loading');
const loadingText = document.getElementById('loading-text');
const sideSelectEl = document.getElementById('side-select');
const statusEl = document.getElementById('status');
const boardEl = document.getElementById('board');
const whiteIndicator = document.getElementById('white-indicator');
const blackIndicator = document.getElementById('black-indicator');

// Initialize the board grid
function initBoard() {
    boardEl.innerHTML = '';
    for (let row = 0; row < BOARD_SIZE; row++) {
        for (let col = 0; col < BOARD_SIZE; col++) {
            const cell = document.createElement('div');
            cell.className = 'cell ' + ((row + col) % 2 === 0 ? 'light' : 'dark');
            cell.dataset.row = row;
            cell.dataset.col = col;
            cell.addEventListener('click', () => handleCellClick(row, col));
            boardEl.appendChild(cell);
        }
    }
}

// Initialize
async function init() {
    initBoard();
    
    try {
        // Load ONNX Model
        loadingText.innerText = "Loading AI Model...";
        ortSession = await ort.InferenceSession.create('./model_embedded.onnx');
        console.log("ONNX Session created");

        // Initialize Web Worker
        loadingText.innerText = "Loading Python Engine...";
        worker = new Worker('worker.js');
        
        worker.onmessage = handleWorkerMessage;
        worker.postMessage({ type: 'INIT' });
        
    } catch (e) {
        console.error(e);
        loadingText.innerText = "Error initializing: " + e.message;
    }
}

// Handle messages from Worker
async function handleWorkerMessage(e) {
    const msg = e.data;
    const type = msg.type;
    console.log("Main thread received:", type, msg);
    
    switch (type) {
        case 'READY':
            console.log("Hiding loading screen, showing side select...");
            loadingEl.style.display = 'none';
            sideSelectEl.style.display = 'flex';
            break;
            
        case 'RUN_INFERENCE':
            console.log("Main: received RUN_INFERENCE with id", msg.id);
            await runInference(msg.id, msg.states);
            console.log("Main: completed RUN_INFERENCE for id", msg.id);
            break;
            
        case 'STATE_UPDATE':
            updateBoard(msg.board, msg.lastMove);
            updatePlayerIndicator(msg.currentPlayer);
            if (msg.gameOver) {
                handleGameOver(msg.winner);
            } else {
                const playerTurnValue = playerSide === 'white' ? 1 : 2;
                if (msg.currentPlayer === playerTurnValue) {
                    // Player's turn
                    boardLocked = false;
                    statusEl.innerText = "Your Turn";
                    statusEl.classList.remove('thinking');
                }
            }
            break;
            
        case 'VALID_MOVES':
            validMoves = msg.moves;
            highlightValidMoves();
            break;
            
        case 'AI_THINKING':
            statusEl.innerText = "AI is thinking...";
            statusEl.classList.add('thinking');
            boardLocked = true;
            break;
    }
}

// Run ONNX Inference
async function runInference(id, flattenedStates) {
    if (!ortSession) return;

    try {
        const batchSize = flattenedStates.length;
        const inputData = new Float32Array(batchSize * 1 * BOARD_SIZE * BOARD_SIZE);
        
        // Fill input tensor
        for (let i = 0; i < batchSize; i++) {
            const state = flattenedStates[i];
            for (let j = 0; j < BOARD_SIZE * BOARD_SIZE; j++) {
                inputData[i * BOARD_SIZE * BOARD_SIZE + j] = state[j];
            }
        }
        
        const tensor = new ort.Tensor('float32', inputData, [batchSize, 1, BOARD_SIZE, BOARD_SIZE]);
        const feeds = { input: tensor };
        
        const results = await ortSession.run(feeds);
        
        // Extract results
        const policyData = results.policy.data;
        const valueData = results.value.data;
        
        // Format for Python
        const policies = [];
        const values = [];
        
        for (let i = 0; i < batchSize; i++) {
            const p = [];
            for (let j = 0; j < ACTION_SIZE; j++) {
                p.push(policyData[i * ACTION_SIZE + j]);
            }
            policies.push(p);
            values.push(valueData[i]);
        }
        
        // Send back to worker
        worker.postMessage({
            type: 'INFERENCE_RESULT',
            id: id,
            result: { policies, values }
        });
        
    } catch (e) {
        console.error("Inference failed", e);
    }
}

// UI Helpers
function updateBoard(board, lastMove) {
    const cells = boardEl.querySelectorAll('.cell');
    
    cells.forEach((cell) => {
        const row = parseInt(cell.dataset.row);
        const col = parseInt(cell.dataset.col);
        const idx = row * BOARD_SIZE + col;
        const val = board[idx];
        
        // Reset classes
        cell.className = 'cell ' + ((row + col) % 2 === 0 ? 'light' : 'dark');
        cell.innerHTML = '';
        
        // Add piece
        if (val !== Piece.EMPTY) {
            const piece = document.createElement('span');
            piece.className = 'piece';
            
            switch (val) {
                case Piece.WHITE_KING:
                    piece.classList.add('white-king');
                    break;
                case Piece.WHITE_KINGLIKE:
                    piece.classList.add('white-kinglike');
                    break;
                case Piece.BLACK_KING:
                    piece.classList.add('black-king');
                    break;
                case Piece.BLACK_KINGLIKE:
                    piece.classList.add('black-kinglike');
                    break;
            }
            
            cell.appendChild(piece);
        }
        
        // Highlight last move
        if (lastMove && lastMove.row === row && lastMove.col === col) {
            cell.classList.add('last-move');
        }
    });
    
    // Clear selection
    selectedPiece = null;
    validMoves = [];
}

function updatePlayerIndicator(currentPlayer) {
    if (currentPlayer === 1) {
        whiteIndicator.classList.add('active');
        blackIndicator.classList.remove('active');
    } else {
        whiteIndicator.classList.remove('active');
        blackIndicator.classList.add('active');
    }
}

function highlightValidMoves() {
    const cells = boardEl.querySelectorAll('.cell');
    
    cells.forEach((cell) => {
        cell.classList.remove('valid-move', 'valid-capture', 'selected');
    });
    
    if (selectedPiece) {
        // Highlight selected piece
        const selectedCell = boardEl.querySelector(`[data-row="${selectedPiece.row}"][data-col="${selectedPiece.col}"]`);
        if (selectedCell) {
            selectedCell.classList.add('selected');
        }
        
        // Highlight valid moves
        for (const move of validMoves) {
            const cell = boardEl.querySelector(`[data-row="${move.row}"][data-col="${move.col}"]`);
            if (cell) {
                // Check if it's a capture
                if (cell.querySelector('.piece')) {
                    cell.classList.add('valid-capture');
                } else {
                    cell.classList.add('valid-move');
                }
            }
        }
    }
}

function handleCellClick(row, col) {
    if (boardLocked) return;
    
    const cell = boardEl.querySelector(`[data-row="${row}"][data-col="${col}"]`);
    const piece = cell.querySelector('.piece');
    
    // Check if clicking on own piece based on player's side
    const isOwnPiece = playerSide === 'white'
        ? piece && (piece.classList.contains('white-king') || piece.classList.contains('white-kinglike'))
        : piece && (piece.classList.contains('black-king') || piece.classList.contains('black-kinglike'));
    
    if (isOwnPiece) {
        // Select this piece - determine if it's king (idx 0) or kinglike (idx 1)
        const isKing = playerSide === 'white' 
            ? piece.classList.contains('white-king')
            : piece.classList.contains('black-king');
        const pieceIdx = isKing ? 0 : 1;
        selectedPiece = { pieceIdx, row, col };
        
        // Request valid moves from worker
        worker.postMessage({ type: 'GET_VALID_MOVES', data: { pieceIdx } });
        return;
    }
    
    // Check if clicking on a valid move destination
    if (selectedPiece) {
        const isValidMove = validMoves.some(m => m.row === row && m.col === col);
        
        if (isValidMove) {
            // Make the move
            boardLocked = true;
            worker.postMessage({
                type: 'MAKE_MOVE',
                data: { pieceIdx: selectedPiece.pieceIdx, row, col }
            });
            selectedPiece = null;
            validMoves = [];
            highlightValidMoves();
        } else {
            // Deselect
            selectedPiece = null;
            validMoves = [];
            highlightValidMoves();
        }
    }
}

function handleGameOver(winner) {
    boardLocked = true;
    statusEl.classList.remove('thinking');
    
    const playerWinValue = playerSide === 'white' ? 1 : 2;
    if (winner === playerWinValue) {
        statusEl.innerText = "🎉 You Won!";
    } else if (winner !== 0) {
        statusEl.innerText = "🤖 AI Won!";
    } else {
        statusEl.innerText = "🤝 Draw!";
    }
}

function restartGame() {
    if (worker) {
        boardLocked = true;
        selectedPiece = null;
        validMoves = [];
        statusEl.classList.remove('thinking');
        boardEl.classList.remove('flipped');
        sideSelectEl.style.display = 'flex';
    }
}

function startGame(side) {
    console.log("startGame called with side:", side);
    playerSide = side;
    sideSelectEl.style.display = 'none';
    
    // Update player indicator labels
    if (side === 'white') {
        whiteIndicator.querySelector('.player-name').innerText = 'You (White)';
        blackIndicator.querySelector('.player-name').innerText = 'AI (Black)';
        boardEl.classList.remove('flipped');
    } else {
        whiteIndicator.querySelector('.player-name').innerText = 'AI (White)';
        blackIndicator.querySelector('.player-name').innerText = 'You (Black)';
        boardEl.classList.add('flipped');
    }
    
    // Reset the game with the chosen side
    console.log("Sending RESET with playerSide:", side);
    worker.postMessage({ type: 'RESET', data: { playerSide: side } });
    statusEl.innerText = side === 'white' ? "Your Turn" : "AI is thinking...";
    if (side === 'black') {
        statusEl.classList.add('thinking');
    }
    updatePlayerIndicator(1); // White always starts
}

// Start
init();
