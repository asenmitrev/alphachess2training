// Main thread script

let worker = null;
let ortSession = null;
let boardLocked = true;

const loadingEl = document.getElementById('loading');
const loadingText = document.getElementById('loading-text');
const statusEl = document.getElementById('status');
const cells = document.querySelectorAll('.cell');

// Initialize
async function init() {
    try {
        // Load ONNX Model
        loadingText.innerText = "Loading AI Model...";
        // Use the embedded model to avoid external data issues
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
    console.log("Main thread received:", type);
    
    switch (type) {
        case 'READY':
            console.log("Hiding loading screen...");
            loadingEl.style.display = 'none';
            boardLocked = false;
            break;
            
        case 'RUN_INFERENCE':
            await runInference(msg.id, msg.states);
            break;
            
        case 'STATE_UPDATE':
            updateBoard(msg.board);
            if (msg.gameOver) {
                handleGameOver(msg.winner);
            } else {
                boardLocked = false;
                statusEl.innerText = "Your Turn (X)";
            }
            break;
            
        case 'AI_THINKING':
            statusEl.innerText = "AI is thinking... 🧠";
            boardLocked = true;
            break;
    }
}

// Run ONNX Inference
async function runInference(id, flattenedStates) {
    if (!ortSession) return;

    try {
        const batchSize = flattenedStates.length;
        const inputData = new Float32Array(batchSize * 1 * 3 * 3);
        
        // Fill input tensor
        for (let i = 0; i < batchSize; i++) {
            const state = flattenedStates[i];
            for (let j = 0; j < 9; j++) {
                inputData[i * 9 + j] = state[j];
            }
        }
        
        const tensor = new ort.Tensor('float32', inputData, [batchSize, 1, 3, 3]);
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
            for (let j = 0; j < 9; j++) {
                p.push(policyData[i * 9 + j]);
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
function updateBoard(board) {
    cells.forEach((cell, idx) => {
        const val = board[idx];
        cell.className = 'cell'; // reset
        cell.innerText = '';
        
        if (val === 1) {
            cell.classList.add('x');
            cell.innerText = '❌';
        } else if (val === 2) {
            cell.classList.add('o');
            cell.innerText = '⭕';
        }
    });
}

function handleGameOver(winner) {
    boardLocked = true;
    if (winner === 1) statusEl.innerText = "You Won! 🎉"; // Should not happen if played perfectly :P
    else if (winner === 2) statusEl.innerText = "AI Won! 🤖";
    else statusEl.innerText = "It's a Draw! 🤝";
}

function restartGame() {
    if (worker) {
        worker.postMessage({ type: 'RESET' });
        statusEl.innerText = "Your Turn (X)";
        boardLocked = false;
    }
}

// Event Listeners
cells.forEach(cell => {
    cell.addEventListener('click', () => {
        if (boardLocked) return;
        
        const idx = parseInt(cell.dataset.index);
        const row = Math.floor(idx / 3);
        const col = idx % 3;
        
        // Optimistic update check (prevent double click)
        if (cell.innerText !== '') return;
        
        boardLocked = true;
        worker.postMessage({ type: 'MAKE_MOVE', data: { row, col } });
    });
});

// Start
init();

