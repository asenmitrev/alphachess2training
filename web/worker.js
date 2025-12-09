// Web Worker for handling Python/Pyodide and game logic

importScripts("https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js");

let pyodide = null;
let pythonMCTS = null;
let pythonGame = null;

// Initialize Pyodide
async function initPyodide() {
    pyodide = await loadPyodide();
    await pyodide.loadPackage("numpy");
    
    // Load local python files
    // In a real deployment, you'd fetch these. For local dev, we might need a way to serve them or 
    // inject them. We'll assume they are served at the root of web/.
    const responseGame = await fetch("game.py");
    const gamePyContent = await responseGame.text();
    pyodide.FS.writeFile("game.py", gamePyContent);

    const responseMCTS = await fetch("mcts_wasm.py");
    const mctsPyContent = await responseMCTS.text();
    pyodide.FS.writeFile("mcts_wasm.py", mctsPyContent);
    
    // Setup Python environment
    await pyodide.runPythonAsync(`
        import sys
        sys.path.append('.')
        from game import TicTacToe, Player
        from mcts_wasm import MCTS
        import numpy as np
        
        game = TicTacToe()
        mcts = None
        
        # We'll define a python wrapper for the inference callback
        async def inference_callback(states):
            # Convert list of numpy arrays to JS list of flattened Float32Arrays
            # JS side will handle batching if needed, but here we just pass one batch
            import js
            
            # states is a list of (3, 3) arrays
            # We flatten them for easier passing
            flattened_states = [s.flatten().tolist() for s in states]
            
            # Call JS function (we need to register this on 'js' module or self)
            # When calling from Python to JS, we need to be careful with types.
            # Pyodide usually handles basic types well.
            result_proxy = await js.runInference(flattened_states)
            # result_proxy is a JsProxy. We can use .to_py() to convert to Python dict/list
            result = result_proxy.to_py()
            
            # Result should be { 'policies': [[...], ...], 'values': [...] }
            policies = [np.array(p) for p in result['policies']]
            values = result['values']
            
            return policies, values

        def init_mcts():
            global mcts
            # Create MCTS instance
            mcts = MCTS(inference_callback, num_simulations=200) # Adjust simulations as needed
            print("MCTS Initialized")

        def make_move(row, col):
            return game.make_move(row, col)
            
        def get_board_state():
            return game.board.flatten().tolist()
            
        def get_valid_moves():
            return game.get_valid_moves()
            
        def check_game_over():
            return game.game_over, game.winner.value if game.winner else 0
        
        def reset_game():
            global game
            game = TicTacToe()
            return True
            
        async def get_ai_move():
            policy = await mcts.search(game)
            # Choose best move (argmax for play)
            action = np.argmax(policy)
            row, col = game.action_to_move(int(action))
            return int(row), int(col)
            
        init_mcts()
    `);
    
    postMessage({ type: "READY" });
}

// Handle messages from main thread
self.onmessage = async (event) => {
    const { type, data } = event.data;
    
    if (type === "INIT") {
        await initPyodide();
    } else if (type === "MAKE_MOVE") {
        const { row, col } = data;
        const valid = pyodide.runPython(`make_move(${row}, ${col})`);
        // Convert PyProxy to JS array
        const board = pyodide.runPython(`get_board_state()`).toJs();
        // Convert Tuple/List proxy to JS array
        const gameOverResult = pyodide.runPython(`check_game_over()`).toJs();
        const gameOver = gameOverResult[0];
        const winner = gameOverResult[1];
        
        postMessage({ type: "STATE_UPDATE", board, gameOver, winner });
        
        if (!gameOver && valid) {
            // Trigger AI move if it was player's turn and game not over
            // Or we can let UI decide when to trigger AI
            postMessage({ type: "AI_THINKING" });
            const aiMove = await pyodide.runPythonAsync(`get_ai_move()`);
            const [aiRow, aiCol] = aiMove.toJs();
            
            // Execute AI move
            pyodide.runPython(`make_move(${aiRow}, ${aiCol})`);
            const boardAfterAI = pyodide.runPython(`get_board_state()`).toJs();
            const gameOverResultAI = pyodide.runPython(`check_game_over()`).toJs();
            const gameOverAI = gameOverResultAI[0];
            const winnerAI = gameOverResultAI[1];
            
            postMessage({ type: "STATE_UPDATE", board: boardAfterAI, gameOver: gameOverAI, winner: winnerAI });
        }
    } else if (type === "RESET") {
        pyodide.runPython(`reset_game()`);
        const board = pyodide.runPython(`get_board_state()`).toJs();
        postMessage({ type: "STATE_UPDATE", board, gameOver: false, winner: 0 });
    }
};

// Expose inference function to Python
self.runInference = async (flattenedStates) => {
    // flattenedStates comes from Python, so it might be a PyProxy.
    // We must convert it to a pure JS object before sending via postMessage.
    let statesJs = flattenedStates;
    if (flattenedStates && typeof flattenedStates.toJs === 'function') {
        // recursive: true is default for toJs in recent Pyodide versions, but let's be safe
        // Actually toJs() converts recursively by default.
        statesJs = flattenedStates.toJs();
    }

    // Send to main thread for ONNX execution
    return new Promise((resolve) => {
        const id = Math.random().toString(36).substring(7);
        
        const handler = (event) => {
            if (event.data.type === "INFERENCE_RESULT" && event.data.id === id) {
                self.removeEventListener("message", handler);
                resolve(event.data.result);
            }
        };
        
        self.addEventListener("message", handler);
        postMessage({ type: "RUN_INFERENCE", id, states: statesJs });
    });
};

