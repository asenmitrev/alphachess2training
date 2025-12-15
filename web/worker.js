// Web Worker for handling Python/Pyodide and King Capture game logic

importScripts("https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js");

let pyodide = null;
let pythonMCTS = null;
let pythonGame = null;
let playerSide = "white"; // Track which side the player is on

// Initialize Pyodide
async function initPyodide() {
  pyodide = await loadPyodide();
  await pyodide.loadPackage("numpy");

  // Load local python files
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
        from game import KingCapture, Player, Piece
        from mcts_wasm import MCTS
        import numpy as np
        
        BOARD_SIZE = 5
        
        game = KingCapture()
        mcts = None
        
        # We'll define a python wrapper for the inference callback
        async def inference_callback(states):
            # Convert list of numpy arrays to JS list of flattened Float32Arrays
            import js
            
            # states is a list of (5, 5) arrays
            # We flatten them for easier passing
            flattened_states = [s.flatten().tolist() for s in states]
            
            # Call JS function
            result_proxy = await js.runInference(flattened_states)
            # result_proxy is a JsProxy. We can use .to_py() to convert to Python dict/list
            result = result_proxy.to_py()
            
            # Result should be { 'policies': [[...], ...], 'values': [...] }
            policies = [np.array(p) for p in result['policies']]
            values = result['values']
            
            return policies, values

        def init_mcts():
            global mcts
            # Create MCTS instance with fewer simulations for browser performance
            mcts = MCTS(inference_callback, num_simulations=400, c_puct=2.0)
            print("MCTS Initialized")

        def make_move(piece_idx, row, col):
            return game.make_move(piece_idx, row, col)
            
        def get_board_state():
            return game.board.flatten().tolist()
            
        def get_current_player():
            return game.current_player.value
            
        def get_valid_moves_for_piece(piece_idx):
            """Get valid moves for a specific piece (0=king, 1=kinglike)."""
            all_moves = game.get_valid_moves()
            # Filter moves for the given piece_idx
            piece_moves = [(row, col) for (p_idx, row, col) in all_moves if p_idx == piece_idx]
            return piece_moves
            
        def check_game_over():
            winner_val = 0
            if game.winner == Player.WHITE:
                winner_val = 1
            elif game.winner == Player.BLACK:
                winner_val = 2
            return game.game_over, winner_val
        
        def reset_game():
            global game
            game = KingCapture()
            return True
            
        async def get_ai_move():
            policy = await mcts.search(game)
            # Choose best move (argmax for play)
            action = int(np.argmax(policy))
            piece_idx, row, col = game.action_to_move(action)
            return int(piece_idx), int(row), int(col)
            
        init_mcts()
    `);

  postMessage({ type: "READY" });
}

// Handle messages from main thread
self.onmessage = async (event) => {
  const { type, data } = event.data;
  console.log("Worker received message:", type, data);

  if (type === "INIT") {
    await initPyodide();
  } else if (type === "MAKE_MOVE") {
    const { pieceIdx, row, col } = data;
    const valid = pyodide.runPython(`make_move(${pieceIdx}, ${row}, ${col})`);

    // Convert PyProxy to JS array
    const board = pyodide.runPython(`get_board_state()`).toJs();
    const currentPlayer = pyodide.runPython(`get_current_player()`);
    const gameOverResult = pyodide.runPython(`check_game_over()`).toJs();
    const gameOver = gameOverResult[0];
    const winner = gameOverResult[1];

    postMessage({
      type: "STATE_UPDATE",
      board,
      currentPlayer,
      gameOver,
      winner,
      lastMove: { row, col },
    });

    // Determine if it's AI's turn: AI plays opposite of playerSide
    const aiSide = playerSide === "white" ? 2 : 1; // AI is black (2) if player is white, else white (1)

    if (!gameOver && valid && currentPlayer === aiSide) {
      // Trigger AI move
      postMessage({ type: "AI_THINKING" });

      try {
        const aiMove = await pyodide.runPythonAsync(`get_ai_move()`);
        const [aiPieceIdx, aiRow, aiCol] = aiMove.toJs();

        // Execute AI move
        pyodide.runPython(`make_move(${aiPieceIdx}, ${aiRow}, ${aiCol})`);
        const boardAfterAI = pyodide.runPython(`get_board_state()`).toJs();
        const currentPlayerAfterAI = pyodide.runPython(`get_current_player()`);
        const gameOverResultAI = pyodide.runPython(`check_game_over()`).toJs();
        const gameOverAI = gameOverResultAI[0];
        const winnerAI = gameOverResultAI[1];

        postMessage({
          type: "STATE_UPDATE",
          board: boardAfterAI,
          currentPlayer: currentPlayerAfterAI,
          gameOver: gameOverAI,
          winner: winnerAI,
          lastMove: { row: aiRow, col: aiCol },
        });
      } catch (e) {
        console.error("AI move error:", e);
      }
    }
  } else if (type === "GET_VALID_MOVES") {
    const { pieceIdx } = data;
    const moves = pyodide
      .runPython(`get_valid_moves_for_piece(${pieceIdx})`)
      .toJs();
    // Convert list of tuples to list of objects
    const movesArray = moves.map((m) => ({ row: m[0], col: m[1] }));
    postMessage({ type: "VALID_MOVES", moves: movesArray });
  } else if (type === "RESET") {
    // Store the player's side choice
    playerSide = data?.playerSide || "white";
    console.log("RESET: playerSide =", playerSide);

    pyodide.runPython(`reset_game()`);
    const board = pyodide.runPython(`get_board_state()`).toJs();
    const currentPlayer = pyodide.runPython(`get_current_player()`);

    postMessage({
      type: "STATE_UPDATE",
      board,
      currentPlayer,
      gameOver: false,
      winner: 0,
      lastMove: null,
    });

    if (playerSide === "black") {
      console.log("RESET: Player chose black, AI (white) will move first");
      postMessage({ type: "AI_THINKING" });

      try {
        console.log("RESET: Calling get_ai_move()...");
        const aiMove = await pyodide.runPythonAsync(`get_ai_move()`);
        console.log("RESET: AI move result:", aiMove);
        const [aiPieceIdx, aiRow, aiCol] = aiMove.toJs();
        console.log("RESET: AI move parsed:", aiPieceIdx, aiRow, aiCol);

        // Execute AI move
        pyodide.runPython(`make_move(${aiPieceIdx}, ${aiRow}, ${aiCol})`);
        const boardAfterAI = pyodide.runPython(`get_board_state()`).toJs();
        const currentPlayerAfterAI = pyodide.runPython(`get_current_player()`);
        const gameOverResultAI = pyodide.runPython(`check_game_over()`).toJs();
        const gameOverAI = gameOverResultAI[0];
        const winnerAI = gameOverResultAI[1];

        console.log("RESET: AI move completed, sending STATE_UPDATE");
        postMessage({
          type: "STATE_UPDATE",
          board: boardAfterAI,
          currentPlayer: currentPlayerAfterAI,
          gameOver: gameOverAI,
          winner: winnerAI,
          lastMove: { row: aiRow, col: aiCol },
        });
      } catch (e) {
        console.error("AI move error:", e);
      }
    }
  }
};

// Expose inference function to Python
self.runInference = async (flattenedStates) => {
  console.log(
    "runInference called with",
    flattenedStates?.length || 0,
    "states"
  );
  // flattenedStates comes from Python, so it might be a PyProxy.
  // We must convert it to a pure JS object before sending via postMessage.
  let statesJs = flattenedStates;
  if (flattenedStates && typeof flattenedStates.toJs === "function") {
    statesJs = flattenedStates.toJs();
  }

  // Send to main thread for ONNX execution
  return new Promise((resolve) => {
    const id = Math.random().toString(36).substring(7);
    console.log("runInference: sending RUN_INFERENCE with id", id);

    const handler = (event) => {
      if (event.data.type === "INFERENCE_RESULT" && event.data.id === id) {
        console.log("runInference: received INFERENCE_RESULT for id", id);
        self.removeEventListener("message", handler);
        resolve(event.data.result);
      }
    };

    self.addEventListener("message", handler);
    postMessage({ type: "RUN_INFERENCE", id, states: statesJs });
  });
};
