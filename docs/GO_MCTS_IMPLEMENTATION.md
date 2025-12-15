# Go MCTS Implementation Summary

## Overview

The MCTS (Monte Carlo Tree Search) algorithm has been implemented in Go for maximum performance. This provides significant speedup over the Python implementation by moving tree operations, game state management, and move generation to compiled Go code, while keeping neural network evaluation in Python.

## Architecture

### Data Flow

```
Python (mcts.py)
  ↓ Game state (5x5 int8)
mcts_go.py (ctypes wrapper)
  ↓ C pointer (zero-copy)
libmctsengine.so (Go compiled)
  ↓ Tree search operations
  ↓ When evaluation needed: canonical state (5x5 float32)
mcts_go.py
  ↓ numpy array
Python (neural network evaluation)
  ↓ Policy (50 floats) + Value (1 float)
mcts_go.py
  ↓ C pointer (zero-copy)
libmctsengine.so
  ↓ Final policy (50 floats)
mcts_go.py
  ↓ numpy array
Python (mcts.py)
```

### Key Components

1. **`mcts_engine.go`** - Core Go MCTS implementation
   - `MCTSNode` - Tree node structure
   - `MCTSContext` - Search context
   - `MCTSCreate()` - Create MCTS context
   - `MCTSSearchStep()` - Perform one search step (iterative API)
   - `MCTSProvideEvaluation()` - Provide neural network evaluation
   - `MCTSGetPolicy()` - Get final policy distribution
   - `MCTSDestroy()` - Clean up context

2. **`mcts_go.py`** - Python ctypes wrapper
   - `GoMCTS` class - High-level interface
   - Handles numpy array ↔ C array conversion
   - Manages iterative search loop
   - Graceful fallback if library not available

3. **`mcts.py`** - Updated Python MCTS
   - Optional Go MCTS integration
   - Falls back to Python implementation if Go unavailable
   - Same interface as before

## API Design

The Go MCTS uses an iterative API to allow Python to provide neural network evaluations:

1. **Create context**: `MCTSCreate(state, current_player, num_simulations, cPuct)`
2. **Search loop**:
   - Call `MCTSSearchStep()` which returns:
     - `0` = Need root evaluation
     - `1` = Need leaf evaluation  
     - `2` = Simulation done (terminal), continue
     - `3` = All simulations complete
   - If evaluation needed, provide via `MCTSProvideEvaluation(policy, value)`
   - Repeat until status is `3`
3. **Get policy**: `MCTSGetPolicy(policy_output)`
4. **Destroy**: `MCTSDestroy(ctx)`

## Performance Benefits

1. **Faster tree operations** - Go structs vs Python objects
2. **Faster game state copying** - Direct memory operations
3. **Better memory management** - Less GC pressure
4. **Zero-copy data passing** - Direct C pointer access
5. **Compiled code** - Native performance vs interpreted Python

Expected speedup: **2-4x** for MCTS operations (excluding neural network evaluation time).

## Usage

The Go MCTS is automatically used when:
- `use_go_mcts=True` (default in `MCTS.__init__`)
- `libmctsengine.so` is present
- Library loads successfully

If unavailable, automatically falls back to Python MCTS.

### Example

```python
from mcts import MCTS
from model import AlphaZeroNet

model = AlphaZeroNet()
mcts = MCTS(model, num_simulations=100, c_puct=1.0, use_go_mcts=True)

# Use as before
policy = mcts.search(game)
```

## Building

```bash
# Build MCTS library
make build-mcts

# Or build both game engine and MCTS
make build-all

# Or manually
go build -buildmode=c-shared -ldflags="-s -w" -trimpath -o libmctsengine.so mcts_engine.go game_engine.go
```

## Integration with Existing Code

The Go MCTS is a drop-in replacement for Python MCTS:
- Same `MCTS` class interface
- Same `search()` method signature
- Same return format (policy distribution)
- Automatic fallback to Python if Go unavailable

No changes needed to training or inference code!

## Implementation Details

### Tree Structure
- Nodes stored as Go structs (faster than Python dicts)
- Children stored in Go maps (efficient lookups)
- State stored as fixed-size arrays (no heap allocation)

### Game Operations
- Uses existing Go game engine functions (`MakeMove`, `GetValidMoves`, `GetCanonicalState`)
- Zero-copy state passing via C pointers
- Direct memory access for maximum performance

### Neural Network Integration
- Python provides evaluation callback function
- Go calls Python when evaluation needed
- Policy and value passed via numpy arrays (zero-copy)

## Testing

```bash
# Test Go MCTS (when test file is created)
python -m pytest test_mcts_go.py -v
```

## Notes

- Neural network evaluation remains in Python (PyTorch requirement)
- Tree search and game operations are in Go
- Policy flipping for Black player handled in Go
- Action masking and normalization handled in Go
- Visit count extraction and policy normalization in Go

## Future Improvements

1. **Parallel simulations** - Use goroutines for parallel MCTS simulations
2. **Batch evaluation** - Batch multiple state evaluations together
3. **Tree reuse** - Reuse tree between moves (requires state management)
4. **Memory pooling** - Pool node allocations for better performance

