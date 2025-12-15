# Building the Go Game Engine

The game engine has been rewritten in Go for maximum performance. This guide explains how to build and use it.

## Prerequisites

1. **Go 1.21 or later** - Install from https://golang.org/dl/ or via package manager:
   ```bash
   # Ubuntu/Debian
   sudo apt install golang-go
   
   # Or use snap
   sudo snap install go --classic
   ```

2. **C compiler** (usually GCC) - Required for building C shared libraries:
   ```bash
   sudo apt install build-essential
   ```

## Building the Library

1. **Build the shared library:**
   ```bash
   make build
   ```
   
   This will create `libgameengine.so` in the project root.

2. **Alternative: Build manually**
   ```bash
   go build -buildmode=c-shared -ldflags="-s -w" -trimpath -o libgameengine.so game_engine.go
   ```

## Usage

The Go engine is automatically used when:
- `USE_GO_ENGINE = True` in `game.py` (default)
- `libgameengine.so` is present in the project directory
- `game_engine_go.py` can successfully load the library

If the Go engine is unavailable, the code automatically falls back to the Python engine.

## Testing

Run the test suite to verify the Go engine works correctly:

```bash
python -m pytest test_go_engine.py -v
```

## Performance

The Go engine provides significant performance improvements:
- **Zero-copy tensor passing** - Direct memory access, no serialization
- **Optimized C bindings** - Minimal overhead between Python and Go
- **No JSON/dictionary overhead** - Pure tensor operations
- **Compiled code** - Native performance vs interpreted Python

Expected speedup: **5-10x** faster than Python engine for move execution.

## Troubleshooting

### Library not found
If you see `FileNotFoundError: libgameengine.so not found`:
1. Make sure you've built the library: `make build`
2. Check that `libgameengine.so` exists in the project root
3. Verify file permissions: `chmod +x libgameengine.so`

### Import errors
If `game_engine_go` fails to import:
1. Check that `libgameengine.so` is built and accessible
2. Verify Go is installed: `go version`
3. Try rebuilding: `make clean && make build`

### Fallback to Python engine
If the code falls back to Python engine:
- Check logs for error messages
- Verify `USE_GO_ENGINE = True` in `game.py`
- Ensure `GO_ENGINE_AVAILABLE = True` after import

## Platform Support

The Makefile includes targets for different platforms:
- `make build-linux` - Linux (default)
- `make build-darwin` - macOS
- `make build-windows` - Windows (creates `.dll`)

For cross-compilation, set `GOOS` and `GOARCH` environment variables.

