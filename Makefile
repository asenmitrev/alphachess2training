.PHONY: build build-mcts clean test

# Build the Go game engine shared library
build:
	go build -buildmode=c-shared -ldflags="-s -w" -trimpath -o libgameengine.so game_engine.go

# Build the Go MCTS engine shared library
build-mcts:
	go build -buildmode=c-shared -ldflags="-s -w" -trimpath -o libmctsengine.so mcts_engine.go game_engine.go

# Build both libraries
build-all: build build-mcts

# Clean build artifacts
clean:
	rm -f libgameengine.so libgameengine.h libmctsengine.so libmctsengine.h

# Build for different platforms (if needed)
build-linux:
	GOOS=linux GOARCH=amd64 go build -buildmode=c-shared -ldflags="-s -w" -trimpath -o libgameengine.so game_engine.go
	GOOS=linux GOARCH=amd64 go build -buildmode=c-shared -ldflags="-s -w" -trimpath -o libmctsengine.so mcts_engine.go game_engine.go

build-darwin:
	GOOS=darwin GOARCH=amd64 go build -buildmode=c-shared -ldflags="-s -w" -trimpath -o libgameengine.so game_engine.go
	GOOS=darwin GOARCH=amd64 go build -buildmode=c-shared -ldflags="-s -w" -trimpath -o libmctsengine.so mcts_engine.go game_engine.go

build-windows:
	GOOS=windows GOARCH=amd64 go build -buildmode=c-shared -ldflags="-s -w" -trimpath -o libgameengine.dll game_engine.go
	GOOS=windows GOARCH=amd64 go build -buildmode=c-shared -ldflags="-s -w" -trimpath -o libmctsengine.dll mcts_engine.go game_engine.go

# Default target
all: build-all

