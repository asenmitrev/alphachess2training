.PHONY: build clean test

# Build the Go shared library
build:
	go build -buildmode=c-shared -ldflags="-s -w" -trimpath -o libgameengine.so game_engine.go

# Clean build artifacts
clean:
	rm -f libgameengine.so libgameengine.h

# Build for different platforms (if needed)
build-linux:
	GOOS=linux GOARCH=amd64 go build -buildmode=c-shared -ldflags="-s -w" -trimpath -o libgameengine.so game_engine.go

build-darwin:
	GOOS=darwin GOARCH=amd64 go build -buildmode=c-shared -ldflags="-s -w" -trimpath -o libgameengine.so game_engine.go

build-windows:
	GOOS=windows GOARCH=amd64 go build -buildmode=c-shared -ldflags="-s -w" -trimpath -o libgameengine.dll game_engine.go

# Default target
all: build

