#!/bin/bash
set -e  # Exit on error

echo "========================================="
echo "  DRP-AI MOIL + YOLOv8 Build Script"
echo "========================================="

# Check if Poky SDK is sourced
if [ -z "$SDKTARGETSYSROOT" ]; then
    echo "WARNING: Poky SDK environment not detected."
    echo "Please source the SDK setup script first:"
    echo "  source /opt/poky/3.1.31/environment-setup-aarch64-poky-linux"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check TVM_HOME
if [ -z "$TVM_HOME" ]; then
    echo "WARNING: TVM_HOME not set. DRP-AI runtime may not load."
    echo "Please export TVM_HOME=/path/to/tvm"
fi

# Clean previous build
echo "Cleaning previous build..."
rm -rf build

# Create build directory
echo "Creating build directory..."
mkdir -p build
cd build

# Run CMake
echo "Running CMake..."
if ! cmake ..; then
    echo "ERROR: CMake configuration failed!"
    exit 1
fi

# Build with make
echo "Building with make (using 8 parallel jobs)..."
if ! make -j8; then
    echo "ERROR: Build failed!"
    exit 1
fi

echo ""
echo "========================================="
echo "  Build successful!"
echo "========================================="
echo "Executable: build/moildev_app+DrpAiYolo"
echo ""
echo "To run:"
echo "  cd build"
echo "  ./moildev_app+DrpAiYolo 0"
echo "========================================="

