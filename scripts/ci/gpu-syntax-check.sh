#!/bin/bash
# GPU header syntax check - no runtime needed, just validates CUDA compilation
# Can run in nvidia/cuda container without GPU
# This is the single source of truth for GPU syntax check logic

set -euo pipefail

# Build directory (can be overridden)
BUILD_DIR="${1:-build-gpu-syntax}"

# Check CUDA availability
if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not found. Install CUDA toolkit or use nvidia/cuda container."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep release | awk '{print $6}')

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "GPU Header Syntax Check (no runtime required)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Build directory: $BUILD_DIR"
echo "CUDA version: $CUDA_VERSION"
echo ""

# Clean and create build directory
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Direct nvcc compilation (bypasses CMake CUDA support issues with newer CUDA versions)
# This is simpler and more portable than relying on CMake's CUDA language support
echo "Compiling GPU headers with nvcc..."
nvcc -c --std=c++20 \
    -x cu \
    -I include \
    -DVRTIGO_ENABLE_CUDA \
    -diag-suppress 177 \
    tests/gpu/compile_check.cu \
    -o "$BUILD_DIR/compile_check.o"

echo ""
echo "✓ GPU headers compile successfully"
