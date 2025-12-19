#!/bin/bash
# GPU build and test - requires CUDA toolkit and GPU runtime
# This is the single source of truth for GPU test logic
#
# This script uses direct nvcc compilation to bypass CMake's CUDA language
# support, which may not yet support newer CUDA versions (e.g., CUDA 13.x).

set -euo pipefail

# Build directory (can be overridden)
BUILD_DIR="${1:-build-gpu}"

# Check CUDA availability
if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not found. Install CUDA toolkit."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep release | awk '{print $6}')

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "GPU Build and Test"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Build directory: $BUILD_DIR"
echo "CUDA version: $CUDA_VERSION"
echo ""

# Clean and create build directory
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Step 1: Build GTest using CMake (CPU-only, no CUDA needed)
echo "Building GTest..."
mkdir -p "$BUILD_DIR/gtest"
cmake -S . -B "$BUILD_DIR/gtest" \
    -DCMAKE_BUILD_TYPE=Release \
    -DVRTIGO_BUILD_TESTS=ON \
    -DVRTIGO_FETCH_DEPENDENCIES=ON \
    -DVRTIGO_ENABLE_CUDA=OFF 2>&1 | grep -E "(GTest|Fetching|Configuring done)" || true

# Build just gtest
cmake --build "$BUILD_DIR/gtest" --target gtest gtest_main 2>/dev/null || {
    # If gtest target doesn't exist, build all (which includes gtest)
    cmake --build "$BUILD_DIR/gtest" -j4 2>&1 | grep -v "^make\[" | grep -v "^Building" || true
}

# Find GTest libraries
GTEST_LIB=$(find "$BUILD_DIR/gtest" -name "libgtest.a" 2>/dev/null | head -1)
GTEST_MAIN_LIB=$(find "$BUILD_DIR/gtest" -name "libgtest_main.a" 2>/dev/null | head -1)
GTEST_INCLUDE=$(find "$BUILD_DIR/gtest" -path "*googletest-src/googletest/include" -type d 2>/dev/null | head -1)

if [[ -z "$GTEST_LIB" ]] || [[ -z "$GTEST_MAIN_LIB" ]] || [[ -z "$GTEST_INCLUDE" ]]; then
    echo "ERROR: Could not find GTest after build"
    echo "GTEST_LIB: $GTEST_LIB"
    echo "GTEST_MAIN_LIB: $GTEST_MAIN_LIB"
    echo "GTEST_INCLUDE: $GTEST_INCLUDE"
    exit 1
fi

echo "Found GTest: $GTEST_LIB"
echo ""

# Step 2: Compile GPU tests with nvcc
echo "Compiling GPU tests with nvcc..."

# Common compile flags
NVCC_FLAGS="-std=c++20 -O2 -DVRTIGO_ENABLE_CUDA -I include -I $GTEST_INCLUDE"

# Compile complex_test.cu
echo "  Compiling complex_test..."
nvcc $NVCC_FLAGS \
    tests/gpu/complex_test.cu \
    -o "$BUILD_DIR/gpu_complex_test" \
    -L "$(dirname "$GTEST_LIB")" \
    -lgtest -lgtest_main -lpthread

# Compile integration_test.cu
echo "  Compiling integration_test..."
nvcc $NVCC_FLAGS \
    tests/gpu/integration_test.cu \
    -o "$BUILD_DIR/gpu_integration_test" \
    -L "$(dirname "$GTEST_LIB")" \
    -lgtest -lgtest_main -lpthread

echo ""

# Step 3: Run tests
echo "Running GPU tests..."
echo ""

echo "--- Complex Test ---"
"$BUILD_DIR/gpu_complex_test" --gtest_color=yes
echo ""

echo "--- Integration Test ---"
"$BUILD_DIR/gpu_integration_test" --gtest_color=yes
echo ""

echo "✓ GPU tests passed"
