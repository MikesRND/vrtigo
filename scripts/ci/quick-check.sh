#!/bin/bash
# Quick validation check - matches CI required gate
# This is the single source of truth for quick-check logic

set -euo pipefail

# Build directory (can be overridden)
BUILD_DIR="${1:-build-quick}"

# Detect number of cores
if command -v nproc &>/dev/null; then
    NPROC=$(nproc)
elif command -v sysctl &>/dev/null; then
    NPROC=$(sysctl -n hw.ncpu)
else
    NPROC=4
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Quick Validation Check (CI Required Gate)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Build directory: $BUILD_DIR"
echo "Parallel jobs: $NPROC"
echo ""

# Clean and create build directory
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Configure
echo "Configuring..."

# Use ccache if available
CCACHE_FLAG=""
if command -v ccache &>/dev/null; then
    CCACHE_FLAG="-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
    echo "Using ccache"
fi

cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DVRTIGO_BUILD_TESTS=ON \
    -DVRTIGO_BUILD_EXAMPLES=ON \
    -DVRTIGO_BUILD_BENCHMARKS=OFF \
    -DVRTIGO_FETCH_DEPENDENCIES=ON \
    $CCACHE_FLAG

# Build
echo ""
echo "Building..."
cmake --build "$BUILD_DIR" -j"$NPROC"

# Test
echo ""
echo "Testing..."
cd "$BUILD_DIR" && ctest --output-on-failure --verbose

echo ""
echo "✓ Quick check passed - CI gate will pass"
