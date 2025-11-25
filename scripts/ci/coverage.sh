#!/bin/bash
# Coverage report generation - builds with coverage instrumentation
# This is the single source of truth for coverage logic

set -euo pipefail

# Check if coverage tools are available
if ! command -v lcov &>/dev/null; then
    echo "⚠ Coverage tools not found (lcov) - skipping coverage report"
    echo "  Install with: sudo apt-get install lcov"
    exit 0
fi

# Build directory (can be overridden)
BUILD_DIR="${1:-build-coverage}"

# Detect number of cores
if command -v nproc &>/dev/null; then
    NPROC=$(nproc)
elif command -v sysctl &>/dev/null; then
    NPROC=$(sysctl -n hw.ncpu)
else
    NPROC=4
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Coverage Report Generation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Build directory: $BUILD_DIR"
echo "Parallel jobs: $NPROC"
echo ""

# Clean and create build directory
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Configure with coverage
echo "Configuring with coverage instrumentation..."

# Use ccache if available
CCACHE_FLAG=""
if command -v ccache &>/dev/null; then
    CCACHE_FLAG="-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
    echo "Using ccache"
fi

cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="--coverage" \
    -DVRTIGO_BUILD_TESTS=ON \
    -DVRTIGO_BUILD_EXAMPLES=ON \
    -DVRTIGO_FETCH_DEPENDENCIES=ON \
    $CCACHE_FLAG

# Build
echo ""
echo "Building..."
cmake --build "$BUILD_DIR" -j"$NPROC"

# Test
echo ""
echo "Running tests..."
cd "$BUILD_DIR" && ctest --output-on-failure

# Generate coverage report
echo ""
echo "Generating coverage report..."

# Capture coverage data
lcov --capture --directory . --output-file coverage.info --ignore-errors mismatch 2>/dev/null

# Remove unwanted files from coverage
lcov --remove coverage.info \
    '/usr/*' \
    '*/tests/*' \
    '*/examples/*' \
    --output-file coverage_filtered.info 2>/dev/null || true

# Display coverage report
echo ""
echo "Coverage by file:"
lcov --list coverage_filtered.info || true

echo ""
echo "Overall summary:"
lcov --summary coverage_filtered.info || true

echo ""
echo "✓ Coverage report generated"
