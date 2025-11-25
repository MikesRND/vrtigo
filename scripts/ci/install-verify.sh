#!/bin/bash
# Install verification - tests cmake install and package consumption
# This is the single source of truth for install-verify logic

set -euo pipefail

# Build directory (can be overridden)
BUILD_DIR="${1:-build-install}"

# Install staging prefix (can be overridden)
INSTALL_PREFIX="${2:-$(pwd)/install-test}"

# Detect number of cores
if command -v nproc &>/dev/null; then
    NPROC=$(nproc)
elif command -v sysctl &>/dev/null; then
    NPROC=$(sysctl -n hw.ncpu)
else
    NPROC=4
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Install Verification Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Build directory: $BUILD_DIR"
echo "Install prefix: $INSTALL_PREFIX"
echo "Parallel jobs: $NPROC"
echo ""

# Clean build and install directories
rm -rf "$BUILD_DIR" "$INSTALL_PREFIX"
mkdir -p "$BUILD_DIR"

# Step 1: Build vrtigo library
echo "Building vrtigo..."
cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DVRTIGO_BUILD_TESTS=OFF \
    -DVRTIGO_BUILD_EXAMPLES=OFF

cmake --build "$BUILD_DIR" -j"$NPROC"

# Step 2: Install to staging prefix
echo ""
echo "Installing to staging prefix..."
cmake --install "$BUILD_DIR" --prefix "$INSTALL_PREFIX"

# Step 3: Build out-of-tree consumer
echo ""
echo "Building out-of-tree consumer..."
CONSUMER_BUILD="$BUILD_DIR-consumer"
rm -rf "$CONSUMER_BUILD"
cmake -S tests/install_verification -B "$CONSUMER_BUILD" \
    -DCMAKE_PREFIX_PATH="$INSTALL_PREFIX"
cmake --build "$CONSUMER_BUILD" -j"$NPROC"

# Step 4: Run consumer
echo ""
echo "Running consumer..."
"$CONSUMER_BUILD/minimal_consumer"

# Cleanup
rm -rf "$INSTALL_PREFIX" "$CONSUMER_BUILD"

echo ""
echo "✓ Install verification passed"
