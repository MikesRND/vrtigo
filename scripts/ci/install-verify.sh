#!/bin/bash
# Install verification - tests cmake install and package consumption
# This is the single source of truth for install-verify logic
#
# Usage: install-verify.sh [BUILD_DIR] [INSTALL_PREFIX] [--gpu]
#   --gpu  Enable GPU/CUDA verification (requires nvcc and GPU)

set -euo pipefail

# Parse arguments
BUILD_DIR="build-install"
INSTALL_PREFIX="$(pwd)/install-test"
TEST_GPU=false

for arg in "$@"; do
    case $arg in
        --gpu)
            TEST_GPU=true
            ;;
        *)
            if [[ "$BUILD_DIR" == "build-install" ]]; then
                BUILD_DIR="$arg"
            else
                INSTALL_PREFIX="$arg"
            fi
            ;;
    esac
done

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
echo "GPU test: $TEST_GPU"
echo ""

# Check CUDA availability if GPU test requested
if $TEST_GPU; then
    if ! command -v nvcc &>/dev/null; then
        echo "ERROR: --gpu requested but nvcc not found"
        exit 1
    fi
    CUDA_VERSION=$(nvcc --version | grep release | awk '{print $6}')
    echo "CUDA version: $CUDA_VERSION"
    echo ""
fi

# Clean build and install directories
rm -rf "$BUILD_DIR" "$INSTALL_PREFIX"
mkdir -p "$BUILD_DIR"

# Step 1: Build vrtigo library
echo "Building vrtigo..."
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DVRTIGO_BUILD_TESTS=OFF
    -DVRTIGO_BUILD_EXAMPLES=OFF
)
if $TEST_GPU; then
    CMAKE_ARGS+=(-DVRTIGO_ENABLE_CUDA=ON)
fi

cmake -S . -B "$BUILD_DIR" "${CMAKE_ARGS[@]}"
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

CONSUMER_ARGS=(-DCMAKE_PREFIX_PATH="$INSTALL_PREFIX")
if $TEST_GPU; then
    CONSUMER_ARGS+=(-DVRTIGO_TEST_GPU=ON)
fi

cmake -S tests/install_verification -B "$CONSUMER_BUILD" "${CONSUMER_ARGS[@]}"
cmake --build "$CONSUMER_BUILD" -j"$NPROC"

# Step 4: Run consumers
echo ""
echo "Running consumer..."
"$CONSUMER_BUILD/minimal_consumer"

if $TEST_GPU; then
    if [[ ! -x "$CONSUMER_BUILD/gpu_consumer" ]]; then
        echo "ERROR: gpu_consumer was not built. Was vrtigo installed with -DVRTIGO_ENABLE_CUDA=ON?"
        exit 1
    fi
    echo ""
    echo "Running GPU consumer..."
    "$CONSUMER_BUILD/gpu_consumer"
fi

# Cleanup
rm -rf "$INSTALL_PREFIX" "$CONSUMER_BUILD"

echo ""
echo "✓ Install verification passed"
