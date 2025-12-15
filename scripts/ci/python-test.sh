#!/bin/bash
# Python bindings build and test - matches CI python-test job
# This is the single source of truth for Python bindings logic

set -euo pipefail

VENV_DIR="bindings/python/.venv"
BUILD_DIR="build-python"
FORCE_RECREATE=0
RUN_TESTS=1

# Parse args: flags and one optional positional BUILD_DIR
POSITIONAL_SET=0
for arg in "$@"; do
    case "$arg" in
        --force-recreate) FORCE_RECREATE=1 ;;
        --no-test) RUN_TESTS=0 ;;
        -*) echo "Unknown flag: $arg"; exit 1 ;;
        *)
            if [[ "$POSITIONAL_SET" == "1" ]]; then
                echo "Error: unexpected argument '$arg' (build dir already set to '$BUILD_DIR')"
                exit 1
            fi
            BUILD_DIR="$arg"
            POSITIONAL_SET=1
            ;;
    esac
done

# Python interpreter (override with PYTHON_BIN env var)
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Verify Python 3.11+
if ! "$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
    PY_VERSION=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo "Error: Python 3.11+ required, found $PY_VERSION"
    exit 1
fi
PY_VERSION=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

# Detect number of cores
if command -v nproc &>/dev/null; then
    NPROC=$(nproc)
elif command -v sysctl &>/dev/null; then
    NPROC=$(sysctl -n hw.ncpu)
else
    NPROC=4
fi

# Use ccache if available
CCACHE_FLAG=""
if command -v ccache &>/dev/null; then
    CCACHE_FLAG="-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Python Bindings Build + Test"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Build directory: $BUILD_DIR"
echo "Python: $PY_VERSION ($PYTHON_BIN)"
echo "Parallel jobs: $NPROC"
echo ""

# Venv: recreate if --force-recreate or missing, otherwise reuse
if [[ "$FORCE_RECREATE" == "1" ]] || [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment..."
    rm -rf "$VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "Reusing existing virtual environment..."
fi

# Always upgrade pip + deps (fast if unchanged)
echo "Upgrading pip and dependencies..."
"$VENV_DIR/bin/python" -m pip install --quiet --upgrade pip
"$VENV_DIR/bin/python" -m pip install --quiet --upgrade -e bindings/python[dev]

# Clean build dir (matches other CI scripts)
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Configure
echo ""
echo "Configuring..."
# Set VIRTUAL_ENV so CMake's FindPython prefers our venv
VIRTUAL_ENV="$(pwd)/$VENV_DIR" cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DVRTIGO_BUILD_PYTHON=ON \
    -DVRTIGO_FETCH_DEPENDENCIES=ON \
    -DPython_ROOT_DIR="$(pwd)/$VENV_DIR" \
    -DPython_FIND_VIRTUALENV=ONLY \
    $CCACHE_FLAG

# Build
echo ""
echo "Building..."
cmake --build "$BUILD_DIR" --target vrtigo_py vrtigo_stub -j"$NPROC"

# Test (unless --no-test)
if [[ "$RUN_TESTS" == "1" ]]; then
    echo ""
    echo "Testing..."
    PYTHONPATH="$BUILD_DIR/bindings/python" "$VENV_DIR/bin/python" -m pytest bindings/python/tests -v
    echo ""
    echo "✓ Python tests passed"
else
    echo ""
    echo "✓ Python bindings built (tests skipped)"
fi
