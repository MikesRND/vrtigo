# CI Scripts

This directory contains the **single source of truth** for all CI validation logic.

## Philosophy: Zero Drift

These scripts ensure that **local validation and CI validation are identical**:

- ✅ **No drift** - change the script once, both local and CI update
- ✅ **Easy testing** - run scripts standalone: `./scripts/ci/quick-check.sh`
- ✅ **Self-documenting** - the script is the spec
- ✅ **Portable** - works with any CI system (GitHub Actions, GitLab CI, etc.)

## Available Scripts

All scripts take an optional `BUILD_DIR` argument (defaults shown below):

### `quick-check.sh` (Required Gate)
**Matches**: CI `quick-check` job (required status)
**Usage**: `./scripts/ci/quick-check.sh [build-quick]`
**What**: Fast validation with Release build, all features enabled
**When**: Run before every `git push`

```bash
# Direct
./scripts/ci/quick-check.sh

# Via Make
make quick-check
```

### `debug-build.sh` (Advisory)
**Matches**: CI `debug-build` job
**Usage**: `./scripts/ci/debug-build.sh [build-debug]`
**What**: Debug build to catch assertions
**When**: Testing debug-only code paths

```bash
# Direct
./scripts/ci/debug-build.sh

# Via Make
make ci-debug
```

### `clang-build.sh` (Advisory)
**Matches**: CI `clang-build` job
**Usage**: `./scripts/ci/clang-build.sh [build-clang]`
**What**: Clang compiler test (different warnings than GCC)
**When**: Catching compiler-specific issues

```bash
# Direct - uses system clang++
./scripts/ci/clang-build.sh

# Override compiler
CXX=clang++-16 ./scripts/ci/clang-build.sh

# Via Make
make ci-clang
```

### `install-verify.sh` (Advisory)
**Matches**: CI `install-verification` job
**Usage**: `./scripts/ci/install-verify.sh [build-install] [install-prefix]`
**What**: Verifies CMake install and out-of-tree consumption
**When**: Before releases or when changing CMake config
**Note**: Consumer builds in `$BUILD_DIR-consumer` (not in source tree)

```bash
# Direct
./scripts/ci/install-verify.sh

# Custom prefix
./scripts/ci/install-verify.sh build /tmp/my-install

# Via Make
make ci-install-verify
```

### `coverage.sh` (Informational)
**Matches**: CI `coverage` job
**Usage**: `./scripts/ci/coverage.sh [build-coverage]`
**What**: Generates code coverage report
**When**: Checking test coverage locally

```bash
# Direct
./scripts/ci/coverage.sh

# Via Make
make ci-coverage

# View report
open build-coverage/coverage_html/index.html
```

### `python-test.sh` (Advisory)
**Matches**: CI `python-test` job
**Usage**: `./scripts/ci/python-test.sh [build-python] [--no-test] [--force-recreate]`
**What**: Build Python bindings and run pytest
**Venv**: Reuses `bindings/python/.venv`, upgrades deps each run
**Requires**: Python 3.11+ (checked at runtime)

**Flags**:
- `--no-test` - build only, skip pytest
- `--force-recreate` - nuke venv if corrupted

```bash
# Direct (build + test)
./scripts/ci/python-test.sh

# Build only
./scripts/ci/python-test.sh --no-test

# Force venv recreation
./scripts/ci/python-test.sh --force-recreate

# Via Make
make python-test    # build + test
make python         # build only

# Override Python interpreter
PYTHON_BIN=python3.12 ./scripts/ci/python-test.sh
```

## Environment Variables

All scripts respect these environment variables:

- `CXX` - C++ compiler (default: `g++` or `clang++`)
- `PYTHON_BIN` - Python interpreter for python-test.sh (default: `python3`)
- Build directory is always customizable via first argument

## Build Caching (ccache)

All scripts automatically use **ccache** if it's available on the system:
- Checks for `ccache` with `command -v ccache`
- Automatically adds `-DCMAKE_CXX_COMPILER_LAUNCHER=ccache` if found
- No configuration needed - just install ccache

**Install ccache:**
```bash
# Ubuntu/Debian
sudo apt-get install ccache

# macOS
brew install ccache
```

**Result**: Second runs are typically 5-10x faster!

## How It Works

### Local (Makefile)
```make
quick-check:
    @./scripts/ci/quick-check.sh build-quick
```

### CI (GitHub Actions)
```yaml
- name: Quick Check
  run: ./scripts/ci/quick-check.sh build
  env:
    CXX: g++-13
```

**Result**: Both call the same script → zero drift!

## Build Configuration

All scripts (except `install-verify.sh`) build with:
- Tests: **ON**
- Benchmarks: **OFF**

IO helpers are always available (no optional toggle).

The `install-verify.sh` script builds a minimal library (tests/benchmarks OFF) to test the install workflow.

## Common Workflows

### Before pushing
```bash
./scripts/ci/quick-check.sh
```

### Full local validation
```bash
./scripts/ci/quick-check.sh
./scripts/ci/debug-build.sh
./scripts/ci/clang-build.sh
```

### Check coverage
```bash
./scripts/ci/coverage.sh
open build-coverage/coverage_html/index.html
```

### Before release
```bash
./scripts/ci/install-verify.sh
```

## Modifying CI Behavior

**To change CI logic:**
1. Edit the relevant script in `scripts/ci/`
2. Test locally: `./scripts/ci/<script>.sh`
3. Commit - both local and CI now use the new logic

**Do NOT** edit the GitHub Actions workflow for build logic - only for:
- Dependency installation
- Caching configuration
- Artifact uploads
- Job dependencies/ordering

## Parallel Execution

All scripts automatically use all available CPU cores via `nproc` (or `sysctl -n hw.ncpu` on macOS).

## Exit Codes

Scripts exit with:
- `0` - success
- `non-zero` - failure (stops execution immediately via `set -euo pipefail`)
