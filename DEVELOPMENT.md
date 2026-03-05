# DEVELOPMENT.md

Guidance for working in this repository.

## Project Overview

VRTIGO is a header-only C++20 library for creating and parsing VITA 49.2 VRT (VITA Radio Transport) packets. It targets zero-allocation, high-performance RF signal processing. Python bindings are provided via nanobind v2.4.0.

## Requirements

- Python 3.11+ for Python bindings
- CMake 3.24+
- C++20 compiler (GCC 13+ or Clang 16+)

When changing version floors, update the code and docs together. The canonical
touchpoints are `bindings/python/CMakeLists.txt`, `bindings/python/pyproject.toml`,
and this file.

### Versioning

[release-please](https://github.com/googleapis/release-please) manages the project version
in three files via `x-release-please-version` markers:

- `CMakeLists.txt` (primary)
- `bindings/python/pyproject.toml`
- `.release-please-manifest.json` (derived mirror)

Do not edit versions manually — the `version-check` CI job fails if they diverge.

## Build Commands

Run these commands from the repository root unless noted otherwise.

```bash
# C++ (default build dir: build/)
make                    # Configure + build everything (Debug)
make test               # Build + run all C++ tests
make quick-check        # Release build validation (CI gate)

# Run a single C++ test
cd build && ctest -R <test_name> --output-on-failure   # e.g. -R roundtrip

# Python bindings (build dir: build/ with VRTIGO_BUILD_PYTHON=ON)
cmake --build build --target vrtigo_py vrtigo_stub
PYTHONPATH=build/bindings/python .venv/bin/pytest bindings/python/tests/ -v
PYTHONPATH=build/bindings/python .venv/bin/pytest bindings/python/tests/test_enums.py -v -k "some_pattern"

# Code quality
make format-check       # Check clang-format (CI gate)
make docs-check         # Check local markdown links
make format-fix         # Auto-fix formatting
make clang-tidy         # Static analysis
make autodocs           # Regenerate documentation extracted from tests
```

## Quickstart Autodoc

Quickstart pages are generated from tested examples in `tests/quickstart/`,
which helps keep the published snippets correct and current.

For marker format and doc-generation tooling details, see
`tests/quickstart/README.md`.

## Architecture

### Dual API Design

- **`vrtigo::typed`** — Compile-time packet builders for transmit. Template parameters encode packet structure; offsets/sizes resolved at compile time.
- **`vrtigo::dynamic`** — Runtime packet parsers for receive. `DataPacketView` and `ContextPacketView` parse from `std::span<const uint8_t>` with zero copies.

### Namespace Map

| Namespace | Purpose | Allocates? |
|-----------|---------|------------|
| `vrtigo::typed` | Compile-time packet building | No |
| `vrtigo::dynamic` | Runtime packet parsing | No |
| `vrtigo::utils` | Extras: file/PCAP/UDP I/O, SampleClock, SampleFramer. Not bound by the core zero-allocation contract; may allocate, throw, use OS APIs. | Yes |
| `vrtigo::detail` | Internal implementation — never include directly | — |
| `vrtigo::gpu` | CUDA-compatible packet parsing | No |

### Key Headers

- `include/vrtigo.hpp` — Umbrella include
- `include/vrtigo/dynamic.hpp` — Runtime parsing API (`DataPacketView`, `ContextPacketView`)
- `include/vrtigo/typed.hpp` — Compile-time building API
- `include/vrtigo/types.hpp` — `PacketType`, `TsiType`, `TsfType`, `ValidationError`
- `include/vrtigo/timestamp.hpp`, `duration.hpp` — Picosecond-accurate time types
- `include/vrtigo/vrtigo_io.hpp` — File/PCAP I/O readers and writers

### Python Bindings

Located in `bindings/python/`. Binding source files are in `bindings/python/src/`:
- `owning_packet_bindings.hpp` — `DataPacket`, `ContextPacket` (own their data)
- `packet_view_bindings.hpp` — `DataPacketView`, `ContextPacketView` (non-owning views)
- `py_types.hpp` — `PyDataPacket`, `PyContextPacket` wrapper types

nanobind v2.4.0 has no `nb::memoryview`; use `PyMemoryView_FromMemory` + `nb::steal()` with `nb::keep_alive<0,1>()` for zero-copy returns.

Test helpers in `bindings/python/tests/test_packets.py` provide `DataPacketBuilder` and `ContextPacketBuilder` for constructing valid VRT packets in Python tests.

Type stubs are auto-generated at `build/bindings/python/vrtigo.pyi` by the `vrtigo_stub` target.

## Conventions

### C++ Style
- Items not specified here should generally follow the C++ Core Guidelines.
- 100-column line limit, 4-space indent, LLVM-based clang-format
- Naming: `CamelCase` classes, `lower_case` functions/variables, `lower_case_` private members, `kCamelCase` constexpr
- Pointers left-aligned: `int* ptr`
- Prefer snake_case getters and `set_`-prefixed setters; getters should generally be `const noexcept`
- Prefer `enum class` with explicit underlying types and snake_case enumerators
- No MSVC; requires GCC 13+ or Clang 16+

### Dependencies
- Core library: zero external dependencies (header-only)
- Tests: GoogleTest v1.14.0 (auto-fetched via FetchContent)
- Python: nanobind v2.4.0, scikit-build-core, Python 3.11+

### CI Pipeline (GitHub Actions)
- **Required gates**: `format-check`, `quick-check`, `version-check`
- **Advisory**: `static-analysis`, `debug-build`, `clang-build`, `python-bindings`
- **PR title lint**: PR titles must follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `chore:`, etc.)
- Run `make format-check && make quick-check` before pushing

### Release Process

Releases use [release-please](https://github.com/googleapis/release-please) with squash merge on main.

1. **PR titles** follow Conventional Commit format — enforced by the `pr-title-lint` CI check
2. **Squash merge** PRs to main — the PR title becomes the commit message
3. release-please auto-opens a **Release PR** that bumps versions and updates `CHANGELOG.md`
4. Merge the Release PR → release-please creates a git tag (`vX.Y.Z`) and GitHub Release
5. Maintainer runs `publish.yml` via **Actions → Run workflow** with the tag (e.g. `v0.1.0`)
6. Publish workflow: verifies GitHub Release exists → runs CI → builds wheels → uploads to GitHub Release
7. To also publish to **PyPI**: check the `pypi` checkbox in the workflow dispatch form

After the first release: remove `release-as` from `release-please-config.json` so
subsequent versions are auto-determined from commit messages.
