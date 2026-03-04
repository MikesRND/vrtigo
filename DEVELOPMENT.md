# DEVELOPMENT.md

Guidance for working in this repository.

## Project Overview

VRTIGO is a header-only C++20 library for creating and parsing VITA 49.2 VRT (VITA Radio Transport) packets. It targets zero-allocation, high-performance RF signal processing. Python bindings are provided via nanobind v2.4.0.

## Requirements

- Python 3.11+ for Python bindings
- CMake 3.24+
- C++20 compiler (GCC 13+ or Clang 16+)

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
make format-fix         # Auto-fix formatting
make clang-tidy         # Static analysis
```

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
- **Required gates**: `format-check`, `quick-check`
- **Advisory**: `static-analysis`, `debug-build`, `clang-build`, `python-bindings`
- Run `make format-check && make quick-check` before pushing
