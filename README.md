<p align="center">
  <img src="docs/images/VRTigo-600x600.png" alt="vrtigo logo" width="128">
</p>

# VRTigo

[![CI](https://github.com/MikesRND/vrtigo/actions/workflows/ci.yml/badge.svg)](https://github.com/MikesRND/vrtigo/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![CMake](https://img.shields.io/badge/CMake-%3E%3D3.24-blue.svg)](https://cmake.org/)
[![Header-Only](https://img.shields.io/badge/Header--Only-Yes-green.svg)]()
[![Clang Format](https://img.shields.io/badge/code%20style-clang--format-blue)](https://clang.llvm.org/docs/ClangFormat.html)
[![Clang Tidy](https://img.shields.io/badge/linter-clang--tidy-blue)](https://clang.llvm.org/extra/clang-tidy/)

An experimental C++20 VRT I/O library co-authored with AI

## Overview

Header-only, allocation-free C++20 library for building and parsing VITA 49.2 VRT packets. Includes Python bindings and optional CUDA support.

See [DEVELOPMENT.md](DEVELOPMENT.md) for build instructions, architecture, and development conventions.

## Documentation

- [Documentation Index](docs/README.md) — Top-level guide to the user-facing docs
- [Quickstart Guide](docs/quickstart.md) — Executable examples for data packets, context packets, file I/O, and parsing
- [Python Bindings](docs/python-bindings.md) — Python packet, reader, and time-type overview
- [Timestamp Math](docs/timestamp-math.md) — Picosecond-accurate time arithmetic
- [CIF Field Access](docs/cif_access.md) — Context Information Field accessors
- [I/O Helpers](docs/utils/io-helpers.md) — File, PCAP, and UDP readers/writers
- [GPU Extensions](docs/gpu-extensions.md) — CUDA-compatible packet parsing
- [Endianness Model](docs/endianness-model.md) — Byte-order handling






