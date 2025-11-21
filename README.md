# vrtigo

An experimental C++20 VRT I/O library co-authored with AI

## Goals

- Lightweight, header‑only C++20 library for VITA 49.2 packet building and parsing.
- Targeting high-performance and low-latency applications.
- Compile‑time packet definitions with automatic size/structure validation.
- Allocation-free so packets operate entirely on user-provided buffers.
- Clean API for building and parsing VRT packets

## Approach

This library is initially being co-developed by a half-decent programmer using AI agents such as Claude Code, OpenAI Codex, and GitHub Copilot.

This library is expected to have little-to-no hand-written code.  

Initial attempts will focus making fast progress, so AI agents will be given a little more leeway than might be given in a production environment. 

The hope is that the API will stabilize quickly, enabling a community of users to validate and improve interoperability and performance.

## Development Environment

- C++20
- CMake build system
- Unit tests with GoogleTest
- Makefile wrapper for common tasks
- Continuous Integration with GitHub Actions










