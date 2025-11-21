# VRTIGO Design Principles

Technical architecture decisions that guide implementation. For code style and formatting, see style.md.

## Architecture Principles


- Target high-performance and low-latency applications
- **Header-only**: Entire library is header-only for easy integration
- **C++20**: Leverage modern features for safety and expressiveness (concepts, if constexpr, ranges)
- **Zero allocation**: Never allocates memory - operates only on user-provided buffers
- **Compile-time optimization**: Maximize compile-time computation to minimize runtime overhead
- **Pay for what you use**: Unused features compile away completely
- **Type safety**: Invalid packets fail at compile time, not runtime
- **VITA 49.2 compliance**: Spec rules enforced by type system
- **No exceptions**: Error codes and optionals for error handling

> **Note:** Optional utilities in `vrtigo::utils` (e.g., file reader) do not strictly
follow these principles, as they may allocate internal scratch buffers and use exceptions
for system errors so they can integrate cleanly with host I/O abstractions.

## Memory Model

- Packets are views/proxies over buffers, not value types
- User owns buffer lifetime - library only borrows
- All buffer I/O uses `std::memcpy` to avoid alignment UB
- Use `std::span` with compile-time extent where possible
- Never copy payload data - always reference user buffer directly

## Dual API Design

### Compile-Time Path
- Use when packet structure known at compile time (transmit side)
- Template parameters encode structure: `DataPacket<Type, ClassId, Timestamp>`
- Zero runtime overhead - all offsets computed at compile time
- Strong type safety - compilation fails if accessing unsupported field
- Examples: `DataPacket`, `ContextPacket`

### Runtime Path
- Use when packet structure unknown (receive side)
- Type-erased parsing with automatic validation
- All field access returns `std::optional` for safety
- Read-only (const) - cannot modify received packets
- Examples: `RuntimeDataPacket`, `RuntimeContextPacket`

## Type Hierarchy (C++20 Concepts)

- `PacketBase` - minimal interface all packets support
- `CompileTimePacket` - packets with static structure
  - `FixedPacketLike` - data packets with fixed layout
  - `VariablePacketLike` - context packets with CIF-determined layout
- `RuntimePacket` - runtime packet parsers
  - `RuntimeDataPacket` - data packet parsers
  - `RuntimeContextPacket` - context packet parsers

### Concept Usage
- Enable/disable methods based on packet capabilities
- Self-documenting requirements in function signatures
- SFINAE-friendly for generic programming
- Compile-time validation of packet operations

## Field Access Patterns

### Context Packet Fields (CIF System)
- FieldProxy caches offset, size, presence on creation
- Three-level access hierarchy:
  - `.bytes()` - on-wire bytes as-is
  - `.encoded()` - structured format (Q52.12, etc.)
  - `.value()` - interpreted values (Hz, dBm) - optional
- Lazy evaluation - field not read until proxy dereferenced
- Works for both compile-time and runtime packets

## Minimal API

- Only 4 public headers: types.hpp, timestamp.hpp, class_id.hpp, field_tags.hpp
- Single entry point: vrtigo.hpp (includes all public headers and implementations)
- All implementation in vrtigo/detail/ (never include directly)
- Optional utilities in vrtigo/utils/ (may allocate/throw)
- Concept-constrained to prevent misuse

## Builder Pattern

- Operates directly on user buffer (no temporary)
- Methods return `*this` for chaining
- Terminal `build()` returns packet view over same buffer
- Concept-constrained methods - only available if packet supports feature
- Named after fields: `builder.stream_id(0x1234).timestamp(ts)`

## Error Handling

- Explicit `ValidationError` enum with descriptive codes
- No exceptions - return error codes or optional
- Compile-time packets: caller must validate when parsing untrusted data
- Runtime parsers: automatic validation on construction
- `validation_error_string()` for user-facing messages

## Endian Safety

- All buffer I/O through `vrtigo/detail/buffer_io.hpp`
- Automatic host ↔ network byte order conversion
- Uses `std::memcpy` + compiler intrinsics for efficiency
- Alignment-safe on all architectures
- Single source of truth for buffer access patterns

## Performance

### Zero-Cost Abstractions
- Template specialization eliminates unused code paths
- `if constexpr` for compile-time branching
- Constexpr functions for compile-time computation
- Small packet objects (views are pointer + minimal metadata)

### Memory Access
- Fields stored in network byte order in buffer
- No intermediate copies during serialization
- Linear buffer layout (cache-friendly, no pointer chasing)
- Lazy field access - only read what you use
- Minimize copies, prefer moves

### Compile-Time Optimization
- Field offsets computed at compile time for known structures
- Dead code elimination removes unused features
- Constant folding for CIF offset calculations
- Template instantiation only for used packet types

### Runtime Optimization
- Validation performed once on construction
- Field presence cached in FieldProxy
- Direct buffer manipulation for payload
- Predictable branch patterns for JIT optimization

## Template Techniques

- `if constexpr` for compile-time branching
- `requires` clauses to conditionally enable methods
- Static assertions for early error detection
- SFINAE for generic code compatibility
- Template metaprogramming for CIF bitmask calculation

## Spec Compliance

### Static Validation
- Template parameters validated against VITA 49.2 rules
- Impossible to create packets violating spec constraints
- CIF enable bits (CIF1/2/3) managed automatically
- Field sizes from spec encoded in field tables

### Type System Encoding
- Stream ID presence rules enforced by type
- Trailer requirements validated at compile time
- Packet type constraints in template signatures
- Unsupported fields cause compilation failure, not runtime error

## Design Invariants

- Buffer passed to constructor/builder must outlive packet object
- Packets are shallow - copying packet doesn't copy buffer
- Const packets cannot modify buffer (enforced at type level)
- All public APIs are `noexcept` (except where noted)
- Field access is thread-safe for const packets (read-only)

## Extension Points

- Custom timestamp types via template parameters
- User-defined field interpretations (`.value()` layer)
- Additional packet types through concept conformance
- Custom I/O backends (file reader is example)
