# Packet Accessor Reference

This document provides a comprehensive reference for accessing VRT packet data. The library provides two parallel APIs:

- **Compile-Time API**: `DataPacket<>`, `ContextPacket<>` - For transmit side where structure is known at compile time
- **Runtime API**: `RuntimeDataPacket`, `RuntimeContextPacket` - For receive side with automatic validation

## Important Terminology

- **Packet Components**: The structural elements of a VRT packet:
  - **Header**: The first 32-bit word containing packet type, counts, and indicator bits
  - **Stream ID**: Optional 32-bit stream identifier (position after header)
  - **Class ID**: Optional 64-bit class identifier
  - **Timestamp**: Optional timestamp (integer and/or fractional parts)
  - **Payload/Context**: Data payload (data packets) or CIF-encoded fields (context packets)
  - **Trailer**: Optional 32-bit trailer word (data packets only)

- **Context Fields**: CIF-encoded fields specific to context packets, accessed via `operator[]` returning `FieldProxy`
  - These are NOT packet components but encoded values within the context section
  - Only available in `ContextPacket` and `RuntimeContextPacket`
  - Examples: bandwidth, reference_level, gain, etc.

## Quick Comparison

| Feature | DataPacket<...> | RuntimeDataPacket | ContextPacket<...> | RuntimeContextPacket |
|---------|----------------|----------------|-------------------|-------------------|
| **Use Case** | Transmit (known structure) | Receive (unknown structure) | Transmit (known structure) | Receive (unknown structure) |
| **Validation** | Manual (call `validate()`) | Automatic on construction | Manual (call `validate()`) | Automatic on construction |
| **Return Types** | Raw values | `std::optional<T>` | Raw values | `std::optional<T>` |
| **Field Access** | Direct getters/setters | Optional getters | `operator[]` → FieldProxy | `operator[]` → FieldProxy |
| **Mutability** | Mutable | Read-only | Mutable | Read-only |
| **Stream ID** | Compile-time optional | Runtime detected | Always present | Always present |
| **Trailer** | Compile-time optional | Runtime detected | Never (spec) | Never (spec) |

---

## Packet-Level Methods

### Size and Validation

| Method/Member | DataPacket | RuntimeDataPacket | ContextPacket | RuntimeContextPacket |
|--------------|-----------|----------------|---------------|-------------------|
| `packet_size_words()` | ✗ (use `size_words`) | ✓ returns `size_t` | ✗ | ✓ returns `size_t` |
| `packet_size_bytes()` | ✗ (use `size_bytes`) | ✓ returns `size_t` | ✗ | ✓ returns `size_t` |
| `buffer_size()` | ✗ | ✓ returns `size_t` | ✗ | ✓ returns `size_t` |
| `size_words` (static) | ✓ `constexpr size_t` | N/A | ✓ `constexpr size_t` | N/A |
| `size_bytes` (static) | ✓ `constexpr size_t` | N/A | ✓ `constexpr size_t` | N/A |
| `validate(size_t)` | ✓ returns `ValidationError` | ✗ (auto-validated) | ✓ returns `ValidationError` | ✗ (auto-validated) |
| `error()` | ✗ | ✓ returns `ValidationError` | ✗ | ✓ returns `ValidationError` |
| `is_valid()` | ✗ | ✓ returns `bool` | ✗ | ✓ returns `bool` |
| `type()` | ✗ (template parameter) | ✓ returns `PacketType` | ✗ (template parameter) | ✓ returns `PacketType` |

**Note**: Compile-time packets (DataPacket, ContextPacket) publish their size through `static constexpr size_words/size_bytes` (rename `total_words` to `size_words`). Runtime views instead expose `packet_size_*()` for the on-wire length and `buffer_size()` for the caller-supplied span length so receivers can check that the buffer actually contains the full packet.

### Packet Count

| Method | DataPacket | RuntimeDataPacket | ContextPacket | RuntimeContextPacket |
|--------|-----------|----------------|---------------|-------------------|
| `packet_count()` | ✓ returns `uint8_t` | ✓ returns `uint8_t` | ✓ returns `uint8_t` | ✓ returns `uint8_t` |
| `set_packet_count(uint8_t)` | ✓ | ✗ | ✓ | ✗ |

**Note**: Packet count is a 4-bit header field (valid range 0-15) available in all packet types per VITA 49.2. Values are wrapped modulo 16 automatically. Debug builds emit warnings when values > 15 are provided to `set_packet_count()`.

---

## Header Word Access

All packet types provide a `header()` accessor that returns a view over the first 32-bit header word:

| Method | DataPacket | RuntimeDataPacket | ContextPacket | RuntimeContextPacket |
|--------|-----------|----------------|---------------|-------------------|
| `header()` | ✓ `MutableHeaderView` | ✓ `HeaderView` | ✓ `MutableHeaderView` | ✓ `HeaderView` |
| `header() const` | ✓ `HeaderView` | — | ✓ `HeaderView` | — |

The returned view provides typed access to all header fields including packet type, size, indicators, and timestamp format metadata.

**See [Header View API](header_view.md) for complete documentation** of HeaderView/MutableHeaderView methods and usage.

## Packet Component Presence Queries

### Compile-Time Packets (DataPacket, ContextPacket)

These use compile-time `static constexpr bool` members determined by template parameters:
- `has_stream_id` - Always `true` for ContextPacket per VITA 49.2 spec
- `has_class_id`
- `has_timestamp` - Combined check for any timestamp components
- `has_timestamp_integer` - TSI component present
- `has_timestamp_fractional` - TSF component present
- `has_trailer` - Always `false` for ContextPacket per spec (bit 26 reserved)

### Runtime Views (RuntimeDataPacket, RuntimeContextPacket)

| Method | RuntimeDataPacket | RuntimeContextPacket | Notes |
|--------|----------------|-------------------|-------|
| `has_stream_id()` | ✓ returns `bool` | ✓ returns `bool` | Type-based for data packets; always `true` for context packets |
| `has_class_id()` | ✓ returns `bool` | ✓ returns `bool` | Header bit 27 (C bit) |
| `has_trailer()` | ✓ returns `bool` | ✓ returns `bool` | Header bit 26 (T bit) for data; always `false` for context |
| `has_timestamp_integer()` | ✓ returns `bool` | ✓ returns `bool` | TSI ≠ none (delegates to `header()`) |
| `has_timestamp_fractional()` | ✓ returns `bool` | ✓ returns `bool` | TSF ≠ none (delegates to `header()`) |
| `tsi_type()` | ✓ returns `TsiType` | ✓ returns `TsiType` | Timestamp integer format metadata |
| `tsf_type()` | ✓ returns `TsfType` | ✓ returns `TsfType` | Timestamp fractional format metadata |

**Delegation Strategy**:
- Header-based queries (`has_class_id`, `has_trailer`, `has_timestamp_*`) delegate to `header()` accessor for single source of truth
- Stream ID presence is packet-type-derived for data packets (not a header bit), always `true` for context packets per spec
- Context packets return `false` for `has_trailer()` per VITA 49.2 (bit 26 reserved for context packet types)

---

## Packet Component Accessors

These methods access the optional packet components that appear between the header and payload/context section.

### Stream ID

| Method | DataPacket | RuntimeDataPacket | ContextPacket | RuntimeContextPacket |
|--------|-----------|----------------|---------------|-------------------|
| `stream_id()` | ✓ `uint32_t` (if `has_stream_id`) | ✓ `optional<uint32_t>` | ✓ `uint32_t` (always) | ✓ `optional<uint32_t>` |
| `set_stream_id(uint32_t)` | ✓ (if `has_stream_id`) | ✗ | ✓ (always) | ✗ |

**Note**: Context packets always include Stream ID per VITA 49.2 spec.

### Class ID

| Method | DataPacket | RuntimeDataPacket | ContextPacket | RuntimeContextPacket |
|--------|-----------|----------------|---------------|-------------------|
| `class_id()` | ✓ `ClassIdValue` (if `has_class_id`) | ✓ `optional<ClassIdValue>` | ✓ `ClassIdValue` (if `has_class_id`) | ✓ `optional<ClassIdValue>` |
| `set_class_id(ClassIdValue)` | ✓ (if `has_class_id`) | ✗ | ✓ (if `has_class_id`) | ✗ |

**Note**: Uses `[[nodiscard]]` attribute to prevent ignoring validation failures.

### Timestamp

**Compile-Time Packets** (`DataPacket`, `ContextPacket`):

| Method | Return Type | Availability |
|--------|-------------|--------------|
| `timestamp()` | `TimeStampType` | When `HasTimestamp<TimeStampType>` |
| `set_timestamp(TimeStampType)` | void | When `HasTimestamp<TimeStampType>` |

**Runtime Views**:

| Method | RuntimeDataPacket | RuntimeContextPacket |
|--------|----------------|-------------------|
| `timestamp_integer()` | ✓ `optional<uint32_t>` | ✓ `optional<uint32_t>` |
| `timestamp_fractional()` | ✓ `optional<uint64_t>` | ✓ `optional<uint64_t>` |

**Note**: Timestamp data is distinct from timestamp format metadata. Use `tsi_type()` and `tsf_type()` (or `header().tsi_type()` / `header().tsf_type()`) to query the timestamp format, and `timestamp_integer()` / `timestamp_fractional()` to access the actual timestamp values.

### Trailer

| Method | DataPacket | RuntimeDataPacket | ContextPacket | RuntimeContextPacket |
|--------|-----------|----------------|---------------|-------------------|
| `trailer()` | ✓ `MutableTrailerView` (if `HasTrailer`) | ✓ `optional<uint32_t>` (raw word) | ✗ (spec forbids) | ✗ (spec forbids) |
| `trailer() const` | ✓ `TrailerView` (if `HasTrailer`) | — | ✗ | ✗ |

**Note**: Context packets never include trailers per VITA 49.2 specification (bit 26 is reserved and must be zero).

For trailer bit manipulation, see [TrailerView API](#trailerview-api-data-packets) below.

---

## Buffer Access

All packet types provide access to the complete packet buffer as a byte span.

| Method/Query | Compile-Time (DataPacket, ContextPacket) | Runtime (RuntimeDataPacket, RuntimeContextPacket) |
|--------------|------------------------------------------|---------------------------------------------|
| `as_bytes()` | ✓ `span<uint8_t, N>` | ✓ `span<const uint8_t>` |
| `as_bytes() const` | ✓ `span<const uint8_t, N>` | — |
| Total packet size | `size_bytes` / `size_words` (static constexpr members) | `packet_size_bytes()` / `packet_size_words()` (methods) |
| Caller-supplied buffer size | ✗ | `buffer_size()` |

**Const Overloading**: Compile-time packets provide both mutable and const overloads for `as_bytes()`. Runtime views are read-only parsers and only provide const access.

**Naming Convention**: Compile-time packets expose sizes as `static constexpr` members (e.g., `size_bytes`, `size_words`) for use in template logic and static assertions. Runtime views provide methods (e.g., `packet_size_bytes()`, `packet_size_words()`) that read decoded header values. This distinction reflects that compile-time sizes are constants while runtime sizes are determined by parsing.

**Note**:
- Compile-time packets use fixed-size spans with compile-time extent (`span<T, N>`)
- Runtime views use dynamic-size spans (`span<T>`) and return empty spans if invalid

---

## Payload Access (Data Packets Only)

Data packets have a traditional data payload section. Context packets use CIF-based field access instead (see next section).

| Method/Query | DataPacket (Compile-Time) | RuntimeDataPacket (Runtime) |
|--------------|---------------------------|--------------------------|
| `payload()` | ✓ `span<uint8_t, N>` | ✓ `span<const uint8_t>` |
| `payload() const` | ✓ `span<const uint8_t, N>` | — |
| `set_payload(const uint8_t*, size_t)` | ✓ | ✗ |
| Payload size in bytes | `payload_size_bytes` (static constexpr member) | `payload_size_bytes()` (method) |
| Payload size in words | `payload_words` (static constexpr member) | `payload_size_words()` (method) |

**Const Overloading**:
- `DataPacket::payload()` (non-const) returns mutable `span<uint8_t>` for writing payload data
- `DataPacket::payload() const` returns `span<const uint8_t>` for read-only access in const contexts
- `RuntimeDataPacket::payload()` is always const (read-only parser)

**Naming Pattern**: Same as buffer size queries - static members for compile-time constants, methods for runtime decoded values.

---

## CIF Field Access (Context Packets Only)

Context packets don't have a traditional "payload" - instead they encode CIF-based context fields.

| Method | ContextPacket | RuntimeContextPacket |
|--------|---------------|-------------------|
| `operator[](field_tag)` | ✓ mutable `FieldProxy` | ✓ const `FieldProxy` |

**Usage**: Access context fields via field tags from `vrtigo::field` namespace (e.g., `packet[field::bandwidth]`).

**Internal helpers** (do not call directly; for FieldProxy use only):
- `context_buffer()` / `mutable_context_buffer()` - Raw buffer pointers to CIF field area
- `context_base_offset()` - Byte offset to start of CIF field area

---

## CIF Metadata (Context Packets Only)

### Compile-Time (ContextPacket)

Static constexpr members publishing CIF masks for builders and tests:

| Member | Type | Description |
|--------|------|-------------|
| `cif0_value` | `uint32_t` | CIF0 with automatic enable bits |
| `cif1_value` | `uint32_t` | CIF1 mask |
| `cif2_value` | `uint32_t` | CIF2 mask |
| `cif3_value` | `uint32_t` | CIF3 mask |

Also available via methods:
- `static constexpr uint32_t cif0()`
- `static constexpr uint32_t cif1()`
- `static constexpr uint32_t cif2()`
- `static constexpr uint32_t cif3()`

### Runtime (RuntimeContextPacket)

| Method | Return Type | Description |
|--------|-------------|-------------|
| `cif0()` | `uint32_t` | Parsed CIF0 word (diagnostic) |
| `cif1()` | `uint32_t` | Parsed CIF1 word (diagnostic) |
| `cif2()` | `uint32_t` | Parsed CIF2 word (diagnostic) |
| `cif3()` | `uint32_t` | Parsed CIF3 word (diagnostic) |

---

## Context Field Access (Context Packets Only)

Both compile-time (`ContextPacket<>`) and runtime (`RuntimeContextPacket`) context packets use `operator[]` to access CIF-encoded fields within the context section of the packet (NOT packet components like stream_id or timestamp). This returns a `FieldProxy` object providing a three-tier access pattern:

```cpp
auto proxy = packet[field::bandwidth];
```

### FieldProxy Methods

| Method | Compile-Time (mutable) | Runtime (const) | Description |
|--------|----------------------|-----------------|-------------|
| `bytes()` | ✓ read | ✓ read | Literal on-wire bytes |
| `set_bytes(span)` | ✓ write | ✗ | Set on-wire bytes in bulk |
| `encoded()` | ✓ read | ✓ read | Structured format (e.g., Q52.12 as `uint64_t`) |
| `set_encoded(T)` | ✓ write | ✗ | Set structured value |
| `value()` | ✓ read | ✓ read | Interpreted units (Hz, dBm, etc.) if defined |
| `set_value(T)` | ✓ write | ✗ | Set interpreted value |
| `operator bool()` | ✓ | ✓ | Check field presence |

**Notes**:
- Variable-length fields include the count word automatically in `bytes()`
- `value()` methods only available if `FieldTraits` specialization defines interpreted conversions
- FieldProxy caches offset, size, and presence on creation for efficiency

**Example Usage**:
```cpp
// Access Context Fields (not packet components like stream_id or timestamp)
// Context Fields are CIF-encoded fields within the context section

// Presence check
if (packet[field::bandwidth]) {  // bandwidth is a Context Field
    // Read interpreted value (Hz)
    double bw = packet[field::bandwidth].value();

    // Read encoded structured value (Q52.12 as uint64_t)
    uint64_t enc = packet[field::bandwidth].encoded();

    // Read raw bytes
    auto bytes = packet[field::bandwidth].bytes();
}

// Set value (compile-time packets only)
packet[field::bandwidth].set_value(20e6); // 20 MHz

// Note: Packet components like stream_id use direct accessors, not operator[]
// packet.stream_id();  // Correct for packet component accessor
// packet[field::stream_id];  // Wrong - stream_id is not a Context Field
```

---

## Trailer Access (Data Packets Only)

Data packets can optionally include a trailer word for signal quality and status information. When `HasTrailer == Trailer::included`, the `trailer()` method returns a view object:

| Method | DataPacket (mutable) | DataPacket (const) | RuntimeDataPacket |
|--------|---------------------|-------------------|----------------|
| `trailer()` | ✓ `MutableTrailerView` | ✓ `TrailerView` | ✓ `optional<uint32_t>` |

The returned view provides typed access to all trailer status and error fields.

**See [Trailer View API](trailer_view.md) for complete documentation** of TrailerView/MutableTrailerView methods, TrailerBuilder, and usage examples.

---

## Return Types and Attributes

### Compile-Time Packets
- Return plain values: `uint32_t`, `ClassIdValue`, `span<uint8_t, N>`
- No optionals - template enforces field presence at compile time
- Methods use `requires` clauses to conditionally compile based on template parameters

### Runtime Views
- Return `std::optional<T>` for any field that could be missing or invalid
- Even "mandatory" fields use optionals to prevent accessing unvalidated memory
- Many methods marked `[[nodiscard]]` to discourage ignoring validation failures

---

## Implementation Reference

For implementation details and exact signatures, see:

### Data Packets (Compile-Time)
- **File**: `include/vrtigo/detail/data_packet.hpp`
- **Class**: `DataPacket<PacketType Type, ClassIdType ClassId, TimeStampType TimeStamp, Trailer HasTrailer, size_t PayloadWords>`
- **Key sections**:
  - Header accessors (packet_count) and static size constants (`size_words`, `size_bytes`)
  - Stream ID (conditional on has_stream_id)
  - Class ID (conditional on has_class_id)
  - Timestamp (conditional on HasTimestamp)
  - Trailer view (conditional on HasTrailer)
  - Payload access (fixed-size spans)
  - Validation method

### Data Packets (Runtime)
- **File**: `include/vrtigo/detail/runtime_data_packet.hpp`
- **Class**: `RuntimeDataPacket`
- **Key sections**:
  - Automatic validation on construction
  - Validation query methods (error, is_valid)
  - Header accessor (header())
  - Type and presence queries (type, has_*, tsi_type, tsf_type)
  - Optional field accessors (stream_id, class_id, timestamp_*, trailer)
  - Payload and size queries (dynamic spans)

### Context Packets (Compile-Time)
- **File**: `include/vrtigo/detail/context_packet.hpp`
- **Class**: `ContextPacketBase<...>` (aliased as `ContextPacket<...>`)
- **Key sections**:
  - Stream ID (always present)
  - Class ID (conditional on has_class_id)
  - Timestamp (conditional on has_timestamp)
  - FieldProxy access via operator[]
  - CIF static constexpr members
  - Validation method

### Context Packets (Runtime)
- **File**: `include/vrtigo/detail/runtime_context_packet.hpp`
- **Class**: `RuntimeContextPacket`
- **Key sections**:
  - Automatic validation on construction
  - Validation query methods
  - Header accessor (header())
  - Timestamp format metadata accessors (tsi_type, tsf_type, has_timestamp_*)
  - Optional accessors (stream_id, class_id, timestamp_integer, timestamp_fractional)
  - CIF query methods (cif0-3)
  - FieldProxy access via operator[]
  - Size queries

### Header Views
- **File**: `include/vrtigo/detail/packet_header_accessor.hpp`
- **Classes**: `HeaderView`, `MutableHeaderView`
- **Documentation**: [Header View API](header_view.md)

### Trailer Views
- **File**: `include/vrtigo/detail/trailer_view.hpp`
- **Classes**: `TrailerView`, `MutableTrailerView`
- **Documentation**: [Trailer View API](trailer_view.md)

### Field Proxy
- **File**: `include/vrtigo/detail/field_proxy.hpp`
- **Template**: `FieldProxy<FieldTag, PacketType>`
- **Methods**: bytes, encoded, value (with corresponding setters for mutable packets)

---

## Notes

- `ContextPacket<...>` is a direct alias over `ContextPacketBase`; all accessors are inherited unchanged
- Runtime views default to `std::optional` to guard against malformed packets
- Context packet CIF access is exclusively routed through field tags and `FieldProxy`
- Direct buffer pointers in context packets are marked as implementation details and should remain internal
- The `[[nodiscard]]` attribute mirrors guidance in `docs/style.md` to prevent ignoring validation problems
