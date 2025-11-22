# Header View API

The HeaderView API provides typed access to VRT packet header fields. All VRT packets contain a 32-bit header word as their first word, which encodes essential packet metadata.

## Overview

The header word contains packet metadata including type, size, timestamp format indicators, and packet-specific flags. The HeaderView classes provide a type-safe interface to these fields without manual bit manipulation.

## Class Hierarchy

```cpp
HeaderView         // Read-only access (const)
  └── MutableHeaderView  // Read-write access (inherits from HeaderView)
```

## Accessing Header Views

Header views are obtained through packet accessors:

| Packet Type | Non-const Method | Const Method |
|-------------|------------------|--------------|
| `DataPacket` | `header()` → `MutableHeaderView` | `header() const` → `HeaderView` |
| `RuntimeDataPacket` | `header()` → `HeaderView` | — |
| `ContextPacket` | `header()` → `MutableHeaderView` | `header() const` → `HeaderView` |
| `RuntimeContextPacket` | `header()` → `HeaderView` | — |

## HeaderView Methods (Read-Only)

### Core Fields

| Method | Return Type | Description |
|--------|-------------|-------------|
| `packet_type()` | `PacketType` | 4-bit packet type field (bits 31-28) |
| `packet_count()` | `uint8_t` | 4-bit sequence number 0-15 (bits 19-16) |
| `packet_size()` | `uint16_t` | Packet size in 32-bit words (bits 15-0) |

### Indicator Bits

| Method | Return Type | Description |
|--------|-------------|-------------|
| `has_class_id()` | `bool` | Class ID indicator - C bit (bit 27) |
| `has_trailer()` | `bool` | Trailer indicator - T bit (bit 26), data packets only |

### Timestamp Format Indicators

| Method | Return Type | Description |
|--------|-------------|-------------|
| `tsi_kind()` | `TsiType` | Timestamp integer format type (bits 23-22) |
| `tsf_kind()` | `TsfType` | Timestamp fractional format type (bits 21-20) |
| `has_timestamp_integer()` | `bool` | Convenience: TSI ≠ none |
| `has_timestamp_fractional()` | `bool` | Convenience: TSF ≠ none |

**Important**: These methods return metadata ABOUT timestamp formats, not the actual timestamp values. To access timestamp data, use the packet-level `timestamp()` or `timestamp_integer()`/`timestamp_fractional()` methods.

### Packet-Type-Specific Indicators

The header interprets bits 24-26 differently based on packet type. These methods return type-specific structs:

| Method | Return Type | Applicable To | Description |
|--------|-------------|---------------|-------------|
| `data_indicators()` | `DataIndicators` | Data packets | Trailer/Spectrum/Nd0 flags |
| `context_indicators()` | `ContextIndicators` | Context packets | TSM/Nd0 flags |
| `command_indicators()` | `CommandIndicators` | Command packets | Ack/Cancel flags |

## MutableHeaderView Methods (Write Access)

MutableHeaderView inherits all read methods and adds:

| Method | Parameters | Description |
|--------|------------|-------------|
| `set_packet_count(uint8_t)` | 0-15 | Set 4-bit sequence number |
| `set_packet_size(uint16_t)` | words | Set packet size in 32-bit words |

**Note**: Only packet count and size are mutable. Other header fields are determined by packet template parameters and cannot be changed after packet creation.

## Indicator Structures

### DataIndicators

Returned by `header().data_indicators()` for data packets:

```cpp
struct DataIndicators {
    bool has_trailer;      // Bit 26: T bit
    bool has_spectrum;     // Bit 25: Spectrum field indicator
    bool not_v49_0;        // Bit 24: Not a V49.0 packet indicator (Nd0)
};
```

### ContextIndicators

Returned by `header().context_indicators()` for context packets:

```cpp
struct ContextIndicators {
    bool timestamp_mode;   // Bit 25: TSM bit (false=fine, true=coarse)
    bool not_v49_0;       // Bit 24: Not a V49.0 packet indicator (Nd0)
    // Bit 26 is reserved (must be 0) for context packets
};
```

### CommandIndicators

Returned by `header().command_indicators()` for command packets:

```cpp
struct CommandIndicators {
    bool acknowledge;      // Bit 26: ACK bit
    bool cancel;          // Bit 25: Cancel bit
    // Bit 24 reserved
};
```

## Usage Examples

### Reading Header Fields

```cpp
// Compile-time packet
DataPacket<...> packet(buffer.data());
auto hdr = packet.header();
uint8_t count = hdr.packet_count();
bool has_trailer = hdr.data_indicators().has_trailer;

// Runtime view
RuntimeDataPacket view(rx_buffer, size);
if (view.is_valid()) {
    auto hdr = view.header();
    PacketType type = hdr.packet_type();
    uint16_t words = hdr.packet_size();
}
```

### Modifying Header Fields

```cpp
DataPacket<...> packet(buffer.data());
packet.header().set_packet_count(5);
packet.header().set_packet_size(100); // 100 words = 400 bytes
```

### Checking Timestamp Presence

```cpp
// Check timestamp format metadata in header
auto hdr = packet.header();
if (hdr.has_timestamp_integer()) {
    // Packet has integer timestamp component
    TsiType format = hdr.tsi_kind(); // UTC, GPS, etc.
}

// Access actual timestamp values via packet
if constexpr (packet.has_timestamp) {
    auto ts = packet.timestamp(); // Actual timestamp data
}
```

## Implementation Notes

- Header word is stored in network byte order (big-endian)
- All bit manipulations handle endianness automatically
- Packet count wraps modulo 16 (4-bit field)
- Debug builds warn when `set_packet_count()` receives values > 15
- Header views are lightweight (single pointer) and cheap to copy

## See Also

- [Packet Accessors](packet_accessors.md) - Packet-level API reference
- [Trailer View API](trailer_view.md) - Trailer field access
- VITA 49.2 Section 5.1.1 - Header word format specification