# PacketSpec Design

Design document for PacketSpec abstractions to simplify the typed packet API.

## Problem Statement

The current typed packet API requires verbose template parameter lists:

```cpp
// Current: verbose, repetitive
using DataPacket = typed::SignalDataPacketBuilder<
    64,                  // PayloadWords (required)
    UtcRealTimestamp,    // TimestampType
    NoClassId,           // ClassIdType (marker type)
    NoTrailer            // TrailerType (marker type)
>;

using ContextPacket = typed::ContextPacketBuilder<
    UtcRealTimestamp,    // TimestampType (same order as data packet)
    NoClassId,           // ClassIdType
    sample_rate,         // Fields...
    bandwidth
>;
```

**Pain Points:**
1. Long template parameter lists (4-5 parameters)
2. Related packets must repeat the same template arguments
3. Packet type (data vs context) not immediately clear

## Requirements

- **Zero-overhead**: Compile-time solution, no runtime cost
- **Packet type first**: Clear distinction between data and context specs
- **Reduce verbosity**: Fewer template parameters, better defaults
- **Share config**: Option to share common settings across packet types
- **Consistent API**: Use marker types for all presence/absence options
- **Maintain type safety**: Invalid configurations fail at compile time

## Proposed Design

### Phase 0: Unify Marker Types ✓ COMPLETED

Trailer and ClassId now both use marker types with consistent naming:

```cpp
namespace vrtigo {

// Marker types for ClassId presence/absence
struct NoClassId {};
struct WithClassId {};

// Marker types for Trailer presence/absence
struct NoTrailer {};
struct WithTrailer {};

// Traits for ClassId
template <typename T> struct ClassIdTraits;
template <> struct ClassIdTraits<NoClassId> { static constexpr size_t size_words = 0; static constexpr bool has_class_id = false; };
template <> struct ClassIdTraits<WithClassId> { static constexpr size_t size_words = 2; static constexpr bool has_class_id = true; };

// Traits for Trailer
template <typename T> struct TrailerTraits;
template <> struct TrailerTraits<NoTrailer> { static constexpr size_t size_words = 0; static constexpr bool has_trailer = false; };
template <> struct TrailerTraits<WithTrailer> { static constexpr size_t size_words = 1; static constexpr bool has_trailer = true; };

// Concepts for type constraints
template <typename T> concept ValidClassIdType = /* ... */;
template <typename T> concept ValidTrailerType = /* ... */;

} // namespace vrtigo
```

`DataPacketBuilder` now uses `typename TrailerType` instead of `Trailer TrailerT`.

### Separate Specs for Data and Context Packets

Packet type is the **primary distinction**, with separate spec types:

```cpp
namespace vrtigo {

// Data packet spec - packet type is first and foremost
template <PacketType Type,
          typename TimestampT = NoTimestamp,
          typename ClassIdT = NoClassId,
          typename TrailerT = NoTrailer>
struct DataPacketSpec {
    static constexpr PacketType packet_type = Type;
    using timestamp_type = TimestampT;
    using class_id_type = ClassIdT;
    using trailer_type = TrailerT;

    // Derive builder with payload size
    template <size_t PayloadWords>
    using Builder = typed::DataPacketBuilder<
        Type, PayloadWords, timestamp_type, class_id_type, trailer_type>;
};

// Context packet spec
template <typename TimestampT = NoTimestamp,
          typename ClassIdT = NoClassId>
struct ContextPacketSpec {
    using timestamp_type = TimestampT;
    using class_id_type = ClassIdT;

    // Derive builder with fields
    template <auto... Fields>
    using Builder = typed::ContextPacketBuilder<
        timestamp_type, class_id_type, Fields...>;
};

} // namespace vrtigo
```

### Convenience Aliases

```cpp
namespace vrtigo {

// Signal data specs (most common)
template <typename TimestampT = NoTimestamp,
          typename ClassIdT = NoClassId,
          typename TrailerT = NoTrailer>
using SignalDataSpec = DataPacketSpec<PacketType::signal_data, TimestampT, ClassIdT, TrailerT>;

template <typename TimestampT = NoTimestamp,
          typename ClassIdT = NoClassId,
          typename TrailerT = NoTrailer>
using SignalDataSpecNoId = DataPacketSpec<PacketType::signal_data_no_id, TimestampT, ClassIdT, TrailerT>;

// Extension data specs
template <typename TimestampT = NoTimestamp,
          typename ClassIdT = NoClassId,
          typename TrailerT = NoTrailer>
using ExtensionDataSpec = DataPacketSpec<PacketType::extension_data, TimestampT, ClassIdT, TrailerT>;

} // namespace vrtigo
```

### Usage Examples

**Basic Data Packet:**
```cpp
// Define spec - packet type is clear
using MyDataSpec = vrtigo::SignalDataSpec<vrtigo::UtcRealTimestamp>;

// Derive builder with payload size
using DataPacket = MyDataSpec::Builder<64>;

// Use as before
alignas(4) std::array<uint8_t, DataPacket::size_bytes()> buffer{};
DataPacket packet(buffer);
packet.set_stream_id(0x1234);
```

**Context Packet:**
```cpp
using MyContextSpec = vrtigo::ContextPacketSpec<vrtigo::UtcRealTimestamp>;

using ContextPacket = MyContextSpec::Builder<sample_rate, bandwidth, gain>;
```

**Full Configuration:**
```cpp
using MyDataSpec = vrtigo::SignalDataSpec<
    vrtigo::UtcRealTimestamp,
    vrtigo::WithClassId,
    vrtigo::WithTrailer
>;

using DataPacket = MyDataSpec::Builder<128>;
```

### Optional: Shared Common Settings

For users who want to share timestamp/class ID between data and context:

```cpp
namespace vrtigo {

// Common settings bundle
template <typename TimestampT = NoTimestamp,
          typename ClassIdT = NoClassId>
struct CommonSpec {
    using timestamp_type = TimestampT;
    using class_id_type = ClassIdT;
};

// Derive packet specs from common settings
template <typename Common, PacketType Type, typename TrailerT = NoTrailer>
using DataPacketSpecFrom = DataPacketSpec<
    Type,
    typename Common::timestamp_type,
    typename Common::class_id_type,
    TrailerT
>;

template <typename Common>
using ContextPacketSpecFrom = ContextPacketSpec<
    typename Common::timestamp_type,
    typename Common::class_id_type
>;

} // namespace vrtigo
```

**Usage with shared settings:**
```cpp
// Define common settings once
using MyCommon = vrtigo::CommonSpec<vrtigo::UtcRealTimestamp, vrtigo::WithClassId>;

// Derive both specs from common
using MyDataSpec = vrtigo::DataPacketSpecFrom<MyCommon, vrtigo::PacketType::signal_data>;
using MyContextSpec = vrtigo::ContextPacketSpecFrom<MyCommon>;

// Create packets
using DataPacket = MyDataSpec::Builder<64>;
using ContextPacket = MyContextSpec::Builder<sample_rate, bandwidth>;
```

## Implementation Plan

### Phase 1: Core Spec Types

1. Create new header: `include/vrtigo/packet_spec.hpp`
2. Implement `DataPacketSpec<Type, Timestamp, ClassId, Trailer>`
3. Implement `ContextPacketSpec<Timestamp, ClassId>`
4. Add convenience aliases (`SignalDataSpec`, `ExtensionDataSpec`, etc.)

### Phase 2: Optional Shared Config

1. Implement `CommonSpec<Timestamp, ClassId>`
2. Add `DataPacketSpecFrom<Common, Type, Trailer>` alias
3. Add `ContextPacketSpecFrom<Common>` alias

### Phase 3: Integration and Testing

1. Include `packet_spec.hpp` in `vrtigo/typed.hpp`
2. Add unit tests demonstrating all usage patterns
3. Update quickstart documentation

## Files to Modify

| File | Change |
|------|--------|
| `include/vrtigo/packet_spec.hpp` | **NEW** - DataPacketSpec, ContextPacketSpec, CommonSpec |
| `include/vrtigo/typed.hpp` | Add `#include "packet_spec.hpp"` |
| `tests/packet_spec_test.cpp` | **NEW** - Unit tests |
| `docs/quickstart/packet_spec.md` | **NEW** - User documentation |

## Design Decisions

### Separate Specs vs Unified Spec

Chose **separate `DataPacketSpec` and `ContextPacketSpec`** because:
- Packet type is immediately clear from the type name
- Data packets have trailer option; context packets don't
- Context packets have CIF fields; data packets have payload size
- Cleaner API that reflects the distinct nature of each packet type

### Why Template Parameters Instead of NTTP Aggregate?

C++20 allows non-type template parameters with class types, enabling:
```cpp
constexpr PacketConfig config{.timestamp = utc, .class_id = true};
template <PacketConfig C> class Packet;
```

However, this adds complexity and the simple template approach achieves the same zero-overhead goal with better compiler support and clearer error messages.

### Why Not Modify Existing Builders?

The existing builders are well-tested and used. PacketSpec is a pure addition:
- No changes to existing builder implementations
- Existing code continues to work
- New code can adopt PacketSpec gradually

### Stream ID Handling

Stream ID is a runtime value, not part of the compile-time spec. Users set it after construction:
```cpp
packet.set_stream_id(0x1234);
```

A future enhancement could add a runtime `StreamSpec` that bundles stream ID with the compile-time spec.
