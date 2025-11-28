# VRTIGO Endianness Model

VRTIGO enforces a strict endianness model for VITA 49.2 compliance. All packet buffers use network byte order on the wire, while APIs operate in host byte order. Conversion happens only at buffer boundaries through designated modules.

## Core Invariants

- **Buffers**: Always network byte order (big-endian) per VITA 49.2
- **APIs**: Always host byte order for user convenience
- **Conversion**: Only at buffer boundary, never in business logic
- **Compile-time**: Endianness handling decided via templates, zero runtime cost

## Implementation Layers

| Module | Purpose | Key Functions | Role |
|--------|---------|---------------|------|
| `endian.hpp` | Platform detection & swap primitives | `network_to_host32()`, `host_to_network32()` | **Primitive source** for byte swapping |
| `buffer_io.hpp` | Alignment-safe buffer access | `read_u32()`, `write_u32()` | Calls endian.hpp for conversion + memcpy |
| `bitfield.hpp` | Bit field access | `BitFieldAccessor<Layout, IsBigEndian>` | Calls endian.hpp for field conversion |

**Key principle**: Endian conversion happens only in these three modules. `endian.hpp` provides primitives; `buffer_io.hpp` and `bitfield.hpp` are the only sanctioned call sites.

## Usage Rules

### VITA 49.2 Constraints
- BitField layouts assume 32-bit word spacing: `byte_offset = word_index * 4`
- Buffer must be at least `Layout::required_bytes` in size
- RuntimeBitFieldAccessor uses same `IsBigEndian` parameter for dynamic buffers

### 1. Network-Order Buffers (Default)
```cpp
// Packet/trailer views always wrap network-order buffers
std::span<const std::byte, Layout::required_bytes> buffer = ...;
ConstBitFieldAccessor<Layout>      // IsBigEndian=true (default)
BitFieldAccessor<Layout>            // IsBigEndian=true (default)
RuntimeBitFieldAccessor<Layout>     // IsBigEndian=true (default) for dynamic spans

// These automatically convert network→host on read, host→network on write
```

### 2. Host-Order Value Objects
```cpp
// For types holding host-order uint32_t values (like TrailerBuilder)
uint32_t value = 0;
std::span<std::byte, sizeof(value)> value_span{
    reinterpret_cast<std::byte*>(&value), sizeof(value)
};
BitFieldAccessor<Layout, false> acc(value_span);  // IsBigEndian=false (no double-swap!)

// IMPORTANT: When writing to network buffer, you still need conversion:
// Option A: Use MutableView.set_raw(value) which does host→network
// Option B: Copy to network buffer with host_to_network32() explicitly
```

### 3. Never Do Ad-Hoc Swaps
- No manual shifting/masking for endianness
- No `__builtin_bswap` outside `endian.hpp`
- All swaps go through the three modules above

## Critical Example: Avoiding Double-Swap

### The Problem
TrailerBuilder stores its `value_` member in host order but needs field access:

```cpp
class TrailerBuilder {
    uint32_t value_ = 0;  // HOST order, not network!

    // WRONG - would double-swap on little-endian!
    void set_field_wrong() {
        std::span<std::byte, 4> span{reinterpret_cast<std::byte*>(&value_), 4};
        BitFieldAccessor<Layout, true> acc(span);
        // IsBigEndian=true expects network order, applies unwanted swap
    }

    // CORRECT - no double-swap for field manipulation
    void set_field_correct() {
        std::span<std::byte, 4> span{reinterpret_cast<std::byte*>(&value_), 4};
        BitFieldAccessor<Layout, false> acc(span);
        // IsBigEndian=false knows buffer is host order, no extra swap
    }

    // BRIDGE to network buffer - still needs conversion!
    MutableTrailerView apply(MutableTrailerView view) const noexcept {
        view.set_raw(value_);  // set_raw() does host→network conversion
        return view;
    }
};
```

**Key insight**: Using `IsBigEndian=false` for host-order field access doesn't eliminate the need for host→network conversion when writing to the wire. The conversion just happens at a different layer (e.g., `set_raw()`).

## Quick Reference

| Data Location | Buffer Format | Use `IsBigEndian=` | Example |
|---------------|---------------|-------------------|---------|
| VRT packet buffer | Network order | `true` (default) | `RuntimeDataPacket`, `TrailerView` |
| Wire/file data | Network order | `true` (default) | Any I/O from VITA 49.2 sources |
| uint32_t member | Host order | `false` | `TrailerBuilder::value_` |
| CPU registers | Host order | `false` | Local computation values |

## Type Aliases (Recommended)

```cpp
// Make intent clear with type aliases
template<typename Layout>
using NetworkBufferAccessor = BitFieldAccessor<Layout, true>;   // For packet buffers

template<typename Layout>
using HostValueAccessor = BitFieldAccessor<Layout, false>;      // For value objects
```

## Testing Endianness

Verify correct handling with these patterns:

```cpp
// Test no double-swap between builder and view
TrailerBuilder builder;
builder.set_some_bit(true);
buffer = builder.apply(view);
assert(view.some_bit() == true);  // Must round-trip correctly

// Test on both little and big-endian platforms
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    // Verify swaps applied
#else
    // Verify no swaps (already correct order)
#endif
```

## Migration Checklist

When updating code to use `BitFieldAccessor`:

- [ ] Identify data source format (network or host)
- [ ] Set `IsBigEndian` parameter accordingly
- [ ] Remove any manual endian swaps
- [ ] Test on little-endian platform (x86/ARM)
- [ ] Verify bit positions match expected values