# CIF Field Access Index

This index provides documentation for Context Information Fields (CIF) in VRT packets. CIF fields carry metadata about signal parameters, system state, and packet relationships.

## Access Patterns

Context packets provide field access through the `operator[]` syntax with field tags:

```cpp
packet[field_tag]  // Returns FieldProxy for the field
```

Each field supports a **three-tier access pattern**:

1. **`.bytes()`** - Raw bytes as they appear in the packet (network byte order)
2. **`.encoded()`** - Structured but uninterpreted value (e.g., Q52.12 fixed-point)
3. **`.value()`** - Interpreted value with units (Hz, dBm, etc.) - *only for fields with interpretation support*

Example:
```cpp
// Check if field is present
if (packet[sample_rate]) {
    // Tier 1: Raw bytes
    auto raw = packet[sample_rate].bytes();        // std::span<std::byte, 8>

    // Tier 2: Encoded value
    uint64_t encoded = packet[sample_rate].encoded();  // Q52.12 fixed-point

    // Tier 3: Interpreted value (only for supported fields)
    double hz = packet[sample_rate].value();           // Hertz
}
```

## Packet-Level Accessors

The **Context Field Change Indicator** (CIF0 bit 31) uses packet-level accessor methods instead of the field proxy pattern:

| CIF | Bit | Field | Accessor | Description |
|-----|-----|-------|----------|-------------|
| CIF0 | 31 | [Change Indicator](cif_access/cif0_31_change_indicator.md) | `change_indicator()` / `set_change_indicator(bool)` | Indicates if any context fields changed |

This field is a special case because the bit value itself carries meaning (not field presence).

## Fields with Interpreted Access

The following fields provide `.value()` methods that return interpreted values with units:

| CIF | Bit | Field | Type | Interpretation |
|-----|-----|-------|------|----------------|
| CIF0 | 15 | [Data Payload Format](cif_access/cif0_15_payload_format.md) | `PayloadFormat::View` | Structured bitfield access |
| CIF0 | 17 | [Device Identifier](cif_access/cif0_17_device_id.md) | `DeviceIdentifier::View` | Manufacturer OUI + device code |
| CIF0 | 21 | [Sample Rate](cif_access/cif0_21_sample_rate.md) | `double` | Hz (from Q52.12) |
| CIF0 | 22 | [Over-Range Count](cif_access/cif0_22_over_range_count.md) | `uint32_t` | Count of over-range samples |
| CIF0 | 23 | [Gain](cif_access/cif0_23_gain.md) | `GainValue` | Two-stage gain in dB (from Q9.7) |
| CIF0 | 24 | [Reference Level](cif_access/cif0_24_reference_level.md) | `double` | dBm (from Q9.7) |
| CIF0 | 25 | [IF Band Offset](cif_access/cif0_25_if_band_offset.md) | `double` | Hz (from Q44.20, two's complement) |
| CIF0 | 26 | [RF Frequency Offset](cif_access/cif0_26_rf_frequency_offset.md) | `double` | Hz (from Q44.20, two's complement) |
| CIF0 | 27 | [RF Reference Frequency](cif_access/cif0_27_rf_reference_frequency.md) | `double` | Hz (from Q44.20, two's complement) |
| CIF0 | 28 | [IF Reference Frequency](cif_access/cif0_28_if_reference_frequency.md) | `double` | Hz (from Q44.20, two's complement) |
| CIF0 | 29 | [Bandwidth](cif_access/cif0_29_bandwidth.md) | `double` | Hz (from Q52.12) |
| CIF0 | 30 | [Reference Point Identifier](cif_access/cif0_30_reference_point_id.md) | `uint32_t` | Stream ID of timing reference point |


## Adding New Field Documentation
Field documentation is auto-generated from self-documenting tests in `tests/cif_access/`. To add documentation for a new field:

1. Create a test file: `tests/cif_access/cif{N}_{bit}_{fieldname}_test.cpp`
2. Use `[TITLE]`, `[EXAMPLE]`, `[DESCRIPTION]`, and `[SNIPPET]` markers
3. Run `make autodocs` to generate the documentation


## Fields Without Interpreted Support

Fields without interpreted access only support `.bytes()` and `.encoded()`.
These can be written and read using manual conversions:

```cpp
// Phase Offset: 32-bit word, phase in 16 LSBs
// Format: Radians with radix point right of bit 7
if (packet[field::phase_offset]) {
    // Must use encoded() for fields without interpretation
    uint32_t raw = packet[field::phase_offset].encoded();

    // Manual conversion: Extract 16 LSBs, convert fixed-point
    uint16_t phase_bits = raw & 0xFFFF;
    double radians = phase_bits / 256.0;  // ÷256 for radix at bit 7

    // To set: convert radians to fixed-point
    double new_phase = 3.14159;
    uint32_t encoded = static_cast<uint32_t>(new_phase * 256.0) & 0xFFFF;
    packet[field::phase_offset].set_encoded(encoded);
}
```

## FieldProxy API Reference

Both compile-time (`ContextPacketBuilder<>`) and dynamic (`dynamic::ContextPacketView`) context packets use `operator[]` to access CIF-encoded fields. This returns a `FieldProxy` object providing the three-tier access pattern described above.

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
| `has_value()` | ✓ | ✓ | Explicit presence check |

**Implementation Notes**:
- Variable-length fields include the count word automatically in `bytes()`
- `value()` methods only available if `FieldTraits` specialization defines interpreted conversions
- FieldProxy caches offset, size, and presence on creation for efficiency
- Packet components (stream_id, timestamp) use direct accessors, not `operator[]`


## See Also

- [Packet Accessors Reference](../packet_accessors.md) - Complete API documentation for packet field access
- [Field Tags Reference](../api/field_tags.md) - Complete list of available field tags
- [VITA 49.2 Specification](https://vitacomms.org/) - Official standard reference