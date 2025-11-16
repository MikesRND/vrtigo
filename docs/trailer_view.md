# Trailer View API

The TrailerView API provides typed access to VRT packet trailer fields. The trailer is an optional 32-bit word that appears at the end of data packets to convey signal quality and status information.

## Overview

The trailer word contains status indicators and error flags related to signal processing and data validity. The TrailerView classes provide a type-safe interface to these fields following the VITA 49.2 enable/indicator bit pairing pattern.

**Important**: Per VITA 49.2 specification:
- Only **data packets** can have trailers
- **Context packets** never have trailers (bit 26 is reserved and must be 0)
- **Command packets** never have trailers

## Class Hierarchy

```cpp
TrailerView              // Read-only access
  └── MutableTrailerView // Read-write access (inherits from TrailerView)

TrailerBuilder           // Value-type builder for composing trailer words
```

## Accessing Trailer Views

Trailer views are obtained through packet's `trailer()` method when `HasTrailer == Trailer::included`:

| Packet Type | Non-const Method | Const Method |
|-------------|------------------|--------------|
| `DataPacket<..., Trailer::included>` | `trailer()` → `MutableTrailerView` | `trailer() const` → `TrailerView` |
| `DataPacketView` | — | `trailer()` → `optional<uint32_t>` (raw word only) |

## Enable/Indicator Bit Pairing

The VITA 49.2 trailer uses enable/indicator bit pairing for 8 named status fields:
- **Enable bit** (bits 31-24): When set to 1, the corresponding indicator bit contains valid information
- **Indicator bit** (bits 19-12): The actual status value (true/false)

When an enable bit is 0, the corresponding indicator value is **undefined** and should not be used. Therefore, all named indicator getters return `std::optional<bool>`:
- `std::nullopt` when enable bit is 0 (indicator not enabled/invalid)
- `true` or `false` when enable bit is 1 (indicator enabled and valid)

## TrailerView Methods (Read-Only)

### Associated Context Packet Count

| Method | Return Type | Description |
|--------|-------------|-------------|
| `context_packet_count()` | `std::optional<uint8_t>` | Returns count (0-127) if E bit is set, nullopt otherwise. Per Rule 5.1.6-13: When E=1, count is valid. When E=0, count is undefined. |

### Named Indicators (Enable/Indicator Pairs)

All named indicator methods return `std::optional<bool>`:
- `std::nullopt` = enable bit is 0 (indicator not valid)
- `true` = enable bit is 1, indicator bit is 1
- `false` = enable bit is 1, indicator bit is 0

| Method | Enable Bit | Indicator Bit | Description |
|--------|-----------|---------------|-------------|
| `calibrated_time()` | 31 | 19 | Calibrated time indicator |
| `valid_data()` | 30 | 18 | Valid data indicator |
| `reference_lock()` | 29 | 17 | Reference lock indicator |
| `agc_mgc()` | 28 | 16 | AGC/MGC indicator |
| `detected_signal()` | 27 | 15 | Detected signal indicator |
| `spectral_inversion()` | 26 | 14 | Spectral inversion indicator |
| `over_range()` | 25 | 13 | Over-range indicator |
| `sample_loss()` | 24 | 12 | Sample loss indicator |

### Sample Frame and User-Defined Fields

These fields have no enable bits and always return `bool`:

| Method | Bit | Description |
|--------|-----|-------------|
| `sample_frame_1()` | 11 | Sample frame bit 1 |
| `sample_frame_0()` | 10 | Sample frame bit 0 |
| `user_defined_1()` | 9 | User-defined bit 1 |
| `user_defined_0()` | 8 | User-defined bit 0 |

### Utility Methods

| Method | Return Type | Description |
|--------|-------------|-------------|
| `raw()` | `uint32_t` | Entire trailer word value |

## MutableTrailerView Methods (Write Access)

MutableTrailerView inherits all read methods from TrailerView and adds:

### Associated Context Packet Count

| Method | Parameters | Description |
|--------|------------|-------------|
| `set_context_packet_count(uint8_t)` | count (0-127) | Sets E bit to 1 and count to specified value |
| `clear_context_packet_count()` | — | Sets E bit to 0, making count invalid |

### Named Indicator Setters

Each setter **automatically sets both the enable bit and indicator bit**:

| Method | Parameters | Description |
|--------|------------|-------------|
| `set_calibrated_time(bool)` | value | Sets enable bit 31 to 1, indicator bit 19 to value |
| `set_valid_data(bool)` | value | Sets enable bit 30 to 1, indicator bit 18 to value |
| `set_reference_lock(bool)` | value | Sets enable bit 29 to 1, indicator bit 17 to value |
| `set_agc_mgc(bool)` | value | Sets enable bit 28 to 1, indicator bit 16 to value |
| `set_detected_signal(bool)` | value | Sets enable bit 27 to 1, indicator bit 15 to value |
| `set_spectral_inversion(bool)` | value | Sets enable bit 26 to 1, indicator bit 14 to value |
| `set_over_range(bool)` | value | Sets enable bit 25 to 1, indicator bit 13 to value |
| `set_sample_loss(bool)` | value | Sets enable bit 24 to 1, indicator bit 12 to value |

### Named Indicator Clear Methods

Each clear method **sets only the enable bit to 0** (indicator bit is left as-is but becomes invalid):

| Method | Description |
|--------|-------------|
| `clear_calibrated_time()` | Sets enable bit 31 to 0 |
| `clear_valid_data()` | Sets enable bit 30 to 0 |
| `clear_reference_lock()` | Sets enable bit 29 to 0 |
| `clear_agc_mgc()` | Sets enable bit 28 to 0 |
| `clear_detected_signal()` | Sets enable bit 27 to 0 |
| `clear_spectral_inversion()` | Sets enable bit 26 to 0 |
| `clear_over_range()` | Sets enable bit 25 to 0 |
| `clear_sample_loss()` | Sets enable bit 24 to 0 |

### Sample Frame and User-Defined Setters

| Method | Parameters | Description |
|--------|------------|-------------|
| `set_sample_frame_1(bool)` | value | Sets bit 11 |
| `set_sample_frame_0(bool)` | value | Sets bit 10 |
| `set_user_defined_1(bool)` | value | Sets bit 9 |
| `set_user_defined_0(bool)` | value | Sets bit 8 |

### Utility Methods

| Method | Parameters | Description |
|--------|------------|-------------|
| `set_raw(uint32_t)` | value | Set entire trailer word |
| `clear()` | — | Zero entire trailer word (all enable bits and indicator bits set to 0) |

## TrailerBuilder API

TrailerBuilder provides a fluent interface for composing trailer words:

```cpp
// Create a trailer with multiple fields set
auto trailer_cfg = TrailerBuilder{}
    .valid_data(true)              // Sets enable bit 30 and indicator bit 18
    .calibrated_time(true)         // Sets enable bit 31 and indicator bit 19
    .context_packet_count(3);      // Sets E bit 7 and count bits 6-0

// Apply to a packet builder
auto packet = PacketBuilder<PacketType>(buffer.data())
    .stream_id(0x1234)
    .trailer(trailer_cfg)
    .build();
```

### Builder Methods

All setter methods return `TrailerBuilder&` for chaining:

| Method | Parameters | Description |
|--------|------------|-------------|
| `raw(uint32_t)` | value | Set entire word |
| `clear()` | — | Zero entire word |
| `context_packet_count(uint8_t)` | count | Set E bit and count (0-127) |
| `calibrated_time(bool)` | value | Set enable bit 31 and indicator bit 19 |
| `valid_data(bool)` | value | Set enable bit 30 and indicator bit 18 |
| `reference_lock(bool)` | value | Set enable bit 29 and indicator bit 17 |
| `agc_mgc(bool)` | value | Set enable bit 28 and indicator bit 16 |
| `detected_signal(bool)` | value | Set enable bit 27 and indicator bit 15 |
| `spectral_inversion(bool)` | value | Set enable bit 26 and indicator bit 14 |
| `over_range(bool)` | value | Set enable bit 25 and indicator bit 13 |
| `sample_loss(bool)` | value | Set enable bit 24 and indicator bit 12 |
| `sample_frame_1(bool)` | value | Set bit 11 |
| `sample_frame_0(bool)` | value | Set bit 10 |
| `user_defined_1(bool)` | value | Set bit 9 |
| `user_defined_0(bool)` | value | Set bit 8 |
| `from_view(TrailerView)` | view | Copy from existing trailer |
| `value()` | — | Get composed `uint32_t` |

## Usage Examples

### Reading Trailer Fields with Optional Handling

```cpp
// Compile-time packet with trailer
using PacketType = SignalDataPacket<NoClassId, TimeStampUTC, Trailer::included, 128>;
PacketType packet(buffer.data());

// Check named indicators (returns std::optional<bool>)
if (auto valid = packet.trailer().valid_data()) {
    if (*valid) {
        // Data is valid
    } else {
        // Data is explicitly marked invalid
    }
} else {
    // Valid data indicator is not enabled (undefined)
}

// Check for errors
if (auto over = packet.trailer().over_range()) {
    if (*over) {
        // Over-range error detected
    }
}
if (auto loss = packet.trailer().sample_loss()) {
    if (*loss) {
        // Sample loss detected
    }
}

// Check context packet count
if (auto count = packet.trailer().context_packet_count()) {
    std::cout << "Associated with " << (int)*count << " context packets\n";
}

// Sample frame and user-defined (always return bool)
bool frame_bit = packet.trailer().sample_frame_1();
```

### Setting Trailer Fields

```cpp
// Set individual named indicators (automatically sets enable bit)
packet.trailer().set_valid_data(true);        // Sets bits 30 and 18
packet.trailer().set_calibrated_time(true);   // Sets bits 31 and 19
packet.trailer().set_reference_lock(true);    // Sets bits 29 and 17

// Set context packet count (automatically sets E bit)
packet.trailer().set_context_packet_count(5);  // Sets bit 7 and bits 6-0 = 5

// Clear individual indicators (sets only enable bit to 0)
packet.trailer().clear_valid_data();           // Clears bit 30

// Clear entire trailer
packet.trailer().clear();                      // All bits = 0
```

### Using TrailerBuilder with PacketBuilder

```cpp
// Build packet with comprehensive status
using PacketType = SignalDataPacket<NoClassId, TimeStampUTC, Trailer::included, 128>;
std::vector<uint8_t> buffer(PacketType::size_bytes);

// Compose trailer configuration
auto good_status = TrailerBuilder{}
    .valid_data(true)
    .calibrated_time(true)
    .reference_lock(true)
    .context_packet_count(2);

// Build packet
auto ts = TimeStampUTC::from_components(1000000, 0);
auto packet = PacketBuilder<PacketType>(buffer.data())
    .stream_id(0x12345678)
    .timestamp(ts)
    .trailer(good_status)
    .packet_count(0)
    .build();

// Verify
assert(packet.trailer().valid_data() == true);
assert(packet.trailer().context_packet_count() == 2);
```

### Error Condition Reporting

```cpp
// Report error conditions
auto error_trailer = TrailerBuilder{}
    .valid_data(false)    // Data not valid
    .over_range(true)     // Over-range detected
    .sample_loss(true);   // Sample loss detected

auto packet = PacketBuilder<PacketType>(buffer.data())
    .stream_id(0xAABBCCDD)
    .trailer(error_trailer)
    .build();

// Check on receive
if (packet.trailer().over_range().value_or(false) ||
    packet.trailer().sample_loss().value_or(false)) {
    std::cerr << "ERROR: Packet has errors\n";
    if (packet.trailer().over_range().value_or(false)) {
        std::cerr << "  - Over-range detected\n";
    }
    if (packet.trailer().sample_loss().value_or(false)) {
        std::cerr << "  - Sample loss detected\n";
    }
}
```

### Inline Trailer Configuration

```cpp
// For simple cases, use PacketBuilder's trailer methods directly
auto packet = PacketBuilder<PacketType>(buffer.data())
    .stream_id(0x1234)
    .trailer_valid_data(true)
    .trailer_calibrated_time(true)
    .trailer_context_packet_count(1)
    .build();
```

## Bit Field Reference

| Bits | Field | Description |
|------|-------|-------------|
| 31 | Calibrated Time Enable | Enable bit for calibrated time indicator |
| 30 | Valid Data Enable | Enable bit for valid data indicator |
| 29 | Reference Lock Enable | Enable bit for reference lock indicator |
| 28 | AGC/MGC Enable | Enable bit for AGC/MGC indicator |
| 27 | Detected Signal Enable | Enable bit for detected signal indicator |
| 26 | Spectral Inversion Enable | Enable bit for spectral inversion indicator |
| 25 | Over Range Enable | Enable bit for over-range indicator |
| 24 | Sample Loss Enable | Enable bit for sample loss indicator |
| 23-20 | Reserved | Must be zero |
| 19 | Calibrated Time Indicator | Timestamp is calibrated |
| 18 | Valid Data Indicator | Data is valid |
| 17 | Reference Lock Indicator | Reference is locked |
| 16 | AGC/MGC Indicator | AGC or MGC active |
| 15 | Detected Signal Indicator | Signal detected |
| 14 | Spectral Inversion Indicator | Spectral inversion present |
| 13 | Over Range Indicator | Over-range occurred |
| 12 | Sample Loss Indicator | Sample loss occurred |
| 11 | Sample Frame 1 | Sample frame bit 1 |
| 10 | Sample Frame 0 | Sample frame bit 0 |
| 9 | User Defined 1 | User-defined bit 1 |
| 8 | User Defined 0 | User-defined bit 0 |
| 7 | E bit | Associated Context Packet Count Enable |
| 6-0 | Context Packet Count | Associated context packet count (0-127) |

## Implementation Notes

- Trailer word is stored in network byte order (big-endian)
- All bit manipulations handle endianness automatically
- Context packet count is limited to 7 bits (0-127)
- The E bit (bit 7) controls validity of context packet count (separate from indicator enable bits)
- Enable bits (31-24) control validity of 8 named indicators (19-12)
- Sample frame and user-defined fields (11-8) have no enable bits and are always valid
- Trailer views are lightweight (single pointer) and cheap to copy
- Using `value_or(false)` on optional indicators treats disabled indicators as false

## See Also

- [Packet Accessors](packet_accessors.md) - Packet-level API reference
- [Header View API](header_view.md) - Header field access
- VITA 49.2 Section 5.1.6 - Trailer word format specification
- VITA 49.2 Table 5.1.6-1 - State and Event Indicator definitions
- VITA 49.2 Rule 5.1.6-13 - E bit and Associated Context Packet Count
