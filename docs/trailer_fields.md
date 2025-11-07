# Trailer Fields Guide

The VRTIO library provides convenient and type-safe access to VITA 49.2 trailer fields through individual getters and setters.

## Overview

The trailer is an optional 32-bit word at the end of VRT packets that contains status indicators and metadata about the signal and data quality. Each bit or field in the trailer has a specific meaning according to the VITA 49.2 standard.

## Trailer Field Definitions

### Status Indicators (Individual Bits)

| Bit(s) | Field Name | Description |
|--------|------------|-------------|
| 17 | Valid Data | Indicates the data in the packet is valid |
| 16 | Calibrated Time | Indicates the timestamp is calibrated |
| 19 | Signal Detected | Signal detector indicator |
| 18 | Reference Point | Reference point indicator |
| 13 | Sample Loss | Indicates samples were lost |
| 12 | Over-range | Indicates over-range condition occurred |
| 11 | Spectral Inversion | Indicates spectral inversion is present |
| 10 | Detected Signal | Detected signal indicator |
| 9 | AGC/MGC | Automatic/Manual Gain Control indicator |
| 8 | Reference Lock | Reference lock indicator |

### Multi-bit Fields

| Bit(s) | Field Name | Description |
|--------|------------|-------------|
| 0-6 | Associated Context Packets | Count of associated context packets (0-127) |

## Usage Examples

### Basic Trailer Access

#### Reading Trailer Fields

```cpp
#include <vrtio/packet/signal_packet.hpp>

// Define a packet type with trailer enabled
using PacketType = vrtio::signal_packet<
    vrtio::packet_type::signal_data_with_stream,
    vrtio::tsi_type::utc,
    vrtio::tsf_type::none,
    true,  // HasTrailer = true
    1024   // Payload words
>;

// Parse received packet
PacketType rx_packet(received_buffer, false);

// Check individual status bits
if (rx_packet.trailer_valid_data()) {
    std::cout << "Data is valid\n";
}

if (rx_packet.trailer_calibrated_time()) {
    std::cout << "Time is calibrated\n";
}

// Check for error conditions
if (rx_packet.trailer_over_range()) {
    std::cout << "WARNING: Over-range condition detected\n";
}

if (rx_packet.trailer_sample_loss()) {
    std::cout << "WARNING: Sample loss detected\n";
}

// Check if any errors occurred
if (rx_packet.trailer_has_errors()) {
    std::cout << "ERROR: Trailer indicates error conditions\n";
}

// Read multi-bit fields
uint8_t context_count = rx_packet.trailer_context_packets();
std::cout << "Associated context packets: " << (int)context_count << "\n";

// Access raw trailer value if needed
uint32_t raw_trailer = rx_packet.trailer();
std::cout << "Raw trailer: 0x" << std::hex << raw_trailer << "\n";
```

#### Setting Trailer Fields

```cpp
// Create a transmit packet
PacketType tx_packet(tx_buffer);

// Set individual status bits
tx_packet.set_trailer_valid_data(true);
tx_packet.set_trailer_calibrated_time(true);

// Set reference lock status
tx_packet.set_trailer_reference_lock(true);

// Set context packets count
tx_packet.set_trailer_context_packets(5);

// Indicate an error condition
tx_packet.set_trailer_over_range(true);

// Set good status (valid data + calibrated time)
tx_packet.set_trailer_good_status();

// Clear all trailer bits
tx_packet.clear_trailer();
```

### Using the Builder Pattern

The builder pattern provides a fluent API for constructing packets with trailer fields:

```cpp
#include <vrtio/packet/builder.hpp>

// Build a packet with good status
auto packet = vrtio::packet_builder<PacketType>(tx_buffer)
    .stream_id(0x12345678)
    .timestamp_integer(get_current_time())
    .trailer_good_status()  // Sets valid data and calibrated time
    .packet_count(0)
    .build();
```

#### Individual Field Setters in Builder

```cpp
// Build a packet setting individual trailer fields
auto packet = vrtio::packet_builder<PacketType>(tx_buffer)
    .stream_id(0x11111111)
    .timestamp_integer(1000000)
    .trailer_valid_data(true)
    .trailer_calibrated_time(true)
    .trailer_reference_lock(true)
    .trailer_context_packets(3)
    .packet_count(5)
    .build();
```

#### Bulk Status Configuration

```cpp
// Configure multiple status flags at once
auto packet = vrtio::packet_builder<PacketType>(tx_buffer)
    .stream_id(0x22222222)
    .timestamp_integer(2000000)
    .trailer_status(
        true,   // valid_data
        true,   // calibrated_time
        false,  // over_range
        false   // sample_loss
    )
    .packet_count(1)
    .build();
```

#### Indicating Error Conditions

```cpp
// Build a packet indicating error conditions
auto error_packet = vrtio::packet_builder<PacketType>(tx_buffer)
    .stream_id(0x33333333)
    .timestamp_integer(3000000)
    .trailer_valid_data(false)      // Data is not valid
    .trailer_calibrated_time(true)
    .trailer_over_range(true)       // Over-range error
    .trailer_sample_loss(true)      // Sample loss error
    .packet_count(2)
    .build();

// Verify error indication
assert(error_packet.trailer_has_errors());
```

### Advanced Usage

#### Monitoring Signal Quality

```cpp
void process_received_packet(PacketType& packet) {
    // Check data validity
    if (!packet.trailer_valid_data()) {
        log_warning("Invalid data received");
        return;
    }

    // Check timestamp quality
    if (!packet.trailer_calibrated_time()) {
        log_info("Timestamp not calibrated");
    }

    // Check for errors
    if (packet.trailer_over_range()) {
        log_error("ADC over-range detected");
        trigger_gain_adjustment();
    }

    if (packet.trailer_sample_loss()) {
        log_error("Sample loss detected");
        update_packet_loss_counter();
    }

    // Check signal detection
    if (packet.trailer_signal_detected()) {
        log_debug("Signal detected");
    }

    // Process the packet data
    process_payload(packet.payload());
}
```

#### Context Packet Association

```cpp
void send_with_context_packets(uint8_t* data_buffer,
                                const std::vector<uint8_t*>& context_buffers) {
    // Send context packets first
    for (auto* ctx_buffer : context_buffers) {
        send_packet(ctx_buffer);
    }

    // Send data packet with context count
    auto data_packet = vrtio::packet_builder<PacketType>(data_buffer)
        .stream_id(STREAM_ID)
        .timestamp_integer(get_timestamp())
        .trailer_good_status()
        .trailer_context_packets(context_buffers.size())
        .packet_count(get_next_packet_count())
        .build();

    send_packet(data_buffer);
}
```

#### AGC/MGC Monitoring

```cpp
void configure_receiver(PacketType& packet) {
    // Check AGC/MGC status
    if (packet.trailer_agc_mgc()) {
        std::cout << "Using Automatic Gain Control\n";
    } else {
        std::cout << "Using Manual Gain Control\n";
    }

    // Check reference lock
    if (!packet.trailer_reference_lock()) {
        log_warning("Reference not locked - may affect frequency accuracy");
    }
}
```

## Working with Raw Trailer Values

If you need to work with the raw 32-bit trailer value directly, you can still use the lower-level API:

```cpp
// Set raw trailer value
uint32_t custom_trailer = 0x00030001;  // Custom bit pattern
packet.set_trailer(custom_trailer);

// Read raw trailer value
uint32_t trailer_value = packet.trailer();

// Use helper functions for bit manipulation
#include <vrtio/core/trailer.hpp>

// Check individual bits
bool is_valid = vrtio::trailer::is_valid_data(trailer_value);
bool is_calibrated = vrtio::trailer::is_calibrated_time(trailer_value);
bool has_errors = vrtio::trailer::has_errors(trailer_value);

// Extract fields
uint8_t ctx_count = vrtio::trailer::get_context_packets(trailer_value);

// Create status patterns
uint32_t good_status = vrtio::trailer::create_good_status();
packet.set_trailer(good_status);
```

## Compile-Time Safety

All trailer field accessors are protected by C++20 `requires` clauses, ensuring they only exist when the packet type has a trailer enabled:

```cpp
// This compiles - packet has trailer
using PacketWithTrailer = vrtio::signal_packet<
    vrtio::packet_type::signal_data_with_stream,
    vrtio::tsi_type::none,
    vrtio::tsf_type::none,
    true,  // HasTrailer
    128
>;

PacketWithTrailer packet(buffer);
packet.set_trailer_valid_data(true);  // OK

// This does NOT compile - packet has no trailer
using PacketWithoutTrailer = vrtio::signal_packet<
    vrtio::packet_type::signal_data_with_stream,
    vrtio::tsi_type::none,
    vrtio::tsf_type::none,
    false,  // HasTrailer = false
    128
>;

PacketWithoutTrailer packet2(buffer);
// packet2.set_trailer_valid_data(true);  // Compiler error - method doesn't exist
```

## Network Byte Order

All trailer field operations properly handle network byte order (big-endian) conversion automatically. You don't need to worry about byte order when reading or writing trailer fields.

```cpp
// Values are automatically converted to/from network byte order
packet.set_trailer_context_packets(42);
uint8_t count = packet.trailer_context_packets();  // count == 42

// This works correctly regardless of host platform endianness
```

## Performance Considerations

- **Zero Overhead**: All trailer field accessors compile down to simple bit operations
- **No Allocations**: All operations work directly on the packet buffer
- **Compile-Time Optimization**: Template metaprogramming ensures only valid operations are generated
- **Inline Operations**: All helper functions are `constexpr` and can be optimized away

### Read-Modify-Write Pattern

When setting multiple trailer bits, each setter performs a read-modify-write cycle. For maximum efficiency when setting many fields, consider using the raw trailer setter:

```cpp
// Less efficient - multiple read-modify-write cycles
packet.set_trailer_valid_data(true);
packet.set_trailer_calibrated_time(true);
packet.set_trailer_reference_lock(true);

// More efficient - single write
uint32_t trailer = vrtio::trailer::valid_data_mask
                 | vrtio::trailer::calibrated_time_mask
                 | vrtio::trailer::reference_lock_mask;
packet.set_trailer(trailer);

// Or use the builder pattern which is optimized for construction
auto packet = vrtio::packet_builder<PacketType>(buffer)
    .trailer_status(true, true)  // Bulk status setting
    .trailer_reference_lock(true)
    .build();
```

## Common Patterns

### Good Status Indication

```cpp
// Quick way to indicate good status
packet.set_trailer_good_status();

// Equivalent to:
packet.set_trailer_valid_data(true);
packet.set_trailer_calibrated_time(true);
```

### Error Checking

```cpp
// Check for any error conditions
if (packet.trailer_has_errors()) {
    // Handle errors
    if (packet.trailer_over_range()) {
        handle_over_range();
    }
    if (packet.trailer_sample_loss()) {
        handle_sample_loss();
    }
}
```

### Clearing Trailer

```cpp
// Clear all trailer bits
packet.clear_trailer();

// Equivalent to:
packet.set_trailer(0);
```

## VITA 49.2 Standard Compliance

The trailer field implementation follows the VITA 49.2 VRT standard specification. The bit positions and meanings are standardized to ensure interoperability between different VRT implementations.

### Reserved Bits

Some bits in the trailer are reserved by the standard (bits 7, 14-15, 20-31). The library does not provide accessors for reserved bits, but they can be accessed through the raw trailer value if needed for future extensions or vendor-specific uses.

## See Also

- [signal_packet.hpp](../include/vrtio/packet/signal_packet.hpp) - Packet class implementation
- [builder.hpp](../include/vrtio/packet/builder.hpp) - Builder pattern implementation
- [trailer.hpp](../include/vrtio/core/trailer.hpp) - Trailer bit definitions and helpers
- [trailer_test.cpp](../tests/trailer_test.cpp) - Comprehensive test examples
