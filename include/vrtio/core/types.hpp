#pragma once

#include <cstdint>
#include <cstddef>

namespace vrtio {

// VRT packet types (VITA 49.2 standard)
enum class packet_type : uint8_t {
    signal_data_no_stream = 0,    // Signal data without stream identifier
    signal_data_with_stream = 1,   // Signal data with stream identifier
    ext_data_no_stream = 2,        // Extension data without stream identifier
    ext_data_with_stream = 3,      // Extension data with stream identifier
    context = 4,                   // Context packet
    ext_context = 5,               // Extension context packet
    command = 6,                   // Command packet (VITA 49.2)
    ext_command = 7                // Extension command packet (VITA 49.2)
};

// Integer timestamp types (TSI field)
enum class tsi_type : uint8_t {
    none = 0,   // No integer timestamp
    utc = 1,    // UTC time
    gps = 2,    // GPS time
    other = 3   // Other/application-defined
};

// Fractional timestamp types (TSF field)
enum class tsf_type : uint8_t {
    none = 0,           // No fractional timestamp
    sample_count = 1,   // Sample count timestamp
    real_time = 2,      // Real-time picosecond timestamp
    free_running = 3    // Free-running count
};

// TSI resolution constants
inline constexpr uint32_t tsi_resolution_seconds = 1;

// TSF resolution constants (picoseconds)
inline constexpr uint64_t tsf_resolution_picoseconds = 1;
inline constexpr uint64_t picoseconds_per_second = 1'000'000'000'000ULL;

// VRT word size (32 bits)
inline constexpr size_t vrt_word_size = 4;
inline constexpr size_t vrt_word_bits = 32;

// Maximum packet size (16-bit size field, in words)
inline constexpr size_t max_packet_words = 65535;
inline constexpr size_t max_packet_bytes = max_packet_words * vrt_word_size;

// Header bit positions and masks
namespace header {
    // Packet type field (bits 28-31)
    inline constexpr uint8_t PACKET_TYPE_SHIFT = 28;
    inline constexpr uint32_t PACKET_TYPE_MASK = 0xF0000000;

    // Indicator bits
    inline constexpr uint32_t CLASS_ID_INDICATOR = (1U << 27);

    // DEPRECATED: Bit 26 only indicates trailer for Signal Data packets
    // For Context packets, bit 26 is Reserved. For Command packets, it's Ack/Control.
    // Use packet-type-specific helpers in detail/header_decode.hpp instead
    inline constexpr uint32_t TRAILER_INDICATOR = (1U << 26);

    // REMOVED: STREAM_ID_INDICATOR was incorrect
    // Stream ID presence is determined by packet TYPE (bits 31-28), NOT by bit 25!
    // Bit 25 is "Nd0" (Not a V49.0 Packet Indicator) for Signal/Context packets
    // Use detail::has_stream_id_field(packet_type) instead

    // TSI field (bits 22-23)
    inline constexpr uint8_t TSI_SHIFT = 22;
    inline constexpr uint32_t TSI_MASK = 0x00C00000;

    // TSF field (bits 20-21)
    inline constexpr uint8_t TSF_SHIFT = 20;
    inline constexpr uint32_t TSF_MASK = 0x00300000;

    // Packet size field (bits 0-15)
    inline constexpr uint32_t SIZE_MASK = 0x0000FFFF;
}  // namespace header

// Helper: Check if packet type is signal data
constexpr bool is_signal_data(packet_type type) noexcept {
    return type == packet_type::signal_data_no_stream ||
           type == packet_type::signal_data_with_stream;
}

// Helper: Check if packet type includes stream ID
// Per VITA 49.2 and RedhawkSDR reference: types 1, 3, 4, 5, 6, 7 have stream ID
// Only types 0 and 2 (UnidentifiedData, UnidentifiedExtData) lack stream ID
constexpr bool has_stream_identifier(packet_type type) noexcept {
    uint8_t t = static_cast<uint8_t>(type);
    return (t != 0) && (t != 2) && (t <= 7);
}

// Validation error codes for packet parsing
enum class validation_error : uint8_t {
    none = 0,                    // No error, packet is valid
    buffer_too_small,            // Buffer size smaller than declared packet size
    packet_type_mismatch,        // Packet type in header doesn't match template
    tsi_mismatch,                // TSI field doesn't match template
    tsf_mismatch,                // TSF field doesn't match template
    trailer_bit_mismatch,        // Trailer indicator doesn't match template
    size_field_mismatch,         // Size field doesn't match expected packet size
    invalid_packet_type,         // Reserved or unsupported packet type value
    unsupported_field,           // Packet contains fields not supported by this implementation
};

// Convert validation error to human-readable string
constexpr const char* validation_error_string(validation_error err) noexcept {
    switch (err) {
        case validation_error::none:
            return "No error";
        case validation_error::buffer_too_small:
            return "Buffer size smaller than declared packet size";
        case validation_error::packet_type_mismatch:
            return "Packet type doesn't match template configuration";
        case validation_error::tsi_mismatch:
            return "TSI field doesn't match template configuration";
        case validation_error::tsf_mismatch:
            return "TSF field doesn't match template configuration";
        case validation_error::trailer_bit_mismatch:
            return "Trailer indicator doesn't match template configuration";
        case validation_error::size_field_mismatch:
            return "Size field doesn't match expected packet size";
        case validation_error::invalid_packet_type:
            return "Invalid or unsupported packet type";
        case validation_error::unsupported_field:
            return "Packet contains fields not supported by this implementation";
        default:
            return "Unknown error";
    }
}

}  // namespace vrtio
