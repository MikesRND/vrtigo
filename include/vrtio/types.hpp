#pragma once

#include <cstddef>
#include <cstdint>

namespace vrtio {

// VRT packet types (VITA 49.2 standard)
enum class PacketType : uint8_t {
    SignalDataNoId = 0,    // Signal data without stream identifier
    SignalData = 1,        // Signal data with stream identifier
    ExtensionDataNoId = 2, // Extension data without stream identifier
    ExtensionData = 3,     // Extension data with stream identifier
    Context = 4,           // Context packet
    ExtensionContext = 5,  // Extension context packet
    Command = 6,           // Command packet (VITA 49.2)
    ExtensionCommand = 7   // Extension command packet (VITA 49.2)
};

// Trailer field indicator
enum class Trailer : uint8_t {
    None = 0,    // No trailer field
    Included = 1 // Trailer field present
};

// Integer timestamp types (TSI field)
enum class TsiType : uint8_t {
    none = 0, // No integer timestamp
    utc = 1,  // UTC time
    gps = 2,  // GPS time
    other = 3 // Other/application-defined
};

// Fractional timestamp types (TSF field)
enum class TsfType : uint8_t {
    none = 0,         // No fractional timestamp
    sample_count = 1, // Sample count timestamp
    real_time = 2,    // Real-time picosecond timestamp
    free_running = 3  // Free-running count
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

// Helper: Check if packet type is signal data
constexpr bool is_signal_data(PacketType type) noexcept {
    return type == PacketType::SignalDataNoId || type == PacketType::SignalData;
}

// Helper: Check if packet type includes stream ID
constexpr bool has_stream_identifier(PacketType type) noexcept {
    uint8_t t = static_cast<uint8_t>(type);
    return (t != 0) && (t != 2) && (t <= 7);
}

// Validation error codes for packet parsing
enum class ValidationError : uint8_t {
    none = 0,              // No error, packet is valid
    buffer_too_small,      // Buffer size smaller than declared packet size
    packet_type_mismatch,  // Packet type in header doesn't match template
    class_id_bit_mismatch, // Class ID indicator doesn't match template
    tsi_mismatch,          // TSI field doesn't match template
    tsf_mismatch,          // TSF field doesn't match template
    trailer_bit_mismatch,  // Trailer indicator doesn't match template
    size_field_mismatch,   // Size field doesn't match expected packet size
    invalid_packet_type,   // Reserved or unsupported packet type value
    unsupported_field,     // Packet contains fields not supported by this implementation
};

// Convert validation error to human-readable string
constexpr const char* validation_error_string(ValidationError err) noexcept {
    switch (err) {
        case ValidationError::none:
            return "No error";
        case ValidationError::buffer_too_small:
            return "Buffer size smaller than declared packet size";
        case ValidationError::packet_type_mismatch:
            return "Packet type doesn't match template configuration";
        case ValidationError::class_id_bit_mismatch:
            return "Class ID indicator doesn't match template configuration";
        case ValidationError::tsi_mismatch:
            return "TSI field doesn't match template configuration";
        case ValidationError::tsf_mismatch:
            return "TSF field doesn't match template configuration";
        case ValidationError::trailer_bit_mismatch:
            return "Trailer indicator doesn't match template configuration";
        case ValidationError::size_field_mismatch:
            return "Size field doesn't match expected packet size";
        case ValidationError::invalid_packet_type:
            return "Invalid or unsupported packet type";
        case ValidationError::unsupported_field:
            return "Packet contains fields not supported by this implementation";
        default:
            return "Unknown error";
    }
}

} // namespace vrtio
