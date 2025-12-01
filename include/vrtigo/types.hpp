#pragma once

#include <concepts>

#include <cstddef>
#include <cstdint>

namespace vrtigo {

// VRT packet types (VITA 49.2 standard)
enum class PacketType : uint8_t {
    signal_data_no_id = 0,    // Signal data without stream identifier
    signal_data = 1,          // Signal data with stream identifier
    extension_data_no_id = 2, // Extension data without stream identifier
    extension_data = 3,       // Extension data with stream identifier
    context = 4,              // Context packet
    extension_context = 5,    // Extension context packet
    command = 6,              // Command packet (VITA 49.2)
    extension_command = 7     // Extension command packet (VITA 49.2)
};

// Marker types for Trailer presence/absence
struct NoTrailer {};   ///< Indicates packet has no Trailer field
struct WithTrailer {}; ///< Indicates packet has Trailer field (1 word)

// Trait system for Trailer presence
template <typename T>
struct TrailerTraits;

template <>
struct TrailerTraits<NoTrailer> {
    static constexpr size_t size_words = 0;
    static constexpr bool has_trailer = false;
};

template <>
struct TrailerTraits<WithTrailer> {
    static constexpr size_t size_words = 1;
    static constexpr bool has_trailer = true;
};

// Concept for valid Trailer presence types
template <typename T>
concept ValidTrailerType = requires {
    { TrailerTraits<T>::size_words } -> std::convertible_to<size_t>;
    { TrailerTraits<T>::has_trailer } -> std::convertible_to<bool>;
};

// Concept for types that have a trailer
template <typename T>
concept HasTrailerType = ValidTrailerType<T> && TrailerTraits<T>::has_trailer;

// Integer timestamp types (TSI field)
enum class TsiType : uint8_t {
    none = 0, // No integer timestamp
    utc = 1,  // UTC time
    gps = 2,  // GPS time
    other = 3 // Other/application-defined
};

// Convert TsiType to human-readable string
constexpr const char* tsi_type_string(TsiType type) noexcept {
    switch (type) {
        case TsiType::none:
            return "none";
        case TsiType::utc:
            return "utc";
        case TsiType::gps:
            return "gps";
        case TsiType::other:
            return "other";
        default:
            return "unknown";
    }
}

// Fractional timestamp types (TSF field)
enum class TsfType : uint8_t {
    none = 0,         // No fractional timestamp
    sample_count = 1, // Sample count timestamp
    real_time = 2,    // Real-time picosecond timestamp
    free_running = 3  // Free-running count
};

// Convert TsfType to human-readable string
constexpr const char* tsf_type_string(TsfType type) noexcept {
    switch (type) {
        case TsfType::none:
            return "none";
        case TsfType::sample_count:
            return "sample_count";
        case TsfType::real_time:
            return "real_time";
        case TsfType::free_running:
            return "free_running";
        default:
            return "unknown";
    }
}

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
    return type == PacketType::signal_data_no_id || type == PacketType::signal_data;
}

// Helper: Check if packet type includes stream ID
constexpr bool has_stream_identifier(PacketType type) noexcept {
    uint8_t t = static_cast<uint8_t>(type);
    return (t != 0) && (t != 2) && (t <= 7);
}

// Convert PacketType to human-readable string
constexpr const char* packet_type_string(PacketType type) noexcept {
    switch (type) {
        case PacketType::signal_data_no_id:
            return "signal_data_no_id";
        case PacketType::signal_data:
            return "signal_data";
        case PacketType::extension_data_no_id:
            return "extension_data_no_id";
        case PacketType::extension_data:
            return "extension_data";
        case PacketType::context:
            return "context";
        case PacketType::extension_context:
            return "extension_context";
        case PacketType::command:
            return "command";
        case PacketType::extension_command:
            return "extension_command";
        default:
            return "unknown";
    }
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

} // namespace vrtigo
