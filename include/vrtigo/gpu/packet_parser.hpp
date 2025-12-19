#pragma once

/**
 * @file packet_parser.hpp
 * @brief Device-side VRT packet header parsing for GPU kernels
 *
 * This header provides functions for parsing VRT packet headers on both
 * host and CUDA device. It mirrors the functionality of detail/header_decode.hpp
 * but with __host__ __device__ annotations for GPU compatibility.
 *
 * The DecodedHeader struct is a POD type suitable for device memory.
 */

#include <cstdint>

#include "endian.hpp"

namespace vrtigo {
namespace gpu {

// ============================================================================
// Header field constants (from VITA 49.2)
// ============================================================================

namespace header {

// Packet Type Field (bits 31-28)
inline constexpr uint8_t packet_type_shift = 28;
inline constexpr uint32_t packet_type_mask = 0xF;

// Class ID Indicator (bit 27)
inline constexpr uint8_t class_id_shift = 27;
inline constexpr uint32_t class_id_mask = 0x1;

// Packet-Specific Indicator Bits (bits 26-24)
inline constexpr uint8_t indicator_bit_26_shift = 26;
inline constexpr uint8_t indicator_bit_25_shift = 25;
inline constexpr uint8_t indicator_bit_24_shift = 24;
inline constexpr uint32_t indicator_bit_mask = 0x1;

// TSI Field (bits 23-22)
inline constexpr uint8_t tsi_shift = 22;
inline constexpr uint32_t tsi_mask = 0x3;

// TSF Field (bits 21-20)
inline constexpr uint8_t tsf_shift = 20;
inline constexpr uint32_t tsf_mask = 0x3;

// Packet Count Field (bits 19-16)
inline constexpr uint8_t packet_count_shift = 16;
inline constexpr uint32_t packet_count_mask = 0xF;

// Packet Size Field (bits 15-0)
inline constexpr uint8_t size_shift = 0;
inline constexpr uint32_t size_mask = 0xFFFF;

} // namespace header

// ============================================================================
// Packet type enum (POD-compatible)
// ============================================================================

/**
 * @brief VRT packet types (VITA 49.2 standard)
 *
 * POD enum suitable for device memory.
 */
enum class PacketType : uint8_t {
    signal_data_no_id = 0,
    signal_data = 1,
    extension_data_no_id = 2,
    extension_data = 3,
    context = 4,
    extension_context = 5,
    command = 6,
    extension_command = 7
};

/**
 * @brief Integer timestamp types (TSI field)
 */
enum class TsiType : uint8_t { none = 0, utc = 1, gps = 2, other = 3 };

/**
 * @brief Fractional timestamp types (TSF field)
 */
enum class TsfType : uint8_t { none = 0, sample_count = 1, real_time = 2, free_running = 3 };

// ============================================================================
// DecodedHeader - POD struct for parsed header fields
// ============================================================================

/**
 * @brief Decoded VRT packet header information
 *
 * POD struct containing all fields extracted from a VRT packet header.
 * Suitable for device memory and kernel parameters.
 *
 * Field interpretation depends on packet type:
 * - Signal/ExtData (0-3): trailer_included, signal_spectrum, nd0 are valid
 * - Context (4-5): context_tsm, nd0 are valid
 * - Command (6-7): command_ack, command_cancel are valid
 */
struct DecodedHeader {
    // Universal fields (valid for all packet types)
    PacketType type;
    uint16_t size_words;
    bool has_class_id;
    TsiType tsi;
    TsfType tsf;
    uint8_t packet_count;

    // Raw indicator bits (for debugging/advanced use)
    bool bit_26;
    bool bit_25;
    bool bit_24;

    // Type-specific interpreted fields
    // Signal/Extension Data packets (types 0-3)
    bool trailer_included;
    bool signal_spectrum;

    // Signal/Extension Data and Context packets (types 0-5)
    bool nd0;

    // Context packets (types 4-5)
    bool context_tsm;

    // Command packets (types 6-7)
    bool command_ack;
    bool command_cancel;
};

// ============================================================================
// Header decoding functions
// ============================================================================

/**
 * @brief Decode a VRT packet header word
 *
 * Extracts all fields from a VRT packet header according to VITA 49.2
 * and interprets the packet-specific indicator bits based on packet type.
 *
 * @param header_word The 32-bit header word in HOST byte order
 *                    (caller must convert from network order if needed)
 * @return DecodedHeader struct with all parsed fields
 */
VRTIGO_HOST_DEVICE inline DecodedHeader decode_header(uint32_t header_word) noexcept {
    DecodedHeader result{};

    // Extract universal fields
    result.type = static_cast<PacketType>((header_word >> header::packet_type_shift) &
                                          header::packet_type_mask);
    result.has_class_id = ((header_word >> header::class_id_shift) & header::class_id_mask) != 0;
    result.tsi = static_cast<TsiType>((header_word >> header::tsi_shift) & header::tsi_mask);
    result.tsf = static_cast<TsfType>((header_word >> header::tsf_shift) & header::tsf_mask);
    result.packet_count = (header_word >> header::packet_count_shift) & header::packet_count_mask;
    result.size_words = (header_word >> header::size_shift) & header::size_mask;

    // Extract raw indicator bits
    result.bit_26 =
        ((header_word >> header::indicator_bit_26_shift) & header::indicator_bit_mask) != 0;
    result.bit_25 =
        ((header_word >> header::indicator_bit_25_shift) & header::indicator_bit_mask) != 0;
    result.bit_24 =
        ((header_word >> header::indicator_bit_24_shift) & header::indicator_bit_mask) != 0;

    // Interpret indicator bits based on packet type
    uint8_t type_value = static_cast<uint8_t>(result.type);

    if (type_value <= 3) {
        // Signal Data (0-1) or Extension Data (2-3)
        result.trailer_included = result.bit_26;
        result.signal_spectrum = result.bit_24;
        result.nd0 = result.bit_25;
        result.context_tsm = false;
        result.command_ack = false;
        result.command_cancel = false;
    } else if (type_value == 4 || type_value == 5) {
        // Context (4) or Extension Context (5)
        result.context_tsm = result.bit_24;
        result.nd0 = result.bit_25;
        result.trailer_included = false;
        result.signal_spectrum = false;
        result.command_ack = false;
        result.command_cancel = false;
    } else if (type_value == 6 || type_value == 7) {
        // Command (6) or Extension Command (7)
        result.command_ack = result.bit_26;
        result.command_cancel = result.bit_24;
        result.trailer_included = false;
        result.signal_spectrum = false;
        result.nd0 = false;
        result.context_tsm = false;
    } else {
        // Invalid/reserved packet type (8-15)
        result.trailer_included = false;
        result.signal_spectrum = false;
        result.nd0 = false;
        result.context_tsm = false;
        result.command_ack = false;
        result.command_cancel = false;
    }

    return result;
}

/**
 * @brief Parse a VRT packet from a byte buffer
 *
 * Reads the header word from the buffer, converts from network byte order,
 * and decodes the header fields.
 *
 * @param buffer Pointer to packet data (must have at least 4 bytes)
 * @return DecodedHeader struct with all parsed fields
 */
VRTIGO_HOST_DEVICE inline DecodedHeader parse_packet(const uint8_t* buffer) noexcept {
    // Read header word (first 4 bytes)
    uint32_t header_word;
    // Use byte-by-byte load to avoid alignment issues
    header_word = (static_cast<uint32_t>(buffer[0]) << 24) |
                  (static_cast<uint32_t>(buffer[1]) << 16) |
                  (static_cast<uint32_t>(buffer[2]) << 8) | static_cast<uint32_t>(buffer[3]);

    // Header is already in big-endian (network) byte order in buffer
    // The byte-by-byte load above reconstructs it in host order
    return decode_header(header_word);
}

// ============================================================================
// Packet type helper functions
// ============================================================================

/**
 * @brief Check if packet type is valid (0-7)
 */
VRTIGO_HOST_DEVICE inline bool is_valid_packet_type(PacketType type) noexcept {
    return static_cast<uint8_t>(type) <= 7;
}

/**
 * @brief Check if packet is a Signal Data packet (types 0-1)
 */
VRTIGO_HOST_DEVICE inline bool is_signal_data_packet(PacketType type) noexcept {
    uint8_t t = static_cast<uint8_t>(type);
    return t == 0 || t == 1;
}

/**
 * @brief Check if packet is an Extension Data packet (types 2-3)
 */
VRTIGO_HOST_DEVICE inline bool is_ext_data_packet(PacketType type) noexcept {
    uint8_t t = static_cast<uint8_t>(type);
    return t == 2 || t == 3;
}

/**
 * @brief Check if packet is a Context packet (types 4-5)
 */
VRTIGO_HOST_DEVICE inline bool is_context_packet(PacketType type) noexcept {
    uint8_t t = static_cast<uint8_t>(type);
    return t == 4 || t == 5;
}

/**
 * @brief Check if packet is a Command packet (types 6-7)
 */
VRTIGO_HOST_DEVICE inline bool is_command_packet(PacketType type) noexcept {
    uint8_t t = static_cast<uint8_t>(type);
    return t == 6 || t == 7;
}

/**
 * @brief Check if packet type includes stream ID
 */
VRTIGO_HOST_DEVICE inline bool has_stream_identifier(PacketType type) noexcept {
    uint8_t t = static_cast<uint8_t>(type);
    // Types 0 and 2 have no stream ID, all others (1,3,4,5,6,7) do
    return (t != 0) && (t != 2) && (t <= 7);
}

// ============================================================================
// Payload offset calculation
// ============================================================================

/**
 * @brief Calculate payload offset in words from decoded header
 *
 * Returns the offset (in 32-bit words) from the start of the packet
 * to the payload data.
 *
 * @param hdr Decoded header
 * @return Payload offset in words
 */
VRTIGO_HOST_DEVICE inline uint16_t payload_offset_words(const DecodedHeader& hdr) noexcept {
    uint16_t offset = 1; // Header word

    // Stream ID (1 word if present)
    if (has_stream_identifier(hdr.type)) {
        offset += 1;
    }

    // Class ID (2 words if present)
    if (hdr.has_class_id) {
        offset += 2;
    }

    // Integer timestamp (1 word if TSI != none)
    if (hdr.tsi != TsiType::none) {
        offset += 1;
    }

    // Fractional timestamp (2 words if TSF != none)
    if (hdr.tsf != TsfType::none) {
        offset += 2;
    }

    return offset;
}

/**
 * @brief Calculate payload size in words from decoded header
 *
 * Returns the size of the payload data in 32-bit words.
 *
 * @param hdr Decoded header
 * @return Payload size in words
 */
VRTIGO_HOST_DEVICE inline uint16_t payload_size_words(const DecodedHeader& hdr) noexcept {
    uint16_t header_size = payload_offset_words(hdr);

    // Trailer (1 word if present, only for Signal/ExtData packets)
    uint16_t trailer_size = 0;
    if (hdr.trailer_included) {
        trailer_size = 1;
    }

    // Payload = total size - header - trailer
    if (hdr.size_words >= header_size + trailer_size) {
        return hdr.size_words - header_size - trailer_size;
    }
    return 0; // Invalid packet
}

/**
 * @brief Get pointer to payload data
 *
 * @param packet_buffer Pointer to start of packet
 * @param hdr Decoded header
 * @return Pointer to payload data
 */
VRTIGO_HOST_DEVICE inline const uint8_t* get_payload_ptr(const uint8_t* packet_buffer,
                                                         const DecodedHeader& hdr) noexcept {
    return packet_buffer + (payload_offset_words(hdr) * 4);
}

} // namespace gpu
} // namespace vrtigo
