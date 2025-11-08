#pragma once

#include "../types.hpp"
#include <cstdint>

namespace vrtio::detail {

/**
 * @brief Decoded VRT packet header information
 *
 * Contains all fields extracted from a VRT packet header word.
 * This struct is returned by decode_header() to provide a single
 * source of truth for header parsing throughout VRTIO.
 */
struct DecodedHeader {
    packet_type type;           ///< Packet type (signal, context, command, etc.)
    uint16_t size_words;        ///< Packet size in 32-bit words
    bool has_stream_id;         ///< Stream ID field present
    bool has_class_id;          ///< Class ID field present
    bool has_trailer;           ///< Trailer field present
    tsi_type tsi;               ///< Integer timestamp type
    tsf_type tsf;               ///< Fractional timestamp type
    uint8_t packet_count;       ///< Packet count (0-15)
};

/**
 * @brief Decode a VRT packet header word
 *
 * Extracts all fields from a VRT packet header according to VITA 49.2.
 * This is the single source of truth for header parsing in VRTIO.
 *
 * Header format (32 bits):
 * - Bits 31-28: Packet type
 * - Bit 27: Class ID present
 * - Bit 26: Trailer present
 * - Bit 25: Stream ID present (always 0 for type 0)
 * - Bits 24-23: Reserved (not extracted)
 * - Bits 22-23: TSI type
 * - Bits 20-21: TSF type
 * - Bits 19-16: Packet count
 * - Bits 15-0: Packet size in words
 *
 * @param header The 32-bit header word in network byte order
 * @return DecodedHeader struct with all extracted fields
 *
 * @note This function does not validate the header. Use is_valid_packet_type()
 *       and other validation functions to check the extracted values.
 */
inline DecodedHeader decode_header(uint32_t header) noexcept {
    DecodedHeader result;

    // Extract packet type (bits 31-28)
    result.type = static_cast<packet_type>((header >> 28) & 0xF);

    // Extract indicator bits (bits 27-25)
    result.has_class_id = (header >> 27) & 1;
    result.has_trailer = (header >> 26) & 1;
    result.has_stream_id = (header >> 25) & 1;

    // Extract timestamp types (bits 23-22 and 21-20)
    result.tsi = static_cast<tsi_type>((header >> 22) & 0x3);
    result.tsf = static_cast<tsf_type>((header >> 20) & 0x3);

    // Extract packet count (bits 19-16)
    result.packet_count = (header >> 16) & 0xF;

    // Extract packet size (bits 15-0)
    result.size_words = header & 0xFFFF;

    return result;
}

/**
 * @brief Check if a packet type value is valid
 *
 * VITA 49.2 defines packet types 0-7:
 * - 0: Signal Data (no stream ID)
 * - 1: Signal Data (with stream ID)
 * - 2: Signal Data Extension (no stream ID)
 * - 3: Signal Data Extension (with stream ID)
 * - 4: Context (no stream ID)
 * - 5: Context (with stream ID)
 * - 6: Command (no stream ID)
 * - 7: Command (with stream ID)
 *
 * Values 8-15 are reserved/invalid.
 *
 * @param type The packet type to validate
 * @return true if type is in range 0-7, false otherwise
 */
inline bool is_valid_packet_type(packet_type type) noexcept {
    uint8_t value = static_cast<uint8_t>(type);
    return value <= 7;
}

/**
 * @brief Check if a TSI type value is valid
 *
 * VITA 49.2 defines TSI types 0-3:
 * - 0: None
 * - 1: UTC
 * - 2: GPS
 * - 3: Other
 *
 * @param tsi The TSI type to validate
 * @return true if valid (all 2-bit values are valid)
 */
inline constexpr bool is_valid_tsi_type(tsi_type tsi) noexcept {
    // All 2-bit values (0-3) are valid TSI types
    return static_cast<uint8_t>(tsi) <= 3;
}

/**
 * @brief Check if a TSF type value is valid
 *
 * VITA 49.2 defines TSF types 0-3:
 * - 0: None
 * - 1: Sample count
 * - 2: Real time (picoseconds)
 * - 3: Free running count
 *
 * @param tsf The TSF type to validate
 * @return true if valid (all 2-bit values are valid)
 */
inline constexpr bool is_valid_tsf_type(tsf_type tsf) noexcept {
    // All 2-bit values (0-3) are valid TSF types
    return static_cast<uint8_t>(tsf) <= 3;
}

} // namespace vrtio::detail
