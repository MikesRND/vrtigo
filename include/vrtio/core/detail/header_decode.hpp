#pragma once

#include "../types.hpp"
#include <cstdint>

namespace vrtio::detail {

/**
 * @brief Decoded VRT packet header information
 *
 * Contains all fields extracted from a VRT packet header word.
 * This struct is returned by decode_header() to provide raw bit extraction.
 *
 * IMPORTANT: Bits 24-26 have DIFFERENT meanings for different packet types!
 * Per VITA 49.2 Table 5.1.1.1-1:
 * - Signal Data: bit 26=Trailer, bit 25=Nd0, bit 24=Spectrum/Time
 * - Context:     bit 26=Reserved, bit 25=Nd0, bit 24=TSM
 * - Command:     bit 26=Ack, bit 25=Reserved, bit 24=CanceLation
 *
 * Use packet-type-specific helper functions to interpret these bits correctly.
 */
struct DecodedHeader {
    packet_type type;           ///< Packet type (signal, context, command, etc.)
    uint16_t size_words;        ///< Packet size in 32-bit words

    // Raw packet-specific indicator bits (bits 26-24)
    // Interpretation depends on packet type - see Table 5.1.1.1-1
    bool bit_26;                ///< Trailer(Signal) / Reserved(Context) / Ack(Command)
    bool bit_25;                ///< Nd0(Signal/Context) / Reserved(Command)
    bool bit_24;                ///< Spectrum/Time(Signal) / TSM(Context) / CanceLation(Command)

    // Universal fields
    bool has_class_id;          ///< Class ID field present (bit 27)
    tsi_type tsi;               ///< Integer timestamp type (bits 23-22)
    tsf_type tsf;               ///< Fractional timestamp type (bits 21-20)
    uint8_t packet_count;       ///< Packet count (bits 19-16)

    // DEPRECATED: Old field names for backward compatibility
    // These are populated with packet-type-agnostic interpretations and may be WRONG!
    // Use packet-specific helper functions instead.
    bool has_stream_id;         ///< DEPRECATED - use is_signal_data_with_stream(type) or (type & 1)
    bool has_trailer;           ///< DEPRECATED - only valid for Signal packets, use bit_26 with type check
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
 * - Bits 26-24: Packet-specific indicator bits (interpretation depends on type!)
 * - Bit 23: Reserved
 * - Bits 22-21: TSI type
 * - Bits 20-19: TSF type
 * - Bits 18-16: Reserved
 * - Bits 15-0: Packet size in words
 *
 * IMPORTANT: Bits 26-24 have different meanings for Signal/Context/Command packets.
 * See DecodedHeader documentation and use packet-specific helper functions.
 *
 * @param header The 32-bit header word in network byte order
 * @return DecodedHeader struct with raw bits extracted
 *
 * @note This function does not validate or interpret packet-specific bits.
 *       Use helper functions like has_signal_trailer(), is_nd0_packet(), etc.
 */
inline DecodedHeader decode_header(uint32_t header) noexcept {
    DecodedHeader result;

    // Extract packet type (bits 31-28)
    result.type = static_cast<packet_type>((header >> 28) & 0xF);

    // Extract universal indicator bit (bit 27)
    result.has_class_id = (header >> 27) & 1;

    // Extract raw packet-specific indicator bits (bits 26-24)
    // Interpretation depends on packet type!
    result.bit_26 = (header >> 26) & 1;
    result.bit_25 = (header >> 25) & 1;
    result.bit_24 = (header >> 24) & 1;

    // Extract timestamp types (bits 22-21 and 20-19)
    result.tsi = static_cast<tsi_type>((header >> 22) & 0x3);
    result.tsf = static_cast<tsf_type>((header >> 20) & 0x3);

    // Extract packet count (bits 19-16)
    result.packet_count = (header >> 16) & 0xF;

    // Extract packet size (bits 15-0)
    result.size_words = header & 0xFFFF;

    // DEPRECATED: Populate old fields for backward compatibility
    // These interpretations may be WRONG for Context/Command packets!
    result.has_trailer = result.bit_26;  // Only correct for Signal packets
    result.has_stream_id = result.bit_25;  // WRONG - should be derived from type

    return result;
}

/**
 * @brief Check if a packet type value is valid
 *
 * VITA 49.2 defines packet types 0-7:
 * - 0: UnidentifiedData / Signal Data (no stream ID)
 * - 1: Data / Signal Data (with stream ID)
 * - 2: UnidentifiedExtData / Extension Data (no stream ID)
 * - 3: ExtData / Extension Data (with stream ID)
 * - 4: Context (with stream ID)
 * - 5: ExtContext / Extension Context (with stream ID)
 * - 6: Command (with stream ID)
 * - 7: ExtCommand / Extension Command (with stream ID)
 *
 * Values 8-15 are reserved/invalid.
 *
 * NOTE: Only types 0 and 2 lack stream IDs. All other types (1,3,4,5,6,7) have stream IDs.
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

// ========================================================================
// Packet-Type-Specific Bit Interpretation Helpers
// ========================================================================

/**
 * @brief Determine if packet type has stream ID field
 *
 * Per VITA 49.2 :
 * - Types 0, 2: NO stream ID (UnidentifiedData, UnidentifiedExtData)
 * - Types 1, 3, 4, 5, 6, 7: HAS stream ID (all others)
 * - Types 8-15: Reserved (no stream ID)
 *
 * Stream ID presence is determined by packet type, NOT by any indicator bit!
 *
 * @param type The packet type
 * @return true if packet has stream ID field
 */
inline constexpr bool has_stream_id_field(packet_type type) noexcept {
    uint8_t t = static_cast<uint8_t>(type);
    // Only types 0 and 2 lack stream ID
    return (t != 0) && (t != 2) && (t <= 7);
}

/**
 * @brief Check if packet is a Signal Data packet (types 0-1)
 */
inline constexpr bool is_signal_data_packet(packet_type type) noexcept {
    uint8_t t = static_cast<uint8_t>(type);
    return t == 0 || t == 1;
}

/**
 * @brief Check if packet is an Extension Data packet (types 2-3)
 */
inline constexpr bool is_ext_data_packet(packet_type type) noexcept {
    uint8_t t = static_cast<uint8_t>(type);
    return t == 2 || t == 3;
}

/**
 * @brief Check if packet is a Context packet (types 4-5)
 */
inline constexpr bool is_context_packet(packet_type type) noexcept {
    uint8_t t = static_cast<uint8_t>(type);
    return t == 4 || t == 5;
}

/**
 * @brief Check if packet is a Command packet (types 6-7)
 */
inline constexpr bool is_command_packet(packet_type type) noexcept {
    uint8_t t = static_cast<uint8_t>(type);
    return t == 6 || t == 7;
}

// ========================================================================
// Signal Data Packet Bit Interpretation (types 0-1)
// ========================================================================

/**
 * @brief Check if Signal Data packet has trailer (bit 26)
 *
 * For Signal Data packets, bit 26 indicates trailer presence.
 * ONLY valid for Signal Data packets (types 0-1)!
 *
 * @param decoded Decoded header
 * @return true if trailer is present (only valid for Signal packets)
 */
inline bool has_signal_trailer(const DecodedHeader& decoded) noexcept {
    return is_signal_data_packet(decoded.type) && decoded.bit_26;
}

/**
 * @brief Check if packet is marked as "Not a V49.0 Packet" (Nd0 bit)
 *
 * For Signal Data and Context packets, bit 25 is the Nd0 indicator:
 * - 0: V49.0 compatible mode
 * - 1: Uses V49.2-specific features
 *
 * ONLY valid for Signal Data (0-1) and Context (4-5) packets!
 *
 * @param decoded Decoded header
 * @return true if Nd0 bit is set
 */
inline bool is_nd0_packet(const DecodedHeader& decoded) noexcept {
    return (is_signal_data_packet(decoded.type) || is_context_packet(decoded.type))
           && decoded.bit_25;
}

/**
 * @brief Check if Signal Data packet is Spectrum Data (bit 24)
 *
 * For Signal Data packets, bit 24 indicates:
 * - 0: Signal Time Data Packet
 * - 1: Signal Spectrum Data Packet
 *
 * ONLY valid for Signal Data packets (types 0-1)!
 *
 * @param decoded Decoded header
 * @return true if spectrum data (only valid for Signal packets)
 */
inline bool is_signal_spectrum(const DecodedHeader& decoded) noexcept {
    return is_signal_data_packet(decoded.type) && decoded.bit_24;
}

// ========================================================================
// Context Packet Bit Interpretation (types 4-5)
// ========================================================================

/**
 * @brief Get Timestamp Mode (TSM) for Context packet (bit 24)
 *
 * For Context packets, bit 24 is the Timestamp Mode indicator.
 * ONLY valid for Context packets (types 4-5)!
 *
 * NOTE: Bit 26 is Reserved for Context packets and should be 0.
 *
 * @param decoded Decoded header
 * @return true if TSM bit is set (only valid for Context packets)
 */
inline bool get_context_tsm(const DecodedHeader& decoded) noexcept {
    return is_context_packet(decoded.type) && decoded.bit_24;
}

// ========================================================================
// Command Packet Bit Interpretation (types 6-7)
// ========================================================================

/**
 * @brief Check if Command packet is an Acknowledge packet (bit 26)
 *
 * For Command packets, bit 26 indicates:
 * - 0: Control packet
 * - 1: Acknowledge packet
 *
 * ONLY valid for Command packets (types 6-7)!
 *
 * @param decoded Decoded header
 * @return true if Acknowledge packet (only valid for Command packets)
 */
inline bool is_command_acknowledge(const DecodedHeader& decoded) noexcept {
    return is_command_packet(decoded.type) && decoded.bit_26;
}

/**
 * @brief Check if Command packet is a CanceLation packet (bit 24)
 *
 * For Command packets, bit 24 indicates CanceLation packet.
 * ONLY valid for Command packets (types 6-7)!
 *
 * NOTE: Bit 25 is Reserved for Command packets and should be 0.
 *
 * @param decoded Decoded header
 * @return true if CanceLation packet (only valid for Command packets)
 */
inline bool is_command_cancelation(const DecodedHeader& decoded) noexcept {
    return is_command_packet(decoded.type) && decoded.bit_24;
}

} // namespace vrtio::detail
