#pragma once

#include <cstdint>

#include "../header.hpp"
#include "../types.hpp"

namespace vrtio::detail {

/**
 * @brief Decoded VRT packet header information with type-aware interpretation
 *
 * Contains all fields extracted from a VRT packet header word, including both
 * raw bits (for debugging) and type-aware interpreted fields (for use in code).
 *
 * Usage: After calling decode_header(), check the packet type and use only the relevant fields:
 * - Signal/ExtData packets (types 0-3): use trailer_included, signal_spectrum, nd0
 * - Context packets (types 4-5): use context_tsm, nd0
 * - Command packets (types 6-7): use command_ack, command_cancel
 */
struct DecodedHeader {
    // ========== Universal Fields (valid for all packet types) ==========
    PacketType type;      ///< Packet type (determines which interpreted fields are valid)
    uint16_t size_words;  ///< Packet size in 32-bit words
    bool has_class_id;    ///< Class ID field present (bit 27)
    TsiType tsi;          ///< Integer timestamp type (bits 23-22)
    TsfType tsf;          ///< Fractional timestamp type (bits 21-20)
    uint8_t packet_count; ///< Packet count (bits 19-16)

    // ========== Raw Indicator Bits (for debugging/advanced use) ==========
    bool bit_26; ///< Raw bit 26: Trailer(Signal/ExtData) / Reserved(Context) / Ack(Command)
    bool bit_25; ///< Raw bit 25: Nd0(Signal/ExtData/Context) / Reserved(Command)
    bool bit_24; ///< Raw bit 24: Spectrum(Signal/ExtData) / TSM(Context) / Cancel(Command)

    // ========== Type-Aware Interpreted Fields ==========
    // Only the fields relevant to the packet type are meaningful!
    // Field names align with VITA 49.2 spec terminology

    // Signal/Extension Data packets (types 0-3)
    bool trailer_included; ///< Trailer field present (bit 26) - ONLY valid for Signal/ExtData
                           ///< packets
    bool
        signal_spectrum; ///< Spectrum vs Time data (bit 24) - ONLY valid for Signal/ExtData packets

    // Signal/Extension Data and Context packets (types 0-5)
    bool nd0; ///< Not a V49.0 packet (bit 25) - ONLY valid for Signal/ExtData/Context packets

    // Context packets (types 4-5)
    bool context_tsm; ///< Timestamp Mode (bit 24) - ONLY valid for Context packets

    // Command packets (types 6-7)
    bool command_ack;    ///< Acknowledge vs Control (bit 26) - ONLY valid for Command packets
    bool command_cancel; ///< CanceLation indicator (bit 24) - ONLY valid for Command packets
};

/**
 * @brief Decode a VRT packet header word with type-aware interpretation
 *
 * Extracts all fields from a VRT packet header according to VITA 49.2 and interprets
 * the packet-specific indicator bits based on the packet type.
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
 * Bit Interpretation by Packet Type (per VITA 49.2 Table 5.1.1.1-1):
 * - Signal/ExtData (0-3): bit 26=Trailer, bit 25=Nd0, bit 24=Spectrum
 * - Context (4-5):        bit 26=Reserved, bit 25=Nd0, bit 24=TSM
 * - Command (6-7):        bit 26=Ack, bit 25=Reserved, bit 24=CanceLation
 *
 * @param header The 32-bit header word (in host byte order after network conversion)
 * @return DecodedHeader struct with both raw bits and type-aware interpreted fields
 */
inline DecodedHeader decode_header(uint32_t header) noexcept {
    DecodedHeader result{}; // Zero-initialize all fields

    // ========== Extract Universal Fields ==========
    result.type =
        static_cast<PacketType>((header >> header::packet_type_shift) & header::packet_type_mask);
    result.has_class_id = (header >> header::class_id_shift) & header::class_id_mask;
    result.tsi = static_cast<TsiType>((header >> header::tsi_shift) & header::tsi_mask);
    result.tsf = static_cast<TsfType>((header >> header::tsf_shift) & header::tsf_mask);
    result.packet_count = (header >> header::packet_count_shift) & header::packet_count_mask;
    result.size_words = (header >> header::size_shift) & header::size_mask;

    // ========== Extract Raw Indicator Bits ==========
    result.bit_26 = (header >> header::indicator_bit_26_shift) & header::indicator_bit_mask;
    result.bit_25 = (header >> header::indicator_bit_25_shift) & header::indicator_bit_mask;
    result.bit_24 = (header >> header::indicator_bit_24_shift) & header::indicator_bit_mask;

    // ========== Interpret Indicator Bits Based on Packet Type ==========
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
        // Set all interpreted fields to false
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
 * @brief Check if a packet type value is valid
 *
 * @param type The packet type to validate
 * @return true if type is in range 0-7, false otherwise
 */
inline bool is_valid_packet_type(PacketType type) noexcept {
    uint8_t value = static_cast<uint8_t>(type);
    return value <= 7;
}

/**
 * @brief Check if a TSI type value is valid
 *
 * @param tsi The TSI type to validate
 * @return true if valid (all 2-bit values are valid)
 */
inline constexpr bool is_valid_tsi_type(TsiType tsi) noexcept {
    // All 2-bit values (0-3) are valid TSI types
    return static_cast<uint8_t>(tsi) <= 3;
}

/**
 * @brief Check if a TSF type value is valid
 *
 * @param tsf The TSF type to validate
 * @return true if valid (all 2-bit values are valid)
 */
inline constexpr bool is_valid_tsf_type(TsfType tsf) noexcept {
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
 *
 * @param type The packet type
 * @return true if packet has stream ID field
 */
inline constexpr bool has_stream_id_field(PacketType type) noexcept {
    uint8_t t = static_cast<uint8_t>(type);
    // Only types 0 and 2 lack stream ID
    return (t != 0) && (t != 2) && (t <= 7);
}

/**
 * @brief Check if packet is a Signal Data packet (types 0-1)
 */
inline constexpr bool is_signal_data_packet(PacketType type) noexcept {
    uint8_t t = static_cast<uint8_t>(type);
    return t == 0 || t == 1;
}

/**
 * @brief Check if packet is an Extension Data packet (types 2-3)
 */
inline constexpr bool is_ext_data_packet(PacketType type) noexcept {
    uint8_t t = static_cast<uint8_t>(type);
    return t == 2 || t == 3;
}

/**
 * @brief Check if packet is a Context packet (types 4-5)
 */
inline constexpr bool is_context_packet(PacketType type) noexcept {
    uint8_t t = static_cast<uint8_t>(type);
    return t == 4 || t == 5;
}

/**
 * @brief Check if packet is a Command packet (types 6-7)
 */
inline constexpr bool is_command_packet(PacketType type) noexcept {
    uint8_t t = static_cast<uint8_t>(type);
    return t == 6 || t == 7;
}

} // namespace vrtio::detail
