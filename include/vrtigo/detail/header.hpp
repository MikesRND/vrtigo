#pragma once

#include <cstdint>

namespace vrtigo {

/**
 * @brief VRT packet header bit positions, masks, and manipulation helpers
 *
 * This namespace contains all constants and helper functions for encoding
 * and decoding VRT packet headers according to VITA 49.2.
 *
 * Header format (32 bits):
 * - Bits 31-28: Packet type (4 bits)
 * - Bit 27: Class ID present
 * - Bits 26-24: Packet-specific indicator bits (interpretation depends on type)
 * - Bit 23: Reserved (must be 0)
 * - Bits 23-22: TSI type (2 bits)
 * - Bits 21-20: TSF type (2 bits)
 * - Bits 19-16: Packet Count (4 bits)
 * - Bits 15-0: Packet size in words (16 bits)
 *
 * This is the single source of truth for all header bit manipulation.
 * Used by both header encoding (DataPacket::init_header) and decoding (decode_header).
 */
namespace header {

// ========================================================================
// Packet Type Field (bits 31-28)
// ========================================================================
inline constexpr uint8_t packet_type_shift = 28;
inline constexpr uint32_t packet_type_mask = 0xF; // After shift

// ========================================================================
// Class ID Indicator (bit 27)
// ========================================================================
inline constexpr uint8_t class_id_shift = 27;
inline constexpr uint32_t class_id_mask = 0x1; // After shift

// ========================================================================
// Packet-Specific Indicator Bits (bits 26-24)
// ========================================================================
// These bits have different meanings depending on packet type:
// - Signal/ExtData (0-3): bit 26=Trailer, bit 25=Nd0, bit 24=Spectrum
// - Context (4-5):        bit 26=Reserved, bit 25=Nd0, bit 24=TSM
// - Command (6-7):        bit 26=Ack, bit 25=Reserved, bit 24=CanceLation

inline constexpr uint8_t indicator_bit_26_shift = 26; // Trailer/Ack
inline constexpr uint8_t indicator_bit_25_shift = 25; // Nd0
inline constexpr uint8_t indicator_bit_24_shift = 24; // Spectrum/TSM/Cancel
inline constexpr uint32_t indicator_bit_mask = 0x1;   // After shift (single bit)

// ========================================================================
// TSI Field (bits 23-22) - Integer Timestamp Type
// ========================================================================
inline constexpr uint8_t tsi_shift = 22;
inline constexpr uint32_t tsi_mask = 0x3; // After shift (2 bits)

// ========================================================================
// TSF Field (bits 21-20) - Fractional Timestamp Type
// ========================================================================
inline constexpr uint8_t tsf_shift = 20;
inline constexpr uint32_t tsf_mask = 0x3; // After shift (2 bits)

// ========================================================================
// Packet Count Field (bits 19-16)
// ========================================================================
inline constexpr uint8_t packet_count_shift = 16;
inline constexpr uint32_t packet_count_mask = 0xF; // After shift (4 bits)

// ========================================================================
// Packet Size Field (bits 15-0)
// ========================================================================
inline constexpr uint8_t size_shift = 0;
inline constexpr uint32_t size_mask = 0xFFFF; // After shift (16 bits)

} // namespace header

} // namespace vrtigo
