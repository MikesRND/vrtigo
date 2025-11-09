#pragma once

#include <cstdint>

namespace vrtio {

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
 * - Bits 22-21: TSI type (2 bits)
 * - Bits 20-19: TSF type (2 bits)
 * - Bits 18-16: Reserved (must be 0)
 * - Bits 15-0: Packet size in words (16 bits)
 *
 * This is the single source of truth for all header bit manipulation.
 * Used by both header encoding (DataPacket::init_header) and decoding (decode_header).
 */
namespace header {

// ========================================================================
// Packet Type Field (bits 31-28)
// ========================================================================
inline constexpr uint8_t PACKET_TYPE_SHIFT = 28;
inline constexpr uint32_t PACKET_TYPE_MASK = 0xF;  // After shift

// ========================================================================
// Class ID Indicator (bit 27)
// ========================================================================
inline constexpr uint8_t CLASS_ID_SHIFT = 27;
inline constexpr uint32_t CLASS_ID_MASK = 0x1;  // After shift

// ========================================================================
// Packet-Specific Indicator Bits (bits 26-24)
// ========================================================================
// These bits have different meanings depending on packet type:
// - Signal/ExtData (0-3): bit 26=Trailer, bit 25=Nd0, bit 24=Spectrum
// - Context (4-5):        bit 26=Reserved, bit 25=Nd0, bit 24=TSM
// - Command (6-7):        bit 26=Ack, bit 25=Reserved, bit 24=CanceLation

inline constexpr uint8_t INDICATOR_BIT_26_SHIFT = 26;  // Trailer/Ack
inline constexpr uint8_t INDICATOR_BIT_25_SHIFT = 25;  // Nd0
inline constexpr uint8_t INDICATOR_BIT_24_SHIFT = 24;  // Spectrum/TSM/Cancel
inline constexpr uint32_t INDICATOR_BIT_MASK = 0x1;    // After shift (single bit)

// ========================================================================
// TSI Field (bits 22-21) - Integer Timestamp Type
// ========================================================================
inline constexpr uint8_t TSI_SHIFT = 22;
inline constexpr uint32_t TSI_MASK = 0x3;  // After shift (2 bits)

// ========================================================================
// TSF Field (bits 20-19) - Fractional Timestamp Type
// ========================================================================
inline constexpr uint8_t TSF_SHIFT = 20;
inline constexpr uint32_t TSF_MASK = 0x3;  // After shift (2 bits)

// ========================================================================
// Packet Count Field (bits 19-16)
// ========================================================================
inline constexpr uint8_t PACKET_COUNT_SHIFT = 16;
inline constexpr uint32_t PACKET_COUNT_MASK = 0xF;  // After shift (4 bits)

// ========================================================================
// Packet Size Field (bits 15-0)
// ========================================================================
inline constexpr uint8_t SIZE_SHIFT = 0;
inline constexpr uint32_t SIZE_MASK = 0xFFFF;  // After shift (16 bits)

}  // namespace header

}  // namespace vrtio
