#pragma once

#include "../types.hpp"
#include "../header.hpp"
#include <cstdint>

namespace vrtio::detail {

/**
 * Build VRT packet header word from component fields
 *
 * This helper consolidates header initialization logic to avoid duplication
 * between DataPacket and ContextPacket constructors.
 *
 * @param packet_type Packet type (4 bits, 31-28)
 * @param has_class_id Class ID indicator (bit 27)
 * @param bit_26 Indicator bit 26 (trailer for data packets, reserved for context)
 * @param bit_25 Indicator bit 25 (Nd0 for data packets, stream ID for context)
 * @param bit_24 Indicator bit 24 (spectrum/time for data, reserved for context)
 * @param tsi Timestamp integer type (2 bits, 23-22)
 * @param tsf Timestamp fractional type (2 bits, 21-20)
 * @param packet_count Packet counter (4 bits, 19-16)
 * @param packet_size_words Packet size in 32-bit words (16 bits, 15-0)
 * @return Constructed header word in host byte order
 */
inline constexpr uint32_t build_header(
    uint8_t packet_type,
    bool has_class_id,
    bool bit_26,
    bool bit_25,
    bool bit_24,
    tsi_type tsi,
    tsf_type tsf,
    uint8_t packet_count,
    uint16_t packet_size_words
) noexcept {
    uint32_t header = 0;

    // Packet type (bits 31-28)
    header |= (static_cast<uint32_t>(packet_type & 0x0F) << header::PACKET_TYPE_SHIFT);

    // Class ID indicator (bit 27)
    header |= (has_class_id ? 1U : 0U) << header::CLASS_ID_SHIFT;

    // Indicator bits (26, 25, 24)
    header |= (bit_26 ? 1U : 0U) << header::INDICATOR_BIT_26_SHIFT;
    header |= (bit_25 ? 1U : 0U) << header::INDICATOR_BIT_25_SHIFT;
    header |= (bit_24 ? 1U : 0U) << header::INDICATOR_BIT_24_SHIFT;

    // TSI (bits 23-22)
    header |= (static_cast<uint32_t>(tsi) << header::TSI_SHIFT);

    // TSF (bits 21-20)
    header |= (static_cast<uint32_t>(tsf) << header::TSF_SHIFT);

    // Packet count (bits 19-16)
    header |= ((static_cast<uint32_t>(packet_count) & 0x0F) << header::PACKET_COUNT_SHIFT);

    // Packet size in words (bits 15-0)
    header |= ((static_cast<uint32_t>(packet_size_words) & header::SIZE_MASK) << header::SIZE_SHIFT);

    return header;
}

}  // namespace vrtio::detail
