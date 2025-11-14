#pragma once

#include <span>

#include <cstring>

#include "../../../detail/buffer_io.hpp"
#include "../../../detail/context_packet_view.hpp"
#include "../../../detail/data_packet_view.hpp"
#include "../../../detail/header_decode.hpp"
#include "../../../types.hpp"
#include "../packet_variant.hpp"

namespace vrtio::utils::fileio {
namespace detail_packet_parsing {

/**
 * @brief Parse and validate a VRT packet from raw bytes
 *
 * This function:
 * 1. Validates minimum buffer size (at least 4 bytes for header)
 * 2. Decodes the packet header to determine packet type
 * 3. Creates the appropriate packet view (DataPacketView or ContextPacketView)
 * 4. Returns the validated view or InvalidPacket on error
 *
 * Supported packet types:
 * - Signal Data (0-1) -> DataPacketView
 * - Extension Data (2-3) -> DataPacketView
 * - Context (4-5) -> ContextPacketView
 * - Command (6-7) -> InvalidPacket (not yet implemented)
 *
 * @param bytes Raw packet bytes (must remain valid while using returned view)
 * @return PacketVariant containing validated view or error information
 */
inline PacketVariant parse_and_validate_packet(std::span<const uint8_t> bytes) noexcept {
    // 1. Validate minimum buffer size
    if (bytes.size() < 4) {
        return InvalidPacket{ValidationError::buffer_too_small,
                             PacketType::SignalDataNoId, // Unknown yet
                             vrtio::detail::DecodedHeader{}, bytes};
    }

    // 2. Decode header to determine packet type
    uint32_t header_word = vrtio::detail::read_u32(bytes.data(), 0);
    auto header = vrtio::detail::decode_header(header_word);

    // 3. Dispatch to appropriate view based on packet type
    uint8_t type_value = static_cast<uint8_t>(header.type);

    if (type_value <= 3) {
        // Signal Data (0-1) or Extension Data (2-3)
        vrtio::DataPacketView view(bytes.data(), bytes.size());
        if (view.is_valid()) {
            return view;
        } else {
            return InvalidPacket{view.error(), header.type, header, bytes};
        }
    } else if (type_value == 4 || type_value == 5) {
        // Context (4) or Extension Context (5)
        vrtio::ContextPacketView view(bytes.data(), bytes.size());
        if (view.is_valid()) {
            // Suppress false positive: GCC's optimizer incorrectly thinks padding bytes
            // in ContextPacketView::ParsedStructure might be uninitialized when copied
            // into std::variant, despite structure_{} initialization in constructor.
#if defined(__GNUC__) && !defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
            return view;
#if defined(__GNUC__) && !defined(__clang__)
    #pragma GCC diagnostic pop
#endif
        } else {
            return InvalidPacket{view.error(), header.type, header, bytes};
        }
    } else if (type_value == 6 || type_value == 7) {
        // Command (6) or Extension Command (7) - not yet implemented
        return InvalidPacket{ValidationError::unsupported_field, header.type, header, bytes};
    } else {
        // Invalid/reserved packet type (8-15)
        return InvalidPacket{ValidationError::invalid_packet_type, header.type, header, bytes};
    }
}

} // namespace detail_packet_parsing
} // namespace vrtio::utils::fileio
