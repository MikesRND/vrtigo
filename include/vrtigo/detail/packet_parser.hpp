#pragma once

#include <span>

#include <cstring>

#include "../types.hpp"
#include "buffer_io.hpp"
#include "header_decode.hpp"
#include "packet_variant.hpp"

namespace vrtigo::detail {

/**
 * @brief Parse and validate a VRT packet from raw bytes (internal implementation)
 *
 * This function:
 * 1. Validates minimum buffer size (at least 4 bytes for header)
 * 2. Decodes the packet header to determine packet type
 * 3. Creates the appropriate packet view (dynamic::DataPacket or dynamic::ContextPacket)
 * 4. Returns the validated view wrapped in ParseResult, or error information
 *
 * Supported packet types:
 * - Signal Data (0-1) -> dynamic::DataPacket
 * - Extension Data (2-3) -> dynamic::DataPacket
 * - Context (4-5) -> dynamic::ContextPacket
 * - Command (6-7) -> ParseError (not yet implemented)
 *
 * @note This is an internal implementation. Users should use vrtigo::parse_packet()
 * from the public API instead.
 *
 * @param bytes Raw packet bytes (must remain valid while using returned view)
 * @return ParseResult<PacketVariant> containing validated view or error information
 */
[[nodiscard]] inline ParseResult<PacketVariant>
parse_packet_impl(std::span<const uint8_t> bytes) noexcept {
    // 1. Validate minimum buffer size
    if (bytes.size() < 4) {
        return ParseError{ValidationError::buffer_too_small,
                          PacketType::signal_data_no_id, // Unknown yet
                          DecodedHeader{}, bytes};
    }

    // 2. Decode header to determine packet type
    uint32_t header_word = vrtigo::detail::read_u32(bytes.data(), 0);
    auto header = vrtigo::detail::decode_header(header_word);

    // 3. Dispatch to appropriate view based on packet type
    uint8_t type_value = static_cast<uint8_t>(header.type);

    if (type_value <= 3) {
        // Signal Data (0-1) or Extension Data (2-3)
        auto result = dynamic::DataPacket::parse(bytes);
        if (result.ok()) {
            // Suppress false positive: GCC's optimizer incorrectly thinks padding bytes
            // in dynamic::DataPacket::ParsedStructure might be uninitialized when copied
            // into std::variant, despite structure_{} initialization in constructor.
#if defined(__GNUC__) && !defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
            return PacketVariant{std::move(result).value()};
#if defined(__GNUC__) && !defined(__clang__)
    #pragma GCC diagnostic pop
#endif
        } else {
            return result.error();
        }
    } else if (type_value == 4 || type_value == 5) {
        // Context (4) or Extension Context (5)
        auto result = dynamic::ContextPacket::parse(bytes);
        if (result.ok()) {
            // Suppress false positive: GCC's optimizer incorrectly thinks padding bytes
            // in dynamic::ContextPacket::ParsedStructure might be uninitialized when copied
            // into std::variant, despite structure_{} initialization in constructor.
#if defined(__GNUC__) && !defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
            return PacketVariant{std::move(result).value()};
#if defined(__GNUC__) && !defined(__clang__)
    #pragma GCC diagnostic pop
#endif
        } else {
            return result.error();
        }
    } else if (type_value == 6 || type_value == 7) {
        // Command (6) or Extension Command (7) - not yet implemented
        return ParseError{ValidationError::unsupported_field, header.type, header, bytes};
    } else {
        // Invalid/reserved packet type (8-15)
        return ParseError{ValidationError::invalid_packet_type, header.type, header, bytes};
    }
}

} // namespace vrtigo::detail

// ==========
// Public API entry points
// ==========
namespace vrtigo {

/**
 * @brief Parse a data packet from raw bytes
 *
 * Use this when you know the buffer contains a data packet (types 0-3).
 * For unknown packet types, use parse_packet() instead.
 *
 * @param bytes Raw packet bytes (must remain valid while using returned view)
 * @return ParseResult<dynamic::DataPacket> containing the packet or error information
 */
[[nodiscard]] inline ParseResult<dynamic::DataPacket>
parse_data_packet(std::span<const uint8_t> bytes) noexcept {
    return dynamic::DataPacket::parse(bytes);
}

/**
 * @brief Parse a context packet from raw bytes
 *
 * Use this when you know the buffer contains a context packet (types 4-5).
 * For unknown packet types, use parse_packet() instead.
 *
 * @param bytes Raw packet bytes (must remain valid while using returned view)
 * @return ParseResult<dynamic::ContextPacket> containing the packet or error information
 */
[[nodiscard]] inline ParseResult<dynamic::ContextPacket>
parse_context_packet(std::span<const uint8_t> bytes) noexcept {
    return dynamic::ContextPacket::parse(bytes);
}

/**
 * @brief Parse any VRT packet from raw bytes
 *
 * Automatically determines the packet type from the header and creates
 * the appropriate view. Returns ParseResult<PacketVariant> which either
 * contains the valid packet or error information.
 *
 * Supported packet types:
 * - Signal Data (0-1) -> dynamic::DataPacket in variant
 * - Extension Data (2-3) -> dynamic::DataPacket in variant
 * - Context (4-5) -> dynamic::ContextPacket in variant
 * - Command (6-7) -> ParseError (not yet supported)
 *
 * @param bytes Raw packet bytes (must remain valid while using returned view)
 * @return ParseResult<PacketVariant> containing the packet or error information
 */
[[nodiscard]] inline ParseResult<PacketVariant>
parse_packet(std::span<const uint8_t> bytes) noexcept {
    return detail::parse_packet_impl(bytes);
}

} // namespace vrtigo
