#pragma once

#include <span>
#include <string>
#include <variant>

#include "../dynamic/context_packet.hpp"
#include "../dynamic/data_packet.hpp"
#include "../types.hpp"
#include "header_decode.hpp"
#include "parse_result.hpp"

namespace vrtigo {

/**
 * @brief Type-safe variant holding validated packet views
 *
 * PacketVariant holds only valid, successfully parsed packets:
 * - dynamic::DataPacketView: Signal or Extension data packets (types 0-3)
 * - dynamic::ContextPacketView: Context or Extension Context packets (types 4-5)
 *
 * To parse a packet, use parse_packet() which returns ParseResult<PacketVariant>.
 * If parsing fails, the error information is in the ParseError, not in the variant.
 */
using PacketVariant = std::variant<dynamic::DataPacketView,   // Signal/Extension data packets
                                   dynamic::ContextPacketView // Context/Extension context packets
                                   >;

// ==========
// Helpers for PacketVariant (always valid)
// ==========

/**
 * @brief Get the packet type from a packet variant
 * @param pkt The packet variant (always valid)
 * @return The packet type
 */
inline PacketType packet_type(const PacketVariant& pkt) noexcept {
    return std::visit(
        [](auto&& p) -> PacketType {
            using T = std::decay_t<decltype(p)>;

            if constexpr (std::is_same_v<T, dynamic::DataPacketView>) {
                return p.type();
            } else if constexpr (std::is_same_v<T, dynamic::ContextPacketView>) {
                return p.type();
            }

            // Should never reach here
            return PacketType::signal_data_no_id;
        },
        pkt);
}

/**
 * @brief Get the stream ID from a packet variant, if present
 * @param pkt The packet variant
 * @return The stream ID if the packet has one, std::nullopt otherwise
 */
inline std::optional<uint32_t> stream_id(const PacketVariant& pkt) noexcept {
    return std::visit(
        [](auto&& p) -> std::optional<uint32_t> {
            using T = std::decay_t<decltype(p)>;

            if constexpr (std::is_same_v<T, dynamic::DataPacketView>) {
                return p.stream_id();
            } else if constexpr (std::is_same_v<T, dynamic::ContextPacketView>) {
                return p.stream_id();
            }

            return std::nullopt;
        },
        pkt);
}

/**
 * @brief Check if a packet variant holds a data packet
 * @param pkt The packet variant to check
 * @return true if the packet is a dynamic::DataPacketView, false otherwise
 */
inline bool is_data_packet(const PacketVariant& pkt) noexcept {
    return std::holds_alternative<dynamic::DataPacketView>(pkt);
}

/**
 * @brief Check if a packet variant holds a context packet
 * @param pkt The packet variant to check
 * @return true if the packet is a dynamic::ContextPacketView, false otherwise
 */
inline bool is_context_packet(const PacketVariant& pkt) noexcept {
    return std::holds_alternative<dynamic::ContextPacketView>(pkt);
}

// ==========
// Helpers for ParseResult<PacketVariant>
// ==========

/**
 * @brief Check if a parse result holds a valid packet
 * @param result The parse result to check
 * @return true if parsing succeeded, false otherwise
 */
inline bool is_valid(const ParseResult<PacketVariant>& result) noexcept {
    return result.has_value();
}

/**
 * @brief Get the packet type from a parse result
 *
 * On success, returns the packet type from the variant.
 * On error, returns the attempted_type from the ParseError.
 *
 * @param result The parse result
 * @return The packet type (actual or attempted)
 */
inline PacketType packet_type(const ParseResult<PacketVariant>& result) noexcept {
    if (result.has_value()) {
        return packet_type(*result);
    }
    return result.error().attempted_type;
}

/**
 * @brief Get the stream ID from a parse result, if present
 * @param result The parse result
 * @return The stream ID if parsing succeeded and packet has one, std::nullopt otherwise
 */
inline std::optional<uint32_t> stream_id(const ParseResult<PacketVariant>& result) noexcept {
    if (result.has_value()) {
        return stream_id(*result);
    }
    return std::nullopt;
}

/**
 * @brief Check if a parse result holds a data packet
 * @param result The parse result to check
 * @return true if parsing succeeded and holds a dynamic::DataPacketView, false otherwise
 */
inline bool is_data_packet(const ParseResult<PacketVariant>& result) noexcept {
    return result.has_value() && is_data_packet(*result);
}

/**
 * @brief Check if a parse result holds a context packet
 * @param result The parse result to check
 * @return true if parsing succeeded and holds a dynamic::ContextPacketView, false otherwise
 */
inline bool is_context_packet(const ParseResult<PacketVariant>& result) noexcept {
    return result.has_value() && is_context_packet(*result);
}

} // namespace vrtigo

namespace vrtigo::detail {

/**
 * @brief Extract bytes from PacketVariant and invoke callback
 *
 * This helper visits a PacketVariant, extracts the raw bytes via as_bytes(),
 * and invokes the provided callback with the byte span.
 *
 * @tparam Func Callback type accepting std::span<const uint8_t>
 * @param packet The packet variant to extract bytes from
 * @param fn Callback to invoke with the byte span
 * @return Result of invoking fn with the byte span
 *
 * @note Span lifetime: The returned span is valid only as long as the
 *       underlying packet view's buffer remains valid.
 */
template <typename Func>
auto visit_packet_bytes(const PacketVariant& packet, Func&& fn) noexcept {
    return std::visit(
        [&fn](auto&& pkt) {
            using T = std::decay_t<decltype(pkt)>;
            static_assert(std::is_same_v<T, dynamic::DataPacketView> ||
                              std::is_same_v<T, dynamic::ContextPacketView>,
                          "visit_packet_bytes: unhandled PacketVariant type - update this visitor");
            return fn(pkt.as_bytes());
        },
        packet);
}

} // namespace vrtigo::detail
