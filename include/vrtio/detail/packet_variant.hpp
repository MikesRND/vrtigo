#pragma once

#include <span>
#include <string>
#include <variant>

#include "../types.hpp"
#include "context_packet_view.hpp"
#include "data_packet_view.hpp"
#include "header_decode.hpp"

namespace vrtio {

/**
 * @brief Error result when a packet fails validation
 *
 * Contains all context needed to debug invalid packets, including the
 * validation error, packet type information, and raw packet bytes.
 */
struct InvalidPacket {
    ValidationError error;              ///< The validation error that occurred
    PacketType attempted_type;          ///< The packet type detected from header
    detail::DecodedHeader header;       ///< Decoded header information
    std::span<const uint8_t> raw_bytes; ///< Raw packet bytes for debugging

    /**
     * @brief Get a human-readable error message
     * @return Description of the validation error
     */
    std::string error_message() const { return std::string(validation_error_string(error)); }
};

/**
 * @brief Type-safe variant holding all possible validated packet views
 *
 * After parsing a packet, it will be validated and returned as one of:
 * - DataPacketView: Signal or Extension data packets (types 0-3)
 * - ContextPacketView: Context or Extension Context packets (types 4-5)
 * - InvalidPacket: Validation failed or unsupported packet type (types 6-7)
 */
using PacketVariant = std::variant<DataPacketView,    // Signal/Extension data packets
                                   ContextPacketView, // Context/Extension context packets
                                   InvalidPacket      // Validation failed or unsupported type
                                   >;

/**
 * @brief Check if a packet variant holds a valid packet
 * @param pkt The packet variant to check
 * @return true if the packet is valid (not InvalidPacket), false otherwise
 */
inline bool is_valid(const PacketVariant& pkt) noexcept {
    return !std::holds_alternative<InvalidPacket>(pkt);
}

/**
 * @brief Get the packet type from a packet variant
 * @param pkt The packet variant
 * @return The packet type
 */
inline PacketType packet_type(const PacketVariant& pkt) noexcept {
    return std::visit(
        [](auto&& p) -> PacketType {
            using T = std::decay_t<decltype(p)>;

            if constexpr (std::is_same_v<T, DataPacketView>) {
                return p.type();
            } else if constexpr (std::is_same_v<T, ContextPacketView>) {
                // ContextPacketView doesn't expose type(), so we need to infer it
                // Context packets are always type 4 or 5
                // We can't tell which without accessing internals, so default to 4
                return PacketType::context;
            } else if constexpr (std::is_same_v<T, InvalidPacket>) {
                return p.attempted_type;
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

            if constexpr (std::is_same_v<T, DataPacketView>) {
                return p.stream_id();
            } else if constexpr (std::is_same_v<T, ContextPacketView>) {
                return p.stream_id();
            }

            return std::nullopt;
        },
        pkt);
}

/**
 * @brief Check if a packet variant holds a data packet
 * @param pkt The packet variant to check
 * @return true if the packet is a DataPacketView, false otherwise
 */
inline bool is_data_packet(const PacketVariant& pkt) noexcept {
    return std::holds_alternative<DataPacketView>(pkt);
}

/**
 * @brief Check if a packet variant holds a context packet
 * @param pkt The packet variant to check
 * @return true if the packet is a ContextPacketView, false otherwise
 */
inline bool is_context_packet(const PacketVariant& pkt) noexcept {
    return std::holds_alternative<ContextPacketView>(pkt);
}

} // namespace vrtio
