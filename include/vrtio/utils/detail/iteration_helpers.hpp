#pragma once

#include <concepts>
#include <optional>
#include <variant>

#include <cstddef>

#include "../../detail/context_packet_view.hpp"
#include "../../detail/data_packet_view.hpp"
#include "../../detail/packet_variant.hpp"

namespace vrtio::utils::detail {

/**
 * @brief Concept for packet readers that provide read_next_packet()
 *
 * Any reader (file, UDP, etc.) that provides read_next_packet() returning
 * optional<PacketVariant> can use these iteration helpers.
 */
template <typename T>
concept PacketReader = requires(T& reader) {
    { reader.read_next_packet() } -> std::same_as<std::optional<vrtio::PacketVariant>>;
};

/**
 * @brief Iterate over all packets with automatic validation
 *
 * Processes all packets, automatically validating each one.
 * The callback receives a PacketVariant for each packet.
 *
 * @tparam Reader Type satisfying PacketReader concept
 * @tparam Callback Function type with signature: bool(const PacketVariant&)
 * @param reader Reader providing read_next_packet()
 * @param callback Function called for each packet. Return false to stop iteration.
 * @return Number of packets processed
 */
template <PacketReader Reader, typename Callback>
size_t for_each_validated_packet(Reader& reader, Callback&& callback) noexcept {
    size_t count = 0;

    while (auto pkt = reader.read_next_packet()) {
        bool continue_processing = callback(*pkt);
        count++;

        if (!continue_processing) {
            break;
        }
    }

    return count;
}

/**
 * @brief Iterate over data packets only (signal/extension data)
 *
 * Processes only valid data packets (types 0-3), skipping context packets
 * and invalid packets. The callback receives a validated DataPacketView.
 *
 * @tparam Reader Type satisfying PacketReader concept
 * @tparam Callback Function type with signature: bool(const vrtio::DataPacketView&)
 * @param reader Reader providing read_next_packet()
 * @param callback Function called for each data packet. Return false to stop.
 * @return Number of data packets processed
 */
template <PacketReader Reader, typename Callback>
size_t for_each_data_packet(Reader& reader, Callback&& callback) noexcept {
    size_t count = 0;

    while (auto pkt = reader.read_next_packet()) {
        if (auto* data_pkt = std::get_if<vrtio::DataPacketView>(&(*pkt))) {
            bool continue_processing = callback(*data_pkt);
            count++;

            if (!continue_processing) {
                break;
            }
        }
    }

    return count;
}

/**
 * @brief Iterate over context packets only (context/extension context)
 *
 * Processes only valid context packets (types 4-5), skipping data packets
 * and invalid packets. The callback receives a validated ContextPacketView.
 *
 * @tparam Reader Type satisfying PacketReader concept
 * @tparam Callback Function type with signature: bool(const vrtio::ContextPacketView&)
 * @param reader Reader providing read_next_packet()
 * @param callback Function called for each context packet. Return false to stop.
 * @return Number of context packets processed
 */
template <PacketReader Reader, typename Callback>
size_t for_each_context_packet(Reader& reader, Callback&& callback) noexcept {
    size_t count = 0;

    while (auto pkt = reader.read_next_packet()) {
        if (auto* ctx_pkt = std::get_if<vrtio::ContextPacketView>(&(*pkt))) {
            bool continue_processing = callback(*ctx_pkt);
            count++;

            if (!continue_processing) {
                break;
            }
        }
    }

    return count;
}

/**
 * @brief Iterate over packets with a specific stream ID
 *
 * Processes only packets that have a stream ID matching the given value.
 * Skips packets without stream IDs (types 0, 2) and invalid packets.
 *
 * @tparam Reader Type satisfying PacketReader concept
 * @tparam Callback Function type with signature: bool(const PacketVariant&)
 * @param reader Reader providing read_next_packet()
 * @param stream_id_filter The stream ID to filter by
 * @param callback Function called for each matching packet. Return false to stop.
 * @return Number of matching packets processed
 */
template <PacketReader Reader, typename Callback>
size_t for_each_packet_with_stream_id(Reader& reader, uint32_t stream_id_filter,
                                      Callback&& callback) noexcept {
    size_t count = 0;

    while (auto pkt = reader.read_next_packet()) {
        auto sid = vrtio::stream_id(*pkt);
        if (sid && *sid == stream_id_filter) {
            bool continue_processing = callback(*pkt);
            count++;

            if (!continue_processing) {
                break;
            }
        }
    }

    return count;
}

} // namespace vrtio::utils::detail
