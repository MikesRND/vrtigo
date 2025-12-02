#pragma once

#include <concepts>
#include <variant>

#include <cstddef>

#include "../../detail/packet_variant.hpp"
#include "../../detail/parse_result.hpp"
#include "../../dynamic.hpp"
#include "../../expected.hpp"
#include "reader_error.hpp"

namespace vrtigo::utils::detail {

/**
 * @brief Concept for packet readers that provide read_next_packet()
 *
 * Any reader (file, UDP, etc.) that provides read_next_packet() returning
 * expected<PacketVariant, ReaderError> can use these iteration helpers.
 */
template <typename T>
concept PacketReader = requires(T& reader) {
    {
        reader.read_next_packet()
    } -> std::same_as<vrtigo::expected<vrtigo::PacketVariant, ReaderError>>;
};

/**
 * @brief Iterate over all valid packets
 *
 * Processes all successfully parsed packets.
 *
 * Error handling contract:
 * - EndOfStream: Stop iteration (normal termination)
 * - IOError: Stop iteration (unrecoverable)
 * - ParseError: Skip and continue (single bad packet)
 *
 * @tparam Reader Type satisfying PacketReader concept
 * @tparam Callback Function type with signature: bool(const PacketVariant&)
 * @param reader Reader providing read_next_packet()
 * @param callback Function called for each valid packet. Return false to stop iteration.
 * @return Number of valid packets processed
 */
template <PacketReader Reader, typename Callback>
size_t for_each_validated_packet(Reader& reader, Callback&& callback) noexcept {
    size_t count = 0;

    while (true) {
        auto result = reader.read_next_packet();

        if (!result.has_value()) {
            const auto& err = result.error();

            // EOF or I/O error -> stop iteration
            if (is_eof(err) || is_io_error(err)) {
                break;
            }

            // ParseError -> skip this packet, continue reading
            continue;
        }

        // Valid packet - invoke callback
        if (!callback(*result)) {
            break; // Callback requested stop
        }
        ++count;
    }

    return count;
}

/**
 * @brief Iterate over data packets only (signal/extension data)
 *
 * Processes only valid data packets (types 0-3), skipping context packets.
 *
 * Error handling contract:
 * - EndOfStream: Stop iteration (normal termination)
 * - IOError: Stop iteration (unrecoverable)
 * - ParseError: Skip and continue (single bad packet)
 *
 * @tparam Reader Type satisfying PacketReader concept
 * @tparam Callback Function type with signature: bool(const vrtigo::dynamic::DataPacketView&)
 * @param reader Reader providing read_next_packet()
 * @param callback Function called for each data packet. Return false to stop.
 * @return Number of data packets processed
 */
template <PacketReader Reader, typename Callback>
size_t for_each_data_packet(Reader& reader, Callback&& callback) noexcept {
    size_t count = 0;

    while (true) {
        auto result = reader.read_next_packet();

        if (!result.has_value()) {
            const auto& err = result.error();

            // EOF or I/O error -> stop iteration
            if (is_eof(err) || is_io_error(err)) {
                break;
            }

            // ParseError -> skip this packet, continue reading
            continue;
        }

        if (auto* data_pkt = std::get_if<vrtigo::dynamic::DataPacketView>(&*result)) {
            if (!callback(*data_pkt)) {
                break; // Callback requested stop
            }
            ++count;
        }
    }

    return count;
}

/**
 * @brief Iterate over context packets only (context/extension context)
 *
 * Processes only valid context packets (types 4-5), skipping data packets.
 *
 * Error handling contract:
 * - EndOfStream: Stop iteration (normal termination)
 * - IOError: Stop iteration (unrecoverable)
 * - ParseError: Skip and continue (single bad packet)
 *
 * @tparam Reader Type satisfying PacketReader concept
 * @tparam Callback Function type with signature: bool(const vrtigo::dynamic::ContextPacketView&)
 * @param reader Reader providing read_next_packet()
 * @param callback Function called for each context packet. Return false to stop.
 * @return Number of context packets processed
 */
template <PacketReader Reader, typename Callback>
size_t for_each_context_packet(Reader& reader, Callback&& callback) noexcept {
    size_t count = 0;

    while (true) {
        auto result = reader.read_next_packet();

        if (!result.has_value()) {
            const auto& err = result.error();

            // EOF or I/O error -> stop iteration
            if (is_eof(err) || is_io_error(err)) {
                break;
            }

            // ParseError -> skip this packet, continue reading
            continue;
        }

        if (auto* ctx_pkt = std::get_if<vrtigo::dynamic::ContextPacketView>(&*result)) {
            if (!callback(*ctx_pkt)) {
                break; // Callback requested stop
            }
            ++count;
        }
    }

    return count;
}

/**
 * @brief Iterate over packets with a specific stream ID
 *
 * Processes only valid packets that have a stream ID matching the given value.
 * Skips packets without stream IDs (types 0, 2).
 *
 * Error handling contract:
 * - EndOfStream: Stop iteration (normal termination)
 * - IOError: Stop iteration (unrecoverable)
 * - ParseError: Skip and continue (single bad packet)
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

    while (true) {
        auto result = reader.read_next_packet();

        if (!result.has_value()) {
            const auto& err = result.error();

            // EOF or I/O error -> stop iteration
            if (is_eof(err) || is_io_error(err)) {
                break;
            }

            // ParseError -> skip this packet, continue reading
            continue;
        }

        auto sid = vrtigo::stream_id(*result);
        if (sid && *sid == stream_id_filter) {
            if (!callback(*result)) {
                break; // Callback requested stop
            }
            ++count;
        }
    }

    return count;
}

} // namespace vrtigo::utils::detail
