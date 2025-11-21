#pragma once

#include "vrtio/utils/detail/writer_concepts.hpp"

#include <cstddef>

namespace vrtio::utils::detail {

/**
 * @brief Write all packets from an iterator range
 *
 * Writes packets from the range [begin, end) to the writer.
 * Stops on the first write error and returns the count of
 * successfully written packets.
 *
 * @tparam Writer Type satisfying PacketWriter concept
 * @tparam Iterator Iterator type yielding packet objects
 * @param writer The packet writer
 * @param begin Beginning of packet range
 * @param end End of packet range
 * @return Number of successfully written packets
 *
 * @note Caller should check writer.status() after this call
 *       to determine if an error occurred.
 */
template <PacketWriter Writer, typename Iterator>
size_t write_all_packets(Writer& writer, Iterator begin, Iterator end) noexcept {
    size_t count = 0;
    for (auto it = begin; it != end; ++it) {
        if (!writer.write_packet(*it)) {
            break; // Stop on first error
        }
        ++count;
    }
    return count;
}

/**
 * @brief Write all packets from an iterator range and flush
 *
 * Writes packets from the range [begin, end) to the writer,
 * then flushes any buffered data. Only available for writers
 * that support flushing (FlushableWriter concept).
 *
 * @tparam Writer Type satisfying FlushableWriter concept
 * @tparam Iterator Iterator type yielding packet objects
 * @param writer The packet writer
 * @param begin Beginning of packet range
 * @param end End of packet range
 * @return Number of successfully written packets
 *
 * @note If flush fails, the return count still reflects successfully
 *       written packets. Caller should check writer.status().
 */
template <FlushableWriter Writer, typename Iterator>
size_t write_all_packets_and_flush(Writer& writer, Iterator begin, Iterator end) noexcept {
    size_t count = write_all_packets(writer, begin, end);
    writer.flush(); // Flush regardless of write success/failure
    return count;
}

} // namespace vrtio::utils::detail
