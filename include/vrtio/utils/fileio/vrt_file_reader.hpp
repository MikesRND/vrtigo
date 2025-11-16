#pragma once

#include <optional>
#include <span>
#include <string>
#include <utility>

#include "../../detail/packet_parser.hpp"
#include "../../detail/packet_variant.hpp"
#include "../detail/iteration_helpers.hpp"
#include "raw_vrt_file_reader.hpp"

namespace vrtio::utils::fileio {

/**
 * @brief VRT file reader with automatic validation and type-safe packet views
 *
 * **This is the recommended VRT file reader for most use cases.**
 *
 * Provides:
 * - Automatic packet type detection from header
 * - Built-in validation using packet views
 * - Type-safe access via PacketVariant (DataPacketView, ContextPacketView, or InvalidPacket)
 * - Filtered iteration helpers (by type, stream ID, etc.)
 * - I/O error detection (corrupt/truncated packets returned as InvalidPacket, not silently treated
 * as EOF)
 *
 * Unlike RawVRTFileReader which returns raw bytes, VRTFileReader returns
 * validated packet views ready for immediate field access.
 *
 * Supported packet types:
 * - Signal Data (0-1) -> DataPacketView
 * - Extension Data (2-3) -> DataPacketView
 * - Context (4-5) -> ContextPacketView
 * - Command (6-7) -> InvalidPacket (not yet implemented)
 *
 * @tparam MaxPacketWords Maximum packet size in 32-bit words (default: 65535)
 *
 * @warning This class is MOVE-ONLY due to the large internal buffer in the underlying reader.
 *
 * @see RawVRTFileReader for low-level raw byte access
 *
 * Example usage:
 * @code
 * VRTFileReader<> reader("data.vrt");
 *
 * // Read and automatically validate each packet
 * while (auto pkt = reader.read_next_packet()) {
 *     std::visit([](auto&& p) {
 *         using T = std::decay_t<decltype(p)>;
 *         if constexpr (std::is_same_v<T, vrtio::DataPacketView>) {
 *             auto payload = p.payload();
 *             // Process data...
 *         }
 *         else if constexpr (std::is_same_v<T, vrtio::ContextPacketView>) {
 *             if (auto bw = p.bandwidth()) {
 *                 std::cout << "BW: " << bw->raw_value() << " Hz\n";
 *             }
 *         }
 *     }, *pkt);
 * }
 *
 * // Or use filtered iteration
 * reader.for_each_data_packet([](const vrtio::DataPacketView& pkt) {
 *     // Only valid data packets here
 *     return true; // continue
 * });
 * @endcode
 */
template <uint16_t MaxPacketWords = 65535>
class VRTFileReader {
public:
    /**
     * @brief Open a VRT file for reading with enhanced validation
     *
     * @param filepath Path to VRT binary file
     * @throws std::runtime_error if file cannot be opened
     */
    explicit VRTFileReader(const char* filepath) : reader_(filepath) {}

    /**
     * @brief Open a VRT file for reading with enhanced validation
     *
     * @param filepath Path to VRT binary file
     * @throws std::runtime_error if file cannot be opened
     */
    explicit VRTFileReader(const std::string& filepath) : reader_(filepath.c_str()) {}

    // Non-copyable (underlying reader is non-copyable)
    VRTFileReader(const VRTFileReader&) = delete;
    VRTFileReader& operator=(const VRTFileReader&) = delete;

    // Move-only semantics
    VRTFileReader(VRTFileReader&&) noexcept = default;
    VRTFileReader& operator=(VRTFileReader&&) noexcept = default;

    /**
     * @brief Read next packet as validated view
     *
     * Reads the next packet from the file, automatically detects its type,
     * validates it, and returns a type-safe variant containing the appropriate view.
     *
     * @return PacketVariant (DataPacketView, ContextPacketView, or InvalidPacket),
     *         or std::nullopt on EOF
     *
     * @note I/O errors (corrupt header, truncated payload, etc.) are returned as
     *       InvalidPacket with full error context. Only true EOF returns std::nullopt.
     *       This ensures all file corruption is visible to the caller rather than
     *       being silently treated as end-of-stream.
     *
     * @note The returned view references the internal reader buffer and is valid
     *       until the next read operation.
     */
    std::optional<vrtio::PacketVariant> read_next_packet() noexcept {
        auto bytes = reader_.read_next_span();

        // Check for EOF
        if (bytes.empty() && reader_.last_error().is_eof()) {
            return std::nullopt;
        }

        // Check for I/O error (not EOF) - convert to InvalidPacket
        if (bytes.empty()) {
            const auto& err = reader_.last_error();
            auto decoded = vrtio::detail::decode_header(err.header);
            return vrtio::PacketVariant{vrtio::InvalidPacket{
                err.error, err.type, decoded,
                std::span<const uint8_t>() // Empty span since read failed
            }};
        }

        // Parse and validate the packet
        return vrtio::detail::parse_packet(bytes);
    }

    /**
     * @brief Iterate over all packets with automatic validation
     *
     * Processes all packets in the file, automatically validating each one.
     * The callback receives a PacketVariant for each packet.
     *
     * @tparam Callback Function type with signature: bool(const PacketVariant&)
     * @param callback Function called for each packet. Return false to stop iteration.
     * @return Number of packets processed
     *
     * Example:
     * @code
     * reader.for_each_validated_packet([](const PacketVariant& pkt) {
     *     if (is_valid(pkt)) {
     *         // Process valid packet...
     *     }
     *     return true; // continue
     * });
     * @endcode
     */
    template <typename Callback>
    size_t for_each_validated_packet(Callback&& callback) noexcept {
        return detail::for_each_validated_packet(*this, std::forward<Callback>(callback));
    }

    /**
     * @brief Iterate over data packets only (signal/extension data)
     *
     * Processes only valid data packets (types 0-3), skipping context packets
     * and invalid packets. The callback receives a validated DataPacketView.
     *
     * @tparam Callback Function type with signature: bool(const vrtio::DataPacketView&)
     * @param callback Function called for each data packet. Return false to stop.
     * @return Number of data packets processed
     *
     * Example:
     * @code
     * reader.for_each_data_packet([](const vrtio::DataPacketView& pkt) {
     *     auto payload = pkt.payload();
     *     process_signal_data(payload);
     *     return true;
     * });
     * @endcode
     */
    template <typename Callback>
    size_t for_each_data_packet(Callback&& callback) noexcept {
        return detail::for_each_data_packet(*this, std::forward<Callback>(callback));
    }

    /**
     * @brief Iterate over context packets only (context/extension context)
     *
     * Processes only valid context packets (types 4-5), skipping data packets
     * and invalid packets. The callback receives a validated ContextPacketView.
     *
     * @tparam Callback Function type with signature: bool(const vrtio::ContextPacketView&)
     * @param callback Function called for each context packet. Return false to stop.
     * @return Number of context packets processed
     *
     * Example:
     * @code
     * reader.for_each_context_packet([](const vrtio::ContextPacketView& pkt) {
     *     if (auto bw = pkt.bandwidth()) {
     *         std::cout << "Bandwidth: " << bw->raw_value() << " Hz\n";
     *     }
     *     return true;
     * });
     * @endcode
     */
    template <typename Callback>
    size_t for_each_context_packet(Callback&& callback) noexcept {
        return detail::for_each_context_packet(*this, std::forward<Callback>(callback));
    }

    /**
     * @brief Iterate over packets with a specific stream ID
     *
     * Processes only packets that have a stream ID matching the given value.
     * Skips packets without stream IDs (types 0, 2) and invalid packets.
     *
     * @tparam Callback Function type with signature: bool(const PacketVariant&)
     * @param stream_id The stream ID to filter by
     * @param callback Function called for each matching packet. Return false to stop.
     * @return Number of matching packets processed
     *
     * Example:
     * @code
     * reader.for_each_packet_with_stream_id(0x12345678, [](const PacketVariant& pkt) {
     *     // Process packet with matching stream ID...
     *     return true;
     * });
     * @endcode
     */
    template <typename Callback>
    size_t for_each_packet_with_stream_id(uint32_t stream_id_filter, Callback&& callback) noexcept {
        return detail::for_each_packet_with_stream_id(*this, stream_id_filter,
                                                      std::forward<Callback>(callback));
    }

    /**
     * @brief Rewind file to beginning for re-reading
     */
    void rewind() noexcept { reader_.rewind(); }

    /**
     * @brief Get current file position in bytes
     */
    size_t tell() const noexcept { return reader_.tell(); }

    /**
     * @brief Get total file size in bytes
     */
    size_t size() const noexcept { return reader_.size(); }

    /**
     * @brief Get number of packets read so far
     */
    size_t packets_read() const noexcept { return reader_.packets_read(); }

    /**
     * @brief Check if file is still open
     */
    bool is_open() const noexcept { return reader_.is_open(); }

    /**
     * @brief Access underlying RawVRTFileReader for advanced use
     *
     * Use this if you need direct access to the low-level reader API.
     *
     * @return Reference to underlying RawVRTFileReader
     */
    RawVRTFileReader<MaxPacketWords>& underlying_reader() noexcept { return reader_; }

    /**
     * @brief Access underlying RawVRTFileReader for advanced use (const)
     *
     * @return Const reference to underlying RawVRTFileReader
     */
    const RawVRTFileReader<MaxPacketWords>& underlying_reader() const noexcept { return reader_; }

private:
    RawVRTFileReader<MaxPacketWords> reader_; ///< Underlying low-level reader
};

} // namespace vrtio::utils::fileio
