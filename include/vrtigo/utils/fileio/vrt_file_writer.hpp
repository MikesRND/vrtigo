// Copyright (c) 2025 Michael Smith
// SPDX-License-Identifier: MIT

#pragma once

#include "vrtigo/detail/packet_concepts.hpp"
#include "vrtigo/detail/packet_variant.hpp"
#include "vrtigo/utils/detail/writer_concepts.hpp"
#include "vrtigo/utils/fileio/raw_vrt_file_writer.hpp"
#include "vrtigo/utils/fileio/writer_status.hpp"

#include <variant>

#include <cerrno>
#include <cstddef>

namespace vrtigo::utils::fileio {

/**
 * @brief High-level VRT file writer with type safety
 *
 * Wraps RawVRTFileWriter to provide type-safe packet writing with
 * automatic validation. Accepts both compile-time packets (from
 * PacketBuilder) and runtime packets (from readers/parsers).
 *
 * Supported Packet Types:
 * - PacketVariant (runtime packets from readers)
 * - dynamic::DataPacket (runtime data packets)
 * - dynamic::ContextPacket (runtime context packets)
 * - Any compile-time packet (from PacketBuilder)
 *
 * Parse Error Handling:
 * - ParseResult errors are rejected (never written)
 * - write_packet() returns false immediately
 * - status() returns WriterStatus::invalid_packet
 * - No bytes written to file
 *
 * Error Propagation:
 * - Combines raw writer errors with high-level validation
 * - status() provides unified error state
 * - errno mapped to appropriate WriterStatus codes
 *
 * Thread Safety:
 * - Not thread-safe: single thread should own this instance
 * - Safe to move between threads (move-only)
 *
 * @tparam MaxPacketWords Maximum packet size in 32-bit words (default 65535)
 */
template <size_t MaxPacketWords = 65535>
class VRTFileWriter {
public:
    /**
     * @brief Create writer for new file
     *
     * Creates or truncates the file at the given path.
     *
     * @param file_path Path to output file
     * @throws std::runtime_error if file cannot be created
     */
    explicit VRTFileWriter(const std::string& file_path)
        : raw_writer_(file_path),
          high_level_status_(WriterStatus::ready) {}

    // Move-only (large buffer, file handle ownership)
    VRTFileWriter(const VRTFileWriter&) = delete;
    VRTFileWriter& operator=(const VRTFileWriter&) = delete;

    VRTFileWriter(VRTFileWriter&&) noexcept = default;
    VRTFileWriter& operator=(VRTFileWriter&&) noexcept = default;

    /**
     * @brief Write packet from variant
     *
     * Writes a packet from a PacketVariant. If the variant holds
     *
     * @param packet The packet variant to write (always valid)
     * @return true on success, false on I/O error
     */
    bool write_packet(const PacketVariant& packet) noexcept {
        // Write the packet using visitor pattern
        // PacketVariant now only contains valid packets (dynamic::DataPacket or
        // dynamic::ContextPacket)
        bool result = std::visit(
            [this](auto&& pkt) -> bool {
                using T = std::decay_t<decltype(pkt)>;

                if constexpr (std::is_same_v<T, vrtigo::dynamic::DataPacket>) {
                    return this->write_packet_impl(pkt);
                } else if constexpr (std::is_same_v<T, vrtigo::dynamic::ContextPacket>) {
                    // dynamic::ContextPacket uses context_buffer() instead of as_bytes()
                    std::span<const uint8_t> bytes{pkt.context_buffer(), pkt.size_bytes()};
                    return this->raw_writer_.write_packet(bytes);
                } else {
                    return false; // Should never reach here
                }
            },
            packet);

        // Clear high-level status on successful write
        if (result) {
            high_level_status_ = WriterStatus::ready;
        }

        return result;
    }

    /**
     * @brief Write data packet view
     *
     * @param packet The data packet to write
     * @return true on success, false on error
     */
    bool write_packet(const vrtigo::dynamic::DataPacket& packet) noexcept {
        bool result = write_packet_impl(packet);
        if (result) {
            high_level_status_ = WriterStatus::ready;
        }
        return result;
    }

    /**
     * @brief Write context packet view
     *
     * @param packet The context packet to write
     * @return true on success, false on error
     */
    bool write_packet(const vrtigo::dynamic::ContextPacket& packet) noexcept {
        // dynamic::ContextPacket uses context_buffer() instead of as_bytes()
        std::span<const uint8_t> bytes{packet.context_buffer(), packet.size_bytes()};
        bool result = raw_writer_.write_packet(bytes);
        if (result) {
            high_level_status_ = WriterStatus::ready;
        }
        return result;
    }

    /**
     * @brief Write compile-time packet
     *
     * Accepts packets from PacketBuilder or any type satisfying
     * CompileTimePacketLike concept. No conversion to variant needed.
     *
     * @tparam PacketType Type satisfying CompileTimePacketLike concept
     * @param packet The packet to write
     * @return true on success, false on error
     */
    template <typename PacketType>
        requires vrtigo::CompileTimePacketLike<PacketType>
    bool write_packet(const PacketType& packet) noexcept {
        auto bytes = packet.as_bytes();
        bool result = raw_writer_.write_packet(bytes);
        if (result) {
            high_level_status_ = WriterStatus::ready;
        }
        return result;
    }

    /**
     * @brief Write multiple packets from iterator range
     *
     * Writes packets from [begin, end). Stops on first error.
     *
     * @tparam Iterator Iterator type yielding packet objects
     * @param begin Beginning of packet range
     * @param end End of packet range
     * @return Number of successfully written packets
     */
    template <typename Iterator>
    size_t write_packets(Iterator begin, Iterator end) noexcept {
        size_t count = 0;
        for (auto it = begin; it != end; ++it) {
            if (!write_packet(*it)) {
                break;
            }
            ++count;
        }
        return count;
    }

    /**
     * @brief Flush buffered data to disk
     *
     * Forces write of any buffered data.
     *
     * @return true on success, false on error
     */
    bool flush() noexcept {
        bool result = raw_writer_.flush();
        if (!result && raw_writer_.has_error()) {
            // Distinguish flush errors from write errors
            high_level_status_ = WriterStatus::flush_error;
        } else if (result) {
            // Clear high-level status on successful flush
            high_level_status_ = WriterStatus::ready;
        }
        return result;
    }

    /**
     * @brief Get unified writer status
     *
     * Combines raw writer errors with high-level validation errors.
     * Checks high-level status first (flush_error, invalid_packet),
     * then raw writer errors, then file state.
     *
     * @return Current writer status
     */
    [[nodiscard]] WriterStatus status() const noexcept {
        // Check high-level status first (flush_error, invalid_packet)
        if (high_level_status_ != WriterStatus::ready) {
            return high_level_status_;
        }

        // Check raw writer errors
        if (raw_writer_.has_error()) {
            return map_errno_to_status(raw_writer_.last_errno());
        }

        // Check file closed state
        if (!raw_writer_.is_open()) {
            return WriterStatus::closed;
        }

        return WriterStatus::ready;
    }

    /**
     * @brief Get number of packets written
     *
     * @return Total packets written successfully
     */
    [[nodiscard]] size_t packets_written() const noexcept { return raw_writer_.packets_written(); }

    /**
     * @brief Get number of bytes written
     *
     * @return Total bytes written (including buffered but not flushed)
     */
    [[nodiscard]] size_t bytes_written() const noexcept { return raw_writer_.bytes_written(); }

    /**
     * @brief Check if file is open
     *
     * @return true if file is open
     */
    [[nodiscard]] bool is_open() const noexcept { return raw_writer_.is_open(); }

    /**
     * @brief Clear error state
     *
     * Resets both raw writer and high-level error state.
     */
    void clear_error() noexcept {
        raw_writer_.clear_error();
        high_level_status_ = WriterStatus::ready;
    }

private:
    /**
     * @brief Write runtime packet view
     *
     * Common implementation for dynamic::DataPacket and dynamic::ContextPacket.
     */
    template <typename PacketView>
    bool write_packet_impl(const PacketView& packet) noexcept {
        auto bytes = packet.as_bytes();
        return raw_writer_.write_packet(bytes);
    }

    /**
     * @brief Map errno to WriterStatus
     *
     * @param err errno value
     * @return Corresponding WriterStatus
     */
    static WriterStatus map_errno_to_status(int err) noexcept {
        switch (err) {
            case 0:
                return WriterStatus::ready;
            case ENOSPC:
                return WriterStatus::disk_full;
            case EACCES:
            case EPERM:
                return WriterStatus::permission_denied;
            default:
                return WriterStatus::write_error;
        }
    }

    RawVRTFileWriter<MaxPacketWords> raw_writer_; ///< Underlying raw writer
    WriterStatus high_level_status_;              ///< High-level validation status
};

} // namespace vrtigo::utils::fileio
