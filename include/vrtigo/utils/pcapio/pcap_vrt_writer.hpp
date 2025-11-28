// Copyright (c) 2025 Michael Smith
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <chrono>
#include <span>
#include <stdexcept>
#include <string>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

#include "../../detail/buffer_io.hpp"
#include "../../detail/packet_variant.hpp"
#include "../../types.hpp"
#include "pcap_common.hpp"

namespace vrtigo::utils::pcapio {

/**
 * @brief Write VRT packets to PCAP capture files
 *
 * Simplified PCAP writer designed for testing and validation. Wraps VRT packets
 * with PCAP record headers and optional link-layer headers (typically Ethernet).
 *
 * Use cases:
 * - Capture live VRT streams to PCAP for analysis with Wireshark/tcpdump
 * - Convert VRT files to PCAP format
 * - Create test data in PCAP format
 *
 * **API matches VRTFileWriter patterns for consistency.**
 *
 * Features:
 * - Automatic PCAP global header generation
 * - Automatic timestamp generation (microsecond precision)
 * - Configurable link-layer header size
 * - Implements PacketWriter concept for write_all_packets() helpers
 * - Buffered writes for performance
 *
 * @warning This class is MOVE-ONLY due to file handle ownership.
 *
 * Example usage:
 * @code
 * // Create PCAP with Ethernet encapsulation (14-byte headers)
 * PCAPVRTWriter writer("output.pcap");
 *
 * // Write VRT packets
 * VRTFileReader<> reader("input.vrt");
 * while (auto pkt = reader.read_next_packet()) {
 *     writer.write_packet(*pkt);
 * }
 * writer.flush();
 *
 * // Or use write_all_packets helper
 * write_all_packets_and_flush(reader, writer);
 * @endcode
 */
class PCAPVRTWriter {
public:
    /**
     * @brief Create PCAP file for writing VRT packets
     *
     * Creates a new PCAP file and writes the global header.
     * If file exists, it will be truncated.
     *
     * @param filepath Path to PCAP file to create
     * @param link_header_size Bytes of link-layer header per packet (default: 14 for Ethernet)
     * @param snaplen Maximum packet length in PCAP (default: 65535)
     * @throws std::runtime_error if file cannot be created
     * @throws std::invalid_argument if link_header_size exceeds MAX_LINK_HEADER_SIZE
     */
    explicit PCAPVRTWriter(const char* filepath, size_t link_header_size = 14,
                           uint32_t snaplen = 65535)
        : fd_(-1),
          packets_written_(0),
          bytes_written_(0),
          link_header_size_(link_header_size),
          snaplen_(snaplen),
          write_buffer_{},
          buffer_pos_(0) {
        // Validate link header size to prevent buffer overruns
        if (link_header_size_ > MAX_LINK_HEADER_SIZE) {
            throw std::invalid_argument("link_header_size (" + std::to_string(link_header_size_) +
                                        ") exceeds maximum (" +
                                        std::to_string(MAX_LINK_HEADER_SIZE) + ")");
        }

        // Open file for writing (create or truncate)
        fd_ = ::open(filepath, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd_ < 0) {
            throw std::runtime_error(std::string("Failed to create PCAP file: ") + filepath);
        }

        // Write PCAP global header
        if (!write_global_header()) {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error(std::string("Failed to write PCAP global header: ") +
                                     filepath);
        }
    }

    /**
     * @brief Create PCAP file for writing VRT packets
     *
     * @param filepath Path to PCAP file to create
     * @param link_header_size Bytes of link-layer header per packet (default: 14 for Ethernet)
     * @param snaplen Maximum packet length in PCAP (default: 65535)
     * @throws std::runtime_error if file cannot be created
     * @throws std::invalid_argument if link_header_size exceeds MAX_LINK_HEADER_SIZE
     */
    explicit PCAPVRTWriter(const std::string& filepath, size_t link_header_size = 14,
                           uint32_t snaplen = 65535)
        : PCAPVRTWriter(filepath.c_str(), link_header_size, snaplen) {}

    /**
     * @brief Destructor - flushes and closes file
     */
    ~PCAPVRTWriter() noexcept {
        if (fd_ >= 0) {
            flush();
            ::close(fd_);
        }
    }

    // Non-copyable due to file descriptor ownership
    PCAPVRTWriter(const PCAPVRTWriter&) = delete;
    PCAPVRTWriter& operator=(const PCAPVRTWriter&) = delete;

    // Move-only semantics
    PCAPVRTWriter(PCAPVRTWriter&& other) noexcept
        : fd_(other.fd_),
          packets_written_(other.packets_written_),
          bytes_written_(other.bytes_written_),
          link_header_size_(other.link_header_size_),
          snaplen_(other.snaplen_),
          write_buffer_(std::move(other.write_buffer_)),
          buffer_pos_(other.buffer_pos_) {
        other.fd_ = -1;
    }

    PCAPVRTWriter& operator=(PCAPVRTWriter&& other) noexcept {
        if (this != &other) {
            if (fd_ >= 0) {
                flush();
                ::close(fd_);
            }
            fd_ = other.fd_;
            packets_written_ = other.packets_written_;
            bytes_written_ = other.bytes_written_;
            link_header_size_ = other.link_header_size_;
            snaplen_ = other.snaplen_;
            write_buffer_ = std::move(other.write_buffer_);
            buffer_pos_ = other.buffer_pos_;
            other.fd_ = -1;
        }
        return *this;
    }

    /**
     * @brief Write VRT packet to PCAP file
     *
     * Wraps the VRT packet with PCAP record header and link-layer header,
     * then writes to file. Uses internal buffering for performance.
     *
     * @param pkt PacketVariant containing VRT packet (always valid)
     * @return true on success, false on I/O error
     *
     * @note Timestamps are generated automatically using system time
     */
    bool write_packet(const vrtigo::PacketVariant& pkt) noexcept {
        // Get packet buffer
        // PacketVariant now only contains valid packets (RuntimeDataPacket or RuntimeContextPacket)
        std::span<const uint8_t> vrt_bytes;
        std::visit(
            [&vrt_bytes](auto&& p) {
                using T = std::decay_t<decltype(p)>;
                if constexpr (std::is_same_v<T, vrtigo::RuntimeDataPacket>) {
                    vrt_bytes = p.as_bytes();
                } else if constexpr (std::is_same_v<T, vrtigo::RuntimeContextPacket>) {
                    vrt_bytes = std::span<const uint8_t>{p.context_buffer(), p.size_bytes()};
                }
            },
            pkt);

        if (vrt_bytes.empty()) {
            return false;
        }

        // Check if packet exceeds snaplen
        size_t total_size = link_header_size_ + vrt_bytes.size();
        if (total_size > snaplen_) {
            return false; // Packet too large
        }

        // Generate timestamp (microseconds since epoch)
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto micros = std::chrono::duration_cast<std::chrono::microseconds>(duration - seconds);

        // Build PCAP packet record header
        PCAPRecordHeader record_header{
            .ts_sec = static_cast<uint32_t>(seconds.count()),
            .ts_usec = static_cast<uint32_t>(micros.count()),
            .incl_len = static_cast<uint32_t>(total_size),
            .orig_len = static_cast<uint32_t>(total_size),
        };

        // Write PCAP record header
        if (!write_to_buffer(reinterpret_cast<const uint8_t*>(&record_header),
                             sizeof(record_header))) {
            return false;
        }

        // Write dummy link-layer header (all zeros)
        // Note: link_header_size_ validated <= MAX_LINK_HEADER_SIZE in constructor
        if (link_header_size_ > 0 && link_header_size_ <= MAX_LINK_HEADER_SIZE) {
            std::array<uint8_t, MAX_LINK_HEADER_SIZE> dummy_header{};
            if (!write_to_buffer(dummy_header.data(), link_header_size_)) {
                return false;
            }
        }

        // Write VRT packet
        if (!write_to_buffer(vrt_bytes.data(), vrt_bytes.size())) {
            return false;
        }

        packets_written_++;
        return true;
    }

    /**
     * @brief Flush internal write buffer to disk
     *
     * Forces all buffered data to be written to the file.
     * Called automatically by destructor.
     *
     * @return true on success, false on error
     */
    bool flush() noexcept {
        if (fd_ < 0 || buffer_pos_ == 0) {
            return true;
        }

        ssize_t written = ::write(fd_, write_buffer_.data(), buffer_pos_);
        if (written < 0 || static_cast<size_t>(written) != buffer_pos_) {
            return false;
        }

        bytes_written_ += buffer_pos_;
        buffer_pos_ = 0;
        return true;
    }

    /**
     * @brief Get number of packets written so far
     */
    size_t packets_written() const noexcept { return packets_written_; }

    /**
     * @brief Get number of bytes written to file (excluding buffer)
     */
    size_t bytes_written() const noexcept { return bytes_written_; }

    /**
     * @brief Check if file is still open
     */
    bool is_open() const noexcept { return fd_ >= 0; }

    /**
     * @brief Get configured link-layer header size
     */
    size_t link_header_size() const noexcept { return link_header_size_; }

    /**
     * @brief Get configured snaplen (maximum packet length)
     */
    uint32_t snaplen() const noexcept { return snaplen_; }

private:
    int fd_;                                  ///< File descriptor
    size_t packets_written_;                  ///< Number of packets written
    size_t bytes_written_;                    ///< Total bytes written (excluding buffer)
    size_t link_header_size_;                 ///< Bytes of link-layer header per packet
    uint32_t snaplen_;                        ///< Maximum packet length
    std::array<uint8_t, 65536> write_buffer_; ///< Internal write buffer
    size_t buffer_pos_;                       ///< Current position in write buffer

    /**
     * @brief Write PCAP global header (24 bytes)
     */
    bool write_global_header() noexcept {
        PCAPGlobalHeader header{
            .magic = PCAP_MAGIC_MICROSEC_LE,
            .version_major = PCAP_VERSION_MAJOR,
            .version_minor = PCAP_VERSION_MINOR,
            .thiszone = 0,
            .sigfigs = 0,
            .snaplen = snaplen_,
            .network = PCAP_LINKTYPE_ETHERNET,
        };

        // Write directly (not through buffer)
        ssize_t written = ::write(fd_, &header, sizeof(header));
        if (written < 0 || static_cast<size_t>(written) != sizeof(header)) {
            return false;
        }

        bytes_written_ += sizeof(header);
        return true;
    }

    /**
     * @brief Write data to internal buffer, flushing if needed
     */
    bool write_to_buffer(const uint8_t* data, size_t size) noexcept {
        // If data is larger than buffer, flush and write directly
        if (size > write_buffer_.size()) {
            if (!flush()) {
                return false;
            }
            ssize_t written = ::write(fd_, data, size);
            if (written < 0 || static_cast<size_t>(written) != size) {
                return false;
            }
            bytes_written_ += size;
            return true;
        }

        // If data doesn't fit in buffer, flush first
        if (buffer_pos_ + size > write_buffer_.size()) {
            if (!flush()) {
                return false;
            }
        }

        // Copy to buffer
        std::memcpy(write_buffer_.data() + buffer_pos_, data, size);
        buffer_pos_ += size;

        return true;
    }
};

} // namespace vrtigo::utils::pcapio
