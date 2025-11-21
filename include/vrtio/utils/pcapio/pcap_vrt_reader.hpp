// Copyright (c) 2025 Michael Smith
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "../../detail/endian.hpp"
#include "../../detail/packet_parser.hpp"
#include "../../detail/packet_variant.hpp"
#include "../../types.hpp"
#include "../detail/iteration_helpers.hpp"
#include "pcap_common.hpp"

namespace vrtio::utils::pcapio {

/**
 * @brief Read VRT packets from PCAP capture files
 *
 * Simplified PCAP reader designed for testing and validation. Strips link-layer headers
 * (typically Ethernet) and returns validated VRT packets.
 *
 * **API matches VRTFileReader for drop-in compatibility.**
 *
 * Assumptions for test data:
 * - All packets use same link-layer type
 * - Packets are complete (not truncated by snaplen)
 * - Link-layer header size is constant
 * - Both little-endian and big-endian PCAP files are supported
 *
 * Common link-layer header sizes:
 * - Ethernet: 14 bytes (default)
 * - Raw IP: 0 bytes (no link-layer header)
 * - Linux cooked capture (SLL): 16 bytes
 *
 * @tparam MaxPacketWords Maximum VRT packet size in 32-bit words (default: 65535)
 *
 * @warning This class is MOVE-ONLY due to the large internal scratch buffer.
 *
 * Example usage:
 * @code
 * // Default: assumes Ethernet (14-byte headers)
 * PCAPVRTReader<> reader("test_data.pcap");
 *
 * // Or specify different link-layer size
 * PCAPVRTReader<> reader("raw_capture.pcap", 0);  // No link-layer header
 *
 * // Identical API to VRTFileReader
 * reader.for_each_data_packet([](const RuntimeDataPacket& pkt) {
 *     auto payload = pkt.payload();
 *     validate_payload(payload);
 *     return true;
 * });
 * @endcode
 */
template <uint16_t MaxPacketWords = 65535>
class PCAPVRTReader {
    static_assert(MaxPacketWords > 0, "MaxPacketWords must be positive");
    static_assert(MaxPacketWords <= max_packet_words,
                  "MaxPacketWords exceeds VRT specification maximum (65535)");

public:
    /**
     * @brief Open PCAP file for reading VRT packets
     *
     * Parses PCAP global header and prepares for packet reading.
     *
     * @param filepath Path to PCAP file
     * @param link_header_size Bytes to skip per packet (default: 14 for Ethernet)
     * @throws std::runtime_error if file cannot be opened or has invalid PCAP header
     * @throws std::invalid_argument if link_header_size exceeds MAX_LINK_HEADER_SIZE
     */
    explicit PCAPVRTReader(const char* filepath, size_t link_header_size = 14)
        : file_(nullptr),
          file_size_(0),
          current_offset_(0),
          packets_read_(0),
          link_header_size_(link_header_size),
          pcap_global_header_size_(PCAP_GLOBAL_HEADER_SIZE),
          big_endian_pcap_(false),
          vrt_buffer_{} {
        // Validate link header size
        if (link_header_size_ > MAX_LINK_HEADER_SIZE) {
            throw std::invalid_argument("link_header_size (" + std::to_string(link_header_size_) +
                                        ") exceeds maximum (" +
                                        std::to_string(MAX_LINK_HEADER_SIZE) + ")");
        }

        // Open file
        file_ = std::fopen(filepath, "rb");
        if (!file_) {
            throw std::runtime_error(std::string("Failed to open PCAP file: ") + filepath);
        }

        // Get file size
        std::fseek(file_, 0, SEEK_END);
        file_size_ = std::ftell(file_);
        std::fseek(file_, 0, SEEK_SET);

        // Parse and validate PCAP global header
        if (!parse_global_header()) {
            std::fclose(file_);
            file_ = nullptr;
            throw std::runtime_error(std::string("Invalid PCAP file format: ") + filepath);
        }

        // Position at first packet record
        current_offset_ = pcap_global_header_size_;
    }

    /**
     * @brief Open PCAP file for reading VRT packets
     *
     * @param filepath Path to PCAP file
     * @param link_header_size Bytes to skip per packet (default: 14 for Ethernet)
     * @throws std::runtime_error if file cannot be opened or has invalid PCAP header
     * @throws std::invalid_argument if link_header_size exceeds MAX_LINK_HEADER_SIZE
     */
    explicit PCAPVRTReader(const std::string& filepath, size_t link_header_size = 14)
        : PCAPVRTReader(filepath.c_str(), link_header_size) {}

    /**
     * @brief Destructor - closes file handle
     */
    ~PCAPVRTReader() noexcept {
        if (file_) {
            std::fclose(file_);
        }
    }

    // Non-copyable due to large scratch buffer and FILE* ownership
    PCAPVRTReader(const PCAPVRTReader&) = delete;
    PCAPVRTReader& operator=(const PCAPVRTReader&) = delete;

    // Move-only semantics
    PCAPVRTReader(PCAPVRTReader&& other) noexcept
        : file_(other.file_),
          file_size_(other.file_size_),
          current_offset_(other.current_offset_),
          packets_read_(other.packets_read_),
          link_header_size_(other.link_header_size_),
          pcap_global_header_size_(other.pcap_global_header_size_),
          big_endian_pcap_(other.big_endian_pcap_),
          vrt_buffer_(std::move(other.vrt_buffer_)) {
        other.file_ = nullptr;
    }

    PCAPVRTReader& operator=(PCAPVRTReader&& other) noexcept {
        if (this != &other) {
            if (file_) {
                std::fclose(file_);
            }
            file_ = other.file_;
            file_size_ = other.file_size_;
            current_offset_ = other.current_offset_;
            packets_read_ = other.packets_read_;
            link_header_size_ = other.link_header_size_;
            pcap_global_header_size_ = other.pcap_global_header_size_;
            big_endian_pcap_ = other.big_endian_pcap_;
            vrt_buffer_ = std::move(other.vrt_buffer_);
            other.file_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief Read next VRT packet from PCAP file
     *
     * Skips PCAP record header and link-layer header, then validates VRT packet.
     *
     * @return PacketVariant (RuntimeDataPacket, RuntimeContextPacket, or InvalidPacket),
     *         or std::nullopt on EOF
     *
     * @note Malformed packets (too small, read errors) are skipped and reading continues.
     *       Only true EOF returns std::nullopt.
     */
    std::optional<vrtio::PacketVariant> read_next_packet() noexcept {
        while (true) {
            // Check for EOF
            if (current_offset_ >= file_size_) {
                return std::nullopt;
            }

            // Read PCAP packet record header
            PCAPRecordHeader record_header;
            if (std::fread(&record_header, sizeof(record_header), 1, file_) != 1) {
                return std::nullopt; // EOF or read error
            }

            // Normalize record header fields to host endianness
            record_header = normalize_record_header(record_header);

            // Extract captured length
            uint32_t incl_len = record_header.incl_len;

            // Sanity check: captured length should be reasonable
            if (incl_len == 0 || incl_len > 65535) {
                // Skip malformed record and try next
                current_offset_ = std::ftell(file_);
                continue;
            }

            // Check if we have enough data for link-layer header
            if (incl_len < link_header_size_) {
                // Packet too small - skip and try next
                std::fseek(file_, incl_len, SEEK_CUR);
                current_offset_ = std::ftell(file_);
                continue;
            }

            // Skip link-layer header
            if (link_header_size_ > 0) {
                std::fseek(file_, link_header_size_, SEEK_CUR);
            }

            // Calculate VRT packet size
            size_t vrt_size = incl_len - link_header_size_;

            // Check if VRT packet size is valid
            if (vrt_size < 4 || vrt_size > vrt_buffer_.size()) {
                // VRT packet too small or too large - skip and try next
                std::fseek(file_, vrt_size, SEEK_CUR);
                current_offset_ = std::ftell(file_);
                continue;
            }

            // Read VRT packet
            if (std::fread(vrt_buffer_.data(), vrt_size, 1, file_) != 1) {
                return std::nullopt; // Read error or EOF
            }

            // Update position and counter
            current_offset_ = std::ftell(file_);
            packets_read_++;

            // Validate and return VRT packet
            auto bytes = std::span<const uint8_t>(vrt_buffer_.data(), vrt_size);
            return vrtio::detail::parse_packet(bytes);
        }
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
     */
    template <typename Callback>
    size_t for_each_validated_packet(Callback&& callback) noexcept {
        return detail::for_each_validated_packet(*this, std::forward<Callback>(callback));
    }

    /**
     * @brief Iterate over data packets only (signal/extension data)
     *
     * Processes only valid data packets (types 0-3), skipping context packets
     * and invalid packets. The callback receives a validated RuntimeDataPacket.
     *
     * @tparam Callback Function type with signature: bool(const vrtio::RuntimeDataPacket&)
     * @param callback Function called for each data packet. Return false to stop.
     * @return Number of data packets processed
     */
    template <typename Callback>
    size_t for_each_data_packet(Callback&& callback) noexcept {
        return detail::for_each_data_packet(*this, std::forward<Callback>(callback));
    }

    /**
     * @brief Iterate over context packets only (context/extension context)
     *
     * Processes only valid context packets (types 4-5), skipping data packets
     * and invalid packets. The callback receives a validated RuntimeContextPacket.
     *
     * @tparam Callback Function type with signature: bool(const vrtio::RuntimeContextPacket&)
     * @param callback Function called for each context packet. Return false to stop.
     * @return Number of context packets processed
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
     * @param stream_id_filter The stream ID to filter by
     * @param callback Function called for each matching packet. Return false to stop.
     * @return Number of matching packets processed
     */
    template <typename Callback>
    size_t for_each_packet_with_stream_id(uint32_t stream_id_filter, Callback&& callback) noexcept {
        return detail::for_each_packet_with_stream_id(*this, stream_id_filter,
                                                      std::forward<Callback>(callback));
    }

    /**
     * @brief Rewind file to beginning for re-reading
     *
     * Resets file position to first packet record.
     */
    void rewind() noexcept {
        if (file_) {
            std::fseek(file_, pcap_global_header_size_, SEEK_SET);
            current_offset_ = pcap_global_header_size_;
            packets_read_ = 0;
        }
    }

    /**
     * @brief Get current file position in bytes
     */
    size_t tell() const noexcept { return current_offset_; }

    /**
     * @brief Get total file size in bytes
     */
    size_t size() const noexcept { return file_size_; }

    /**
     * @brief Get number of packets read so far
     */
    size_t packets_read() const noexcept { return packets_read_; }

    /**
     * @brief Check if file is still open
     */
    bool is_open() const noexcept { return file_ != nullptr; }

    /**
     * @brief Get configured link-layer header size
     *
     * @return Number of bytes skipped per packet for link-layer headers
     */
    size_t link_header_size() const noexcept { return link_header_size_; }

    /**
     * @brief Set link-layer header size
     *
     * Allows changing the link-layer header size after construction.
     * Use rewind() to re-read from the beginning with the new setting.
     *
     * @param size Number of bytes to skip per packet
     */
    void set_link_header_size(size_t size) noexcept { link_header_size_ = size; }

private:
    FILE* file_;                     ///< File handle
    size_t file_size_;               ///< Total file size in bytes
    size_t current_offset_;          ///< Current read position
    size_t packets_read_;            ///< Number of packets read
    size_t link_header_size_;        ///< Bytes to skip per packet
    size_t pcap_global_header_size_; ///< Size of PCAP global header (24)
    bool big_endian_pcap_;           ///< True if PCAP file uses big-endian byte order
    std::array<uint8_t, MaxPacketWords * vrt_word_size> vrt_buffer_; ///< VRT packet buffer

    /**
     * @brief Normalize PCAP record header to host endianness
     *
     * Converts record header fields from file byte order to host byte order.
     * If the PCAP file uses little-endian format (most common), returns header as-is.
     * If the PCAP file uses big-endian format, byte-swaps all 32-bit fields.
     *
     * @param header Record header as read from file
     * @return Record header with fields in host byte order
     */
    PCAPRecordHeader normalize_record_header(const PCAPRecordHeader& header) const noexcept {
        if (!big_endian_pcap_) {
            return header; // Already in host byte order
        }

        // Byte-swap all fields from big-endian to host (little-endian)
        PCAPRecordHeader swapped{
            vrtio::detail::byteswap32(header.ts_sec), vrtio::detail::byteswap32(header.ts_usec),
            vrtio::detail::byteswap32(header.incl_len), vrtio::detail::byteswap32(header.orig_len)};
        return swapped;
    }

    /**
     * @brief Parse and validate PCAP global header
     *
     * Reads the 24-byte PCAP global header and validates the magic number.
     * Endianness is inferred from the magic value and used when parsing record headers.
     * Nanosecond precision files are accepted.
     *
     * @return true if valid PCAP header, false otherwise
     */
    bool parse_global_header() noexcept {
        PCAPGlobalHeader header;

        // Read global header
        if (std::fread(&header, sizeof(header), 1, file_) != 1) {
            return false;
        }

        // Validate magic number using common helper
        if (!is_valid_pcap_magic(header.magic)) {
            return false; // Not a valid PCAP file
        }

        // Track endianness for later record header parsing
        big_endian_pcap_ = is_big_endian_pcap(header.magic);

        // For testing purposes, we don't need to parse version, snaplen, etc.
        // Just validate that it's a PCAP file and position at first packet.
        return true;
    }
};

} // namespace vrtio::utils::pcapio
