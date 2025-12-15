// Copyright (c) 2025 Michael Smith
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <span>
#include <stdexcept>
#include <string>

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "../../detail/endian.hpp"
#include "../../detail/packet_parser.hpp"
#include "../../detail/packet_variant.hpp"
#include "../../expected.hpp"
#include "../../types.hpp"
#include "../detail/iteration_helpers.hpp"
#include "../detail/reader_error.hpp"
#include "pcap_common.hpp"

namespace vrtigo::utils::pcapio {

/**
 * @brief Read VRT packets from PCAP capture files
 *
 * Simplified PCAP reader designed for testing and validation. Strips link-layer headers
 * (typically Ethernet) and returns validated VRT packets.
 *
 * **API matches VRTFileReader for drop-in compatibility.**
 *
 * **Auto-detects UDP encapsulation:** When EtherType is IPv4 (0x0800), automatically
 * skips the additional IP and UDP headers (28 bytes). This allows reading both:
 * - Raw VRT in Ethernet frames (14-byte link header)
 * - VRT in UDP/IP/Ethernet frames (42-byte total header)
 *
 * Assumptions for test data:
 * - All packets use same link-layer type
 * - Packets are complete (not truncated by snaplen)
 * - Link-layer header size is constant (auto-detection adds IP/UDP if present)
 * - Both little-endian and big-endian PCAP files are supported
 *
 * Common link-layer header sizes:
 * - Ethernet: 14 bytes (default) - auto-detects UDP encapsulation
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
 * reader.for_each_data_packet([](const dynamic::DataPacketView& pkt) {
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
          nanosecond_precision_(false),
          last_ts_sec_(0),
          last_ts_usec_(0),
          last_src_port_(0),
          last_dst_port_(0),
          last_src_ip_(0),
          last_dst_ip_(0),
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
          nanosecond_precision_(other.nanosecond_precision_),
          last_ts_sec_(other.last_ts_sec_),
          last_ts_usec_(other.last_ts_usec_),
          last_src_port_(other.last_src_port_),
          last_dst_port_(other.last_dst_port_),
          last_src_ip_(other.last_src_ip_),
          last_dst_ip_(other.last_dst_ip_),
          last_status_(other.last_status_),
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
            nanosecond_precision_ = other.nanosecond_precision_;
            last_ts_sec_ = other.last_ts_sec_;
            last_ts_usec_ = other.last_ts_usec_;
            last_src_port_ = other.last_src_port_;
            last_dst_port_ = other.last_dst_port_;
            last_src_ip_ = other.last_src_ip_;
            last_dst_ip_ = other.last_dst_ip_;
            last_status_ = other.last_status_;
            vrt_buffer_ = std::move(other.vrt_buffer_);
            other.file_ = nullptr;
        }
        return *this;
    }

    /// Result type for read operations
    using ReadResult = vrtigo::expected<vrtigo::PacketVariant, utils::ReaderError>;

    /**
     * @brief Read next VRT packet from PCAP file
     *
     * Skips PCAP record header and link-layer header, then validates VRT packet.
     *
     * @return expected<PacketVariant, ReaderError>:
     *         - Value: Valid packet (DataPacketView or ContextPacketView)
     *         - unexpected(EndOfStream{}): End of file reached
     *         - unexpected(IOError{...}): Read error
     *         - unexpected(ParseError{...}): Parse/validation error
     *
     * @note Malformed packets (too small, read errors) are skipped and reading continues.
     */
    ReadResult read_next_packet() noexcept {
        while (true) {
            // Reset status at start of each read attempt
            last_status_ = PCAPReadStatus::ok;

            // Check for EOF
            if (current_offset_ >= file_size_) {
                last_status_ = PCAPReadStatus::eof;
                return vrtigo::unexpected(utils::ReaderError{utils::EndOfStream{}});
            }

            // Read PCAP packet record header
            PCAPRecordHeader record_header;
            if (std::fread(&record_header, sizeof(record_header), 1, file_) != 1) {
                // EOF or read error
                if (std::feof(file_)) {
                    last_status_ = PCAPReadStatus::eof;
                    return vrtigo::unexpected(utils::ReaderError{utils::EndOfStream{}});
                }
                last_status_ = PCAPReadStatus::read_error;
                return vrtigo::unexpected(utils::ReaderError{utils::IOError{
                    utils::IOError::Kind::read_error, errno, PacketType::signal_data_no_id,
                    vrtigo::detail::DecodedHeader{}, std::span<const uint8_t>()}});
            }

            // Normalize record header fields to host endianness
            record_header = normalize_record_header(record_header);

            // Store PCAP timestamp
            last_ts_sec_ = record_header.ts_sec;
            last_ts_usec_ = record_header.ts_usec;

            // Reset network metadata (will be set if UDP encapsulated)
            last_src_port_ = 0;
            last_dst_port_ = 0;
            last_src_ip_ = 0;
            last_dst_ip_ = 0;

            // Extract captured length
            uint32_t incl_len = record_header.incl_len;

            // Sanity check: captured length should be reasonable
            if (incl_len == 0 || incl_len > 65535) {
                // Skip malformed record and try next
                last_status_ = PCAPReadStatus::invalid_pcap;
                current_offset_ = std::ftell(file_);
                continue;
            }

            // Check if we have enough data for link-layer header
            if (incl_len < link_header_size_) {
                // Packet too small - skip and try next
                last_status_ = PCAPReadStatus::packet_truncated;
                std::fseek(file_, incl_len, SEEK_CUR);
                current_offset_ = std::ftell(file_);
                continue;
            }

            // Read and skip link-layer header, detecting UDP encapsulation
            size_t total_header_size = link_header_size_;
            size_t vrt_size_from_headers =
                0; // VRT size derived from IP/UDP lengths (0 = use captured size)

            if (link_header_size_ >= sizeof(EthernetHeader)) {
                // Read Ethernet header to check EtherType
                EthernetHeader eth_header;
                if (std::fread(&eth_header, sizeof(eth_header), 1, file_) != 1) {
                    current_offset_ = std::ftell(file_);
                    continue;
                }

                // Check for IPv4 encapsulation (EtherType 0x0800)
                if (eth_header.ethertype == host_to_network16(ETHERTYPE_IPV4_HOST)) {
                    // Need at least minimum IP header to read IHL
                    if (incl_len < link_header_size_ + sizeof(IPv4Header)) {
                        std::fseek(file_, incl_len - sizeof(EthernetHeader), SEEK_CUR);
                        current_offset_ = std::ftell(file_);
                        continue;
                    }

                    // Skip any remaining link header padding after Ethernet
                    if (link_header_size_ > sizeof(EthernetHeader)) {
                        std::fseek(file_, link_header_size_ - sizeof(EthernetHeader), SEEK_CUR);
                    }

                    // Read IPv4 header to extract IPs and check protocol
                    IPv4Header ip_header;
                    if (std::fread(&ip_header, sizeof(ip_header), 1, file_) != 1) {
                        current_offset_ = std::ftell(file_);
                        continue;
                    }

                    // Extract actual IP header length from IHL field (lower 4 bits, in 32-bit
                    // words)
                    size_t ip_header_len = (ip_header.version_ihl & 0x0F) * 4;
                    if (ip_header_len < sizeof(IPv4Header)) {
                        // Invalid IHL (must be at least 5 = 20 bytes) - skip packet
                        std::fseek(file_, incl_len - link_header_size_ - sizeof(IPv4Header),
                                   SEEK_CUR);
                        current_offset_ = std::ftell(file_);
                        continue;
                    }

                    // Validate IP total_length field
                    uint16_t ip_total_length = network_to_host16(ip_header.total_length);
                    if (ip_total_length < ip_header_len + sizeof(UDPHeader)) {
                        // IP total_length too small for UDP - skip packet
                        std::fseek(file_, incl_len - link_header_size_ - sizeof(IPv4Header),
                                   SEEK_CUR);
                        current_offset_ = std::ftell(file_);
                        continue;
                    }

                    // Revalidate incl_len with actual IP header size
                    if (incl_len < link_header_size_ + ip_header_len + sizeof(UDPHeader)) {
                        // Not enough data for IP header + UDP - skip packet
                        std::fseek(file_, incl_len - link_header_size_ - sizeof(IPv4Header),
                                   SEEK_CUR);
                        current_offset_ = std::ftell(file_);
                        continue;
                    }

                    // Only process UDP packets (protocol 17)
                    if (ip_header.protocol != IP_PROTOCOL_UDP) {
                        // Not UDP - skip rest of packet
                        std::fseek(file_, incl_len - link_header_size_ - sizeof(IPv4Header),
                                   SEEK_CUR);
                        current_offset_ = std::ftell(file_);
                        continue;
                    }

                    last_src_ip_ = ip_header.src_ip;
                    last_dst_ip_ = ip_header.dst_ip;

                    // Skip any IP options (bytes beyond the fixed 20-byte header)
                    if (ip_header_len > sizeof(IPv4Header)) {
                        std::fseek(file_, ip_header_len - sizeof(IPv4Header), SEEK_CUR);
                    }

                    // Read UDP header to extract ports
                    UDPHeader udp_header;
                    if (std::fread(&udp_header, sizeof(udp_header), 1, file_) != 1) {
                        current_offset_ = std::ftell(file_);
                        continue;
                    }
                    last_src_port_ = network_to_host16(udp_header.src_port);
                    last_dst_port_ = network_to_host16(udp_header.dst_port);

                    // Validate UDP length field
                    uint16_t udp_length = network_to_host16(udp_header.length);
                    if (udp_length < sizeof(UDPHeader)) {
                        // UDP length too small - skip packet
                        std::fseek(file_,
                                   incl_len - link_header_size_ - ip_header_len - sizeof(UDPHeader),
                                   SEEK_CUR);
                        current_offset_ = std::ftell(file_);
                        continue;
                    }

                    // Calculate VRT size from protocol headers (use minimum of IP and UDP indicated
                    // sizes)
                    size_t ip_indicated_payload =
                        ip_total_length - ip_header_len - sizeof(UDPHeader);
                    size_t udp_indicated_payload = udp_length - sizeof(UDPHeader);
                    vrt_size_from_headers = std::min(ip_indicated_payload, udp_indicated_payload);

                    total_header_size = link_header_size_ + ip_header_len + sizeof(UDPHeader);
                } else {
                    // Not IPv4 - skip remaining link header only
                    if (link_header_size_ > sizeof(EthernetHeader)) {
                        std::fseek(file_, link_header_size_ - sizeof(EthernetHeader), SEEK_CUR);
                    }
                }
            } else if (link_header_size_ > 0) {
                std::fseek(file_, link_header_size_, SEEK_CUR);
            }

            // Calculate VRT packet size
            // Use protocol-indicated size if available, bounded by captured bytes
            size_t captured_payload = incl_len - total_header_size;
            size_t vrt_size = (vrt_size_from_headers > 0)
                                  ? std::min(vrt_size_from_headers, captured_payload)
                                  : captured_payload;

            // Check if VRT packet size is valid
            if (vrt_size < 4 || vrt_size > vrt_buffer_.size()) {
                // VRT packet too small or too large - skip and try next
                std::fseek(file_, captured_payload, SEEK_CUR);
                current_offset_ = std::ftell(file_);
                continue;
            }

            // Read VRT packet
            if (std::fread(vrt_buffer_.data(), vrt_size, 1, file_) != 1) {
                // Read error or EOF
                if (std::feof(file_)) {
                    last_status_ = PCAPReadStatus::eof;
                    return vrtigo::unexpected(utils::ReaderError{utils::EndOfStream{}});
                }
                last_status_ = PCAPReadStatus::read_error;
                return vrtigo::unexpected(utils::ReaderError{utils::IOError{
                    utils::IOError::Kind::read_error, errno, PacketType::signal_data_no_id,
                    vrtigo::detail::DecodedHeader{}, std::span<const uint8_t>()}});
            }

            // Skip any trailing captured bytes beyond protocol-indicated payload
            if (vrt_size < captured_payload) {
                std::fseek(file_, captured_payload - vrt_size, SEEK_CUR);
            }

            // Update position and counter
            current_offset_ = std::ftell(file_);
            packets_read_++;

            // Validate and return VRT packet
            auto bytes = std::span<const uint8_t>(vrt_buffer_.data(), vrt_size);
            auto parse_result = vrtigo::parse_packet(bytes);
            if (parse_result.has_value()) {
                last_status_ = PCAPReadStatus::ok;
                return *std::move(parse_result);
            }

            // Parse/validation error - set status and return error
            last_status_ = PCAPReadStatus::parse_error;
            return vrtigo::unexpected(utils::ReaderError{parse_result.error()});
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
     * and invalid packets. The callback receives a validated dynamic::DataPacketView.
     *
     * @tparam Callback Function type with signature: bool(const vrtigo::dynamic::DataPacketView&)
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
     * and invalid packets. The callback receives a validated dynamic::ContextPacketView.
     *
     * @tparam Callback Function type with signature: bool(const
     * vrtigo::dynamic::ContextPacketView&)
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

    /**
     * @brief Check if timestamps use nanosecond precision
     *
     * @return true if PCAP file uses nanosecond timestamps, false for microsecond
     */
    bool is_nanosecond_precision() const noexcept { return nanosecond_precision_; }

    /**
     * @brief Get PCAP timestamp seconds from last read packet
     *
     * @return Seconds since epoch from PCAP record header
     */
    uint32_t last_timestamp_sec() const noexcept { return last_ts_sec_; }

    /**
     * @brief Get PCAP timestamp sub-seconds from last read packet
     *
     * @return Microseconds or nanoseconds (check is_nanosecond_precision())
     */
    uint32_t last_timestamp_subsec() const noexcept { return last_ts_usec_; }

    /**
     * @brief Get source UDP port from last read packet
     *
     * @return Source port in host byte order, or 0 if not UDP encapsulated
     */
    uint16_t last_src_port() const noexcept { return last_src_port_; }

    /**
     * @brief Get destination UDP port from last read packet
     *
     * @return Destination port in host byte order, or 0 if not UDP encapsulated
     */
    uint16_t last_dst_port() const noexcept { return last_dst_port_; }

    /**
     * @brief Get source IP address from last read packet
     *
     * @return Source IP in network byte order, or 0 if not UDP encapsulated
     */
    uint32_t last_src_ip() const noexcept { return last_src_ip_; }

    /**
     * @brief Get destination IP address from last read packet
     *
     * @return Destination IP in network byte order, or 0 if not UDP encapsulated
     */
    uint32_t last_dst_ip() const noexcept { return last_dst_ip_; }

    /**
     * @brief Read next VRT packet as raw bytes (no parsing)
     *
     * Reads the raw packet bytes without validation or parsing. This is useful
     * for low-level processing or when you want to defer validation.
     *
     * @return Span valid until next read call; empty span on error/EOF
     * @note After empty span, check last_status() to distinguish EOF from errors
     */
    std::span<const uint8_t> read_next_raw() noexcept;

    /**
     * @brief Error accessor - inspect why read_next_raw() returned empty span
     *
     * @return Status of last read operation
     */
    PCAPReadStatus last_status() const noexcept { return last_status_; }

private:
    FILE* file_;                     ///< File handle
    size_t file_size_;               ///< Total file size in bytes
    size_t current_offset_;          ///< Current read position
    size_t packets_read_;            ///< Number of packets read
    size_t link_header_size_;        ///< Bytes to skip per packet
    size_t pcap_global_header_size_; ///< Size of PCAP global header (24)
    bool big_endian_pcap_;           ///< True if PCAP file uses big-endian byte order
    bool nanosecond_precision_;      ///< True if PCAP uses nanosecond timestamps
    uint32_t last_ts_sec_;           ///< PCAP timestamp seconds of last packet
    uint32_t last_ts_usec_;          ///< PCAP timestamp usec/nsec of last packet
    uint16_t last_src_port_;         ///< Source UDP port of last packet (0 if not UDP)
    uint16_t last_dst_port_;         ///< Destination UDP port of last packet (0 if not UDP)
    uint32_t last_src_ip_;           ///< Source IP of last packet (0 if not UDP)
    uint32_t last_dst_ip_;           ///< Destination IP of last packet (0 if not UDP)
    PCAPReadStatus last_status_ = PCAPReadStatus::ok; ///< Status of last read operation
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
        PCAPRecordHeader swapped{vrtigo::detail::byteswap32(header.ts_sec),
                                 vrtigo::detail::byteswap32(header.ts_usec),
                                 vrtigo::detail::byteswap32(header.incl_len),
                                 vrtigo::detail::byteswap32(header.orig_len)};
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

        // Track endianness and precision for later record header parsing
        big_endian_pcap_ = is_big_endian_pcap(header.magic);
        nanosecond_precision_ = vrtigo::utils::pcapio::is_nanosecond_precision(header.magic);

        // For testing purposes, we don't need to parse version, snaplen, etc.
        // Just validate that it's a PCAP file and position at first packet.
        return true;
    }
};

// =============================================================================
// Out-of-line template implementation
// =============================================================================

template <uint16_t MaxPacketWords>
std::span<const uint8_t> PCAPVRTReader<MaxPacketWords>::read_next_raw() noexcept {
    // Reset status to ok at start
    last_status_ = PCAPReadStatus::ok;

    // Check for EOF
    if (current_offset_ >= file_size_) {
        last_status_ = PCAPReadStatus::eof;
        return {};
    }

    // Read PCAP packet record header
    PCAPRecordHeader record_header;
    if (std::fread(&record_header, sizeof(record_header), 1, file_) != 1) {
        // EOF or read error
        if (std::feof(file_)) {
            last_status_ = PCAPReadStatus::eof;
        } else {
            last_status_ = PCAPReadStatus::read_error;
        }
        return {};
    }

    // Normalize record header fields to host endianness
    record_header = normalize_record_header(record_header);

    // Store PCAP timestamp
    last_ts_sec_ = record_header.ts_sec;
    last_ts_usec_ = record_header.ts_usec;

    // Reset network metadata (will be set if UDP encapsulated)
    last_src_port_ = 0;
    last_dst_port_ = 0;
    last_src_ip_ = 0;
    last_dst_ip_ = 0;

    // Extract captured length
    uint32_t incl_len = record_header.incl_len;

    // Sanity check: captured length should be reasonable
    if (incl_len == 0 || incl_len > 65535) {
        last_status_ = PCAPReadStatus::invalid_pcap;
        std::fseek(file_, incl_len, SEEK_CUR);
        current_offset_ = std::ftell(file_);
        return {};
    }

    // Check if we have enough data for link-layer header
    if (incl_len < link_header_size_) {
        last_status_ = PCAPReadStatus::invalid_pcap;
        std::fseek(file_, incl_len, SEEK_CUR);
        current_offset_ = std::ftell(file_);
        return {};
    }

    // Read and skip link-layer header, detecting UDP encapsulation
    size_t total_header_size = link_header_size_;
    size_t vrt_size_from_headers =
        0; // VRT size derived from IP/UDP lengths (0 = use captured size)

    if (link_header_size_ >= sizeof(EthernetHeader)) {
        // Read Ethernet header to check EtherType
        EthernetHeader eth_header;
        if (std::fread(&eth_header, sizeof(eth_header), 1, file_) != 1) {
            last_status_ = PCAPReadStatus::read_error;
            current_offset_ = std::ftell(file_);
            return {};
        }

        // Check for IPv4 encapsulation (EtherType 0x0800)
        if (eth_header.ethertype == host_to_network16(ETHERTYPE_IPV4_HOST)) {
            // Need at least minimum IP header to read IHL
            if (incl_len < link_header_size_ + sizeof(IPv4Header)) {
                last_status_ = PCAPReadStatus::packet_truncated;
                std::fseek(file_, incl_len - sizeof(EthernetHeader), SEEK_CUR);
                current_offset_ = std::ftell(file_);
                return {};
            }

            // Skip any remaining link header padding after Ethernet
            if (link_header_size_ > sizeof(EthernetHeader)) {
                std::fseek(file_, link_header_size_ - sizeof(EthernetHeader), SEEK_CUR);
            }

            // Read IPv4 header to extract IPs and check protocol
            IPv4Header ip_header;
            if (std::fread(&ip_header, sizeof(ip_header), 1, file_) != 1) {
                last_status_ = PCAPReadStatus::read_error;
                current_offset_ = std::ftell(file_);
                return {};
            }

            // Extract actual IP header length from IHL field (lower 4 bits, in 32-bit words)
            size_t ip_header_len = (ip_header.version_ihl & 0x0F) * 4;
            if (ip_header_len < sizeof(IPv4Header)) {
                // Invalid IHL (must be at least 5 = 20 bytes) - skip packet
                last_status_ = PCAPReadStatus::invalid_pcap;
                std::fseek(file_, incl_len - link_header_size_ - sizeof(IPv4Header), SEEK_CUR);
                current_offset_ = std::ftell(file_);
                return {};
            }

            // Validate IP total_length field
            uint16_t ip_total_length = network_to_host16(ip_header.total_length);
            if (ip_total_length < ip_header_len + sizeof(UDPHeader)) {
                // IP total_length too small for UDP - skip packet
                last_status_ = PCAPReadStatus::invalid_pcap;
                std::fseek(file_, incl_len - link_header_size_ - sizeof(IPv4Header), SEEK_CUR);
                current_offset_ = std::ftell(file_);
                return {};
            }

            // Revalidate incl_len with actual IP header size
            if (incl_len < link_header_size_ + ip_header_len + sizeof(UDPHeader)) {
                // Not enough data for IP header + UDP - skip packet
                last_status_ = PCAPReadStatus::packet_truncated;
                std::fseek(file_, incl_len - link_header_size_ - sizeof(IPv4Header), SEEK_CUR);
                current_offset_ = std::ftell(file_);
                return {};
            }

            // Only process UDP packets (protocol 17)
            if (ip_header.protocol != IP_PROTOCOL_UDP) {
                // Not UDP - skip rest of packet
                std::fseek(file_, incl_len - link_header_size_ - sizeof(IPv4Header), SEEK_CUR);
                current_offset_ = std::ftell(file_);
                last_status_ = PCAPReadStatus::invalid_pcap;
                return {};
            }

            last_src_ip_ = ip_header.src_ip;
            last_dst_ip_ = ip_header.dst_ip;

            // Skip any IP options (bytes beyond the fixed 20-byte header)
            if (ip_header_len > sizeof(IPv4Header)) {
                std::fseek(file_, ip_header_len - sizeof(IPv4Header), SEEK_CUR);
            }

            // Read UDP header to extract ports
            UDPHeader udp_header;
            if (std::fread(&udp_header, sizeof(udp_header), 1, file_) != 1) {
                last_status_ = PCAPReadStatus::read_error;
                current_offset_ = std::ftell(file_);
                return {};
            }
            last_src_port_ = network_to_host16(udp_header.src_port);
            last_dst_port_ = network_to_host16(udp_header.dst_port);

            // Validate UDP length field
            uint16_t udp_length = network_to_host16(udp_header.length);
            if (udp_length < sizeof(UDPHeader)) {
                // UDP length too small - skip packet
                last_status_ = PCAPReadStatus::invalid_pcap;
                std::fseek(file_, incl_len - link_header_size_ - ip_header_len - sizeof(UDPHeader),
                           SEEK_CUR);
                current_offset_ = std::ftell(file_);
                return {};
            }

            // Calculate VRT size from protocol headers (use minimum of IP and UDP indicated sizes)
            size_t ip_indicated_payload = ip_total_length - ip_header_len - sizeof(UDPHeader);
            size_t udp_indicated_payload = udp_length - sizeof(UDPHeader);
            vrt_size_from_headers = std::min(ip_indicated_payload, udp_indicated_payload);

            total_header_size = link_header_size_ + ip_header_len + sizeof(UDPHeader);
        } else {
            // Not IPv4 - skip remaining link header only
            if (link_header_size_ > sizeof(EthernetHeader)) {
                std::fseek(file_, link_header_size_ - sizeof(EthernetHeader), SEEK_CUR);
            }
        }
    } else if (link_header_size_ > 0) {
        std::fseek(file_, link_header_size_, SEEK_CUR);
    }

    // Calculate VRT packet size
    // Use protocol-indicated size if available, bounded by captured bytes
    size_t captured_payload = incl_len - total_header_size;
    size_t vrt_size = (vrt_size_from_headers > 0)
                          ? std::min(vrt_size_from_headers, captured_payload)
                          : captured_payload;

    // Check if VRT packet size is valid
    if (vrt_size < 4 || vrt_size > vrt_buffer_.size()) {
        // VRT packet too small or too large - skip and return error
        last_status_ = PCAPReadStatus::invalid_pcap;
        std::fseek(file_, captured_payload, SEEK_CUR);
        current_offset_ = std::ftell(file_);
        return {};
    }

    // Read VRT packet
    if (std::fread(vrt_buffer_.data(), vrt_size, 1, file_) != 1) {
        // Read error or EOF
        if (std::feof(file_)) {
            last_status_ = PCAPReadStatus::eof;
        } else {
            last_status_ = PCAPReadStatus::read_error;
        }
        return {};
    }

    // Skip any trailing captured bytes beyond protocol-indicated payload
    if (vrt_size < captured_payload) {
        std::fseek(file_, captured_payload - vrt_size, SEEK_CUR);
    }

    // Update position and counter
    current_offset_ = std::ftell(file_);
    packets_read_++;

    // Return raw bytes without validation
    last_status_ = PCAPReadStatus::ok;
    return std::span<const uint8_t>(vrt_buffer_.data(), vrt_size);
}

} // namespace vrtigo::utils::pcapio
