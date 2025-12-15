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
#include "../../detail/header_decode.hpp"
#include "../../detail/packet_variant.hpp"
#include "../../types.hpp"
#include "pcap_common.hpp"

namespace vrtigo::utils::pcapio {

/**
 * @brief Write VRT packets to PCAP capture files with proper UDP encapsulation
 *
 * PCAP writer that wraps VRT packets in Ethernet/IPv4/UDP headers for
 * analysis with Wireshark/tcpdump. The VITA 49 dissector will automatically
 * decode packets on the configured UDP port (default 4991).
 *
 * Use cases:
 * - Capture live VRT streams to PCAP for analysis with Wireshark/tcpdump
 * - Convert VRT files to PCAP format
 * - Create test data in PCAP format
 *
 * **API matches VRTFileWriter patterns for consistency.**
 *
 * Features:
 * - Proper Ethernet + IPv4 + UDP encapsulation (42-byte header)
 * - Configurable IP addresses and UDP ports
 * - Optional VRT timestamp extraction for accurate pcap timing
 * - Automatic IP header checksum calculation
 * - Implements PacketWriter concept for write_all_packets() helpers
 * - Buffered writes for performance
 *
 * @warning This class is MOVE-ONLY due to file handle ownership.
 *
 * Example usage:
 * @code
 * // Create PCAP with default settings (port 4991, dummy IPs)
 * PCAPVRTWriter writer("output.pcap");
 *
 * // Or with custom settings
 * PCAPVRTWriter writer("output.pcap", 5000, 50000, "192.168.1.100", "192.168.1.200");
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
     * @brief Create PCAP file for writing VRT packets with UDP encapsulation
     *
     * Creates a new PCAP file and writes the global header.
     * If file exists, it will be truncated.
     *
     * @param filepath Path to PCAP file to create
     * @param dst_port Destination UDP port (default: 4991, VITA 49 standard)
     * @param src_port Source UDP port (default: 50000)
     * @param src_ip Source IP address string (default: "10.0.0.1")
     * @param dst_ip Destination IP address string (default: "10.0.0.2")
     * @param snaplen Maximum packet length in PCAP (default: 65535)
     * @throws std::runtime_error if file cannot be created
     * @throws std::invalid_argument if IP address cannot be parsed
     */
    explicit PCAPVRTWriter(const char* filepath, uint16_t dst_port = 4991,
                           uint16_t src_port = 50000, const std::string& src_ip = "10.0.0.1",
                           const std::string& dst_ip = "10.0.0.2", uint32_t snaplen = 65535)
        : fd_(-1),
          packets_written_(0),
          bytes_written_(0),
          ip_identification_(0),
          snaplen_(snaplen),
          write_buffer_{},
          buffer_pos_(0) {
        // Parse and validate IP addresses
        auto parsed_src = parse_ipv4(src_ip);
        auto parsed_dst = parse_ipv4(dst_ip);
        if (!parsed_src.has_value() || !parsed_dst.has_value()) {
            throw std::invalid_argument("Invalid IP address format");
        }
        src_ip_ = *parsed_src;
        dst_ip_ = *parsed_dst;

        // Store ports in network byte order
        src_port_ = host_to_network16(src_port);
        dst_port_ = host_to_network16(dst_port);

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
     * @brief Create PCAP file for writing VRT packets with UDP encapsulation
     *
     * @param filepath Path to PCAP file to create
     * @param dst_port Destination UDP port (default: 4991, VITA 49 standard)
     * @param src_port Source UDP port (default: 50000)
     * @param src_ip Source IP address string (default: "10.0.0.1")
     * @param dst_ip Destination IP address string (default: "10.0.0.2")
     * @param snaplen Maximum packet length in PCAP (default: 65535)
     * @throws std::runtime_error if file cannot be created
     * @throws std::invalid_argument if IP address cannot be parsed
     */
    explicit PCAPVRTWriter(const std::string& filepath, uint16_t dst_port = 4991,
                           uint16_t src_port = 50000, const std::string& src_ip = "10.0.0.1",
                           const std::string& dst_ip = "10.0.0.2", uint32_t snaplen = 65535)
        : PCAPVRTWriter(filepath.c_str(), dst_port, src_port, src_ip, dst_ip, snaplen) {}

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
          ip_identification_(other.ip_identification_),
          src_ip_(other.src_ip_),
          dst_ip_(other.dst_ip_),
          src_port_(other.src_port_),
          dst_port_(other.dst_port_),
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
            ip_identification_ = other.ip_identification_;
            src_ip_ = other.src_ip_;
            dst_ip_ = other.dst_ip_;
            src_port_ = other.src_port_;
            dst_port_ = other.dst_port_;
            snaplen_ = other.snaplen_;
            write_buffer_ = std::move(other.write_buffer_);
            buffer_pos_ = other.buffer_pos_;
            other.fd_ = -1;
        }
        return *this;
    }

    /**
     * @brief Write VRT packet to PCAP file with explicit PCAP timestamp
     *
     * Wraps the VRT packet with PCAP record header and Ethernet/IPv4/UDP headers,
     * then writes to file. Uses internal buffering for performance.
     *
     * This is the most deterministic overload - timestamps are used exactly as provided.
     *
     * @param bytes VRT packet bytes to write
     * @param ts_sec PCAP timestamp seconds (Unix epoch)
     * @param ts_usec PCAP timestamp microseconds (0-999999)
     * @return true on success, false on I/O error
     *
     * @note The span contents are copied; caller's buffer can be reused immediately after return.
     */
    bool write_packet(std::span<const uint8_t> bytes, uint32_t ts_sec, uint32_t ts_usec) noexcept {
        return write_packet_impl(bytes, ts_sec, ts_usec);
    }

    /**
     * @brief Write VRT packet to PCAP file, extracting timestamp from VRT header
     *
     * Parses the VRT header to extract TSI/TSF timestamp fields. If present,
     * converts to PCAP epoch format. If no timestamp present, falls back to system time.
     *
     * @param bytes VRT packet bytes to write
     * @return true on success, false on I/O error
     *
     * @note The span contents are copied; caller's buffer can be reused immediately after return.
     * @note For deterministic timestamps, use the overload with explicit ts_sec/ts_usec parameters.
     */
    bool write_packet(std::span<const uint8_t> bytes) noexcept {
        // Parse minimal VRT header to check for timestamp
        if (bytes.size() < 4) {
            return false; // Invalid packet - need at least header word
        }

        // Read header word
        uint32_t header_word = vrtigo::detail::read_u32(bytes.data(), 0);
        auto decoded = vrtigo::detail::decode_header(header_word);

        uint32_t ts_sec = 0;
        uint32_t ts_usec = 0;
        bool has_vrt_timestamp = false;

        // Check if packet has TSI (integer timestamp)
        if (decoded.tsi != TsiType::none) {
            // Calculate offset to TSI field
            size_t offset = 4; // After header word
            if (decoded.type == PacketType::signal_data_no_id ||
                decoded.type == PacketType::signal_data ||
                decoded.type == PacketType::extension_data_no_id ||
                decoded.type == PacketType::extension_data) {
                // Stream ID present for types 1, 3 (signal_data, extension_data)
                if (decoded.type == PacketType::signal_data ||
                    decoded.type == PacketType::extension_data) {
                    offset += 4;
                }
            } else if (decoded.type == PacketType::context ||
                       decoded.type == PacketType::extension_context) {
                offset += 4; // Stream ID always present for context packets
            }

            if (decoded.has_class_id) {
                offset += 8; // Class ID is 64 bits
            }

            // Read TSI (integer timestamp) - always 32 bits
            if (offset + 4 <= bytes.size()) {
                ts_sec = vrtigo::detail::read_u32(bytes.data(), offset);
                has_vrt_timestamp = true;
                offset += 4;

                // Check if packet has TSF (fractional timestamp)
                if (decoded.tsf != TsfType::none && offset + 8 <= bytes.size()) {
                    uint64_t tsf = vrtigo::detail::read_u64(bytes.data(), offset);

                    // Convert TSF to microseconds based on type
                    if (decoded.tsf == TsfType::real_time) {
                        // Real-time: picoseconds since last integer second
                        constexpr uint64_t PICOS_PER_MICRO = 1'000'000ULL;
                        ts_usec = static_cast<uint32_t>(tsf / PICOS_PER_MICRO);
                    }
                    // For other TSF types, leave ts_usec as 0
                }
            }
        }

        // Fall back to system time if no VRT timestamp
        if (!has_vrt_timestamp) {
            auto now = std::chrono::system_clock::now();
            auto duration = now.time_since_epoch();
            auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
            auto micros = std::chrono::duration_cast<std::chrono::microseconds>(duration - seconds);
            ts_sec = static_cast<uint32_t>(seconds.count());
            ts_usec = static_cast<uint32_t>(micros.count());
        }

        return write_packet_impl(bytes, ts_sec, ts_usec);
    }

    /**
     * @brief Write VRT packet to PCAP file (dynamic path)
     *
     * Extracts VRT timestamp from the packet variant if available,
     * otherwise falls back to system time.
     *
     * @param pkt PacketVariant containing VRT packet (always valid)
     * @return true on success, false on I/O error
     *
     * @note The span contents are copied; caller's buffer can be reused immediately after return.
     * @note For deterministic timestamps, use the overload with explicit ts_sec/ts_usec parameters.
     */
    bool write_packet(const vrtigo::PacketVariant& pkt) noexcept {
        // Extract bytes and timestamp from PacketVariant
        uint32_t vrt_ts_sec = 0;
        uint32_t vrt_ts_usec = 0;
        bool has_vrt_timestamp = false;

        // Use visit_packet_bytes to get the raw bytes span
        auto bytes = vrtigo::detail::visit_packet_bytes(pkt, [](auto span) { return span; });

        // Extract timestamp if this is a data packet
        if (std::holds_alternative<vrtigo::dynamic::DataPacketView>(pkt)) {
            const auto& data_pkt = std::get<vrtigo::dynamic::DataPacketView>(pkt);
            if (auto ts = data_pkt.timestamp()) {
                // Try to narrow to UTC real_time for best conversion
                if (auto utc_ts = ts->template as<TsiType::utc, TsfType::real_time>()) {
                    // Use chrono conversion for accurate microseconds
                    auto tp = utc_ts->to_chrono();
                    auto duration = tp.time_since_epoch();
                    auto secs = std::chrono::duration_cast<std::chrono::seconds>(duration);
                    auto usecs =
                        std::chrono::duration_cast<std::chrono::microseconds>(duration - secs);
                    vrt_ts_sec = static_cast<uint32_t>(secs.count());
                    vrt_ts_usec = static_cast<uint32_t>(usecs.count());
                    has_vrt_timestamp = true;
                } else if (ts->has_tsi()) {
                    // Fallback: use raw values
                    vrt_ts_sec = ts->tsi();
                    has_vrt_timestamp = true;
                    if (ts->has_tsf() && ts->tsf_kind() == TsfType::real_time) {
                        // Convert picoseconds to microseconds
                        constexpr uint64_t PICOS_PER_MICRO = 1'000'000ULL;
                        vrt_ts_usec = static_cast<uint32_t>(ts->tsf() / PICOS_PER_MICRO);
                    }
                }
            }
        }

        // Fall back to system time if no VRT timestamp
        if (!has_vrt_timestamp) {
            auto now = std::chrono::system_clock::now();
            auto duration = now.time_since_epoch();
            auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
            auto micros = std::chrono::duration_cast<std::chrono::microseconds>(duration - seconds);
            vrt_ts_sec = static_cast<uint32_t>(seconds.count());
            vrt_ts_usec = static_cast<uint32_t>(micros.count());
        }

        return write_packet_impl(bytes, vrt_ts_sec, vrt_ts_usec);
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
     * @brief Get configured snaplen (maximum packet length)
     */
    uint32_t snaplen() const noexcept { return snaplen_; }

    /**
     * @brief Get configured destination UDP port (host byte order)
     */
    uint16_t dst_port() const noexcept { return network_to_host16(dst_port_); }

    /**
     * @brief Get configured source UDP port (host byte order)
     */
    uint16_t src_port() const noexcept { return network_to_host16(src_port_); }

private:
    int fd_;                                  ///< File descriptor
    size_t packets_written_;                  ///< Number of packets written
    size_t bytes_written_;                    ///< Total bytes written (excluding buffer)
    uint16_t ip_identification_;              ///< IP identification counter
    uint32_t src_ip_;                         ///< Source IP (network byte order)
    uint32_t dst_ip_;                         ///< Destination IP (network byte order)
    uint16_t src_port_;                       ///< Source UDP port (network byte order)
    uint16_t dst_port_;                       ///< Destination UDP port (network byte order)
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

    /**
     * @brief Internal implementation: Write packet with explicit timestamp
     *
     * Common implementation used by all write_packet overloads.
     *
     * @param vrt_bytes VRT packet bytes
     * @param ts_sec PCAP timestamp seconds (Unix epoch)
     * @param ts_usec PCAP timestamp microseconds (0-999999)
     * @return true on success, false on I/O error
     */
    bool write_packet_impl(std::span<const uint8_t> vrt_bytes, uint32_t ts_sec,
                           uint32_t ts_usec) noexcept {
        if (vrt_bytes.empty()) {
            return false;
        }

        // Calculate sizes
        size_t udp_payload_size = vrt_bytes.size();
        size_t udp_length = sizeof(UDPHeader) + udp_payload_size;
        size_t ip_total_length = sizeof(IPv4Header) + udp_length;
        size_t total_frame_size = UDP_ENCAP_HEADER_SIZE + udp_payload_size;

        // Check if packet exceeds snaplen
        if (total_frame_size > snaplen_) {
            return false;
        }

        // Build PCAP packet record header
        PCAPRecordHeader record_header{
            .ts_sec = ts_sec,
            .ts_usec = ts_usec,
            .incl_len = static_cast<uint32_t>(total_frame_size),
            .orig_len = static_cast<uint32_t>(total_frame_size),
        };

        // Build Ethernet header
        EthernetHeader eth_header{
            .dst_mac = {0x00, 0x00, 0x00, 0x00, 0x00, 0x02}, // Dummy destination
            .src_mac = {0x00, 0x00, 0x00, 0x00, 0x00, 0x01}, // Dummy source
            .ethertype = host_to_network16(ETHERTYPE_IPV4_HOST),
        };

        // Build IPv4 header
        IPv4Header ip_header{
            .version_ihl = 0x45, // IPv4, 5 words (20 bytes)
            .dscp_ecn = 0,
            .total_length = host_to_network16(static_cast<uint16_t>(ip_total_length)),
            .identification = host_to_network16(ip_identification_++),
            .flags_fragment = host_to_network16(0x4000), // Don't fragment
            .ttl = IP_DEFAULT_TTL,
            .protocol = IP_PROTOCOL_UDP,
            .checksum = 0, // Calculated below
            .src_ip = src_ip_,
            .dst_ip = dst_ip_,
        };
        ip_header.checksum = calculate_ip_checksum(&ip_header);

        // Build UDP header
        UDPHeader udp_header{
            .src_port = src_port_,
            .dst_port = dst_port_,
            .length = host_to_network16(static_cast<uint16_t>(udp_length)),
            .checksum = 0, // UDP checksum optional for IPv4
        };

        // Write PCAP record header
        if (!write_to_buffer(reinterpret_cast<const uint8_t*>(&record_header),
                             sizeof(record_header))) {
            return false;
        }

        // Write Ethernet header
        if (!write_to_buffer(reinterpret_cast<const uint8_t*>(&eth_header), sizeof(eth_header))) {
            return false;
        }

        // Write IPv4 header
        if (!write_to_buffer(reinterpret_cast<const uint8_t*>(&ip_header), sizeof(ip_header))) {
            return false;
        }

        // Write UDP header
        if (!write_to_buffer(reinterpret_cast<const uint8_t*>(&udp_header), sizeof(udp_header))) {
            return false;
        }

        // Write VRT packet payload
        if (!write_to_buffer(vrt_bytes.data(), vrt_bytes.size())) {
            return false;
        }

        packets_written_++;
        return true;
    }
};

} // namespace vrtigo::utils::pcapio
