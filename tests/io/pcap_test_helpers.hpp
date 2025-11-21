#pragma once

#include <filesystem>
#include <fstream>
#include <span>
#include <vector>

#include <cstdint>
#include <cstring>
#include <vrtigo/detail/endian.hpp>
#include <vrtigo/vrtigo_utils.hpp>

namespace vrtigo::utils::pcapio::test {

// =============================================================================
// VRT Packet Creation Helpers
// =============================================================================

/**
 * @brief Create simple VRT data packet for testing
 *
 * Creates a minimal signal data packet (type 1) with stream ID and payload.
 *
 * @param stream_id Stream identifier for the packet
 * @param payload_words Number of 32-bit words in payload (default: 10)
 * @return Vector containing VRT packet bytes in network byte order
 */
inline std::vector<uint8_t> create_simple_data_packet(uint32_t stream_id,
                                                      size_t payload_words = 10) {
    // Build minimal data packet with stream ID
    std::vector<uint8_t> packet;

    // Header: type=1 (signal data with stream ID), no class ID, no trailer
    // Packet size = 1 (header) + 1 (stream ID) + payload_words
    uint16_t packet_size_words = static_cast<uint16_t>(2 + payload_words);
    uint32_t header = (1 << 28) |                   // Packet type = 1
                      (0 << 27) |                   // No class ID
                      (0 << 26) |                   // No trailer
                      (packet_size_words & 0xFFFF); // Packet size

    // Convert to network byte order (big-endian)
    header = vrtigo::detail::host_to_network32(header);
    packet.resize(packet_size_words * 4);

    std::memcpy(packet.data(), &header, 4);

    // Stream ID (network byte order)
    uint32_t sid = vrtigo::detail::host_to_network32(stream_id);
    std::memcpy(packet.data() + 4, &sid, 4);

    // Payload (zeros for simplicity)
    std::fill(packet.begin() + 8, packet.end(), 0);

    return packet;
}

/**
 * @brief Parse VRT packet bytes into PacketVariant
 *
 * Convenience wrapper around detail::parse_packet for testing.
 *
 * @param bytes Vector of packet bytes
 * @return PacketVariant (RuntimeDataPacket, RuntimeContextPacket, or InvalidPacket)
 */
inline vrtigo::PacketVariant parse_test_packet(const std::vector<uint8_t>& bytes) {
    return vrtigo::detail::parse_packet(std::span<const uint8_t>(bytes.data(), bytes.size()));
}

// =============================================================================
// PCAP Test File Creation
// =============================================================================

/**
 * @brief RAII helper for creating temporary PCAP test files
 *
 * Automatically deletes the file on destruction to keep test directory clean.
 */
class PCAPTestFile {
public:
    explicit PCAPTestFile(const std::filesystem::path& path, bool big_endian = false)
        : path_(path),
          big_endian_(big_endian) {}

    ~PCAPTestFile() {
        if (std::filesystem::exists(path_)) {
            std::filesystem::remove(path_);
        }
    }

    /**
     * @brief Create PCAP file containing VRT packets
     *
     * @param vrt_packets Vector of VRT packet byte arrays to include
     * @param link_header_size Size of link-layer header (default: 14 for Ethernet)
     */
    void create_with_vrt_packets(const std::vector<std::vector<uint8_t>>& vrt_packets,
                                 size_t link_header_size = DEFAULT_LINK_HEADER_SIZE) {
        std::ofstream file(path_, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to create test PCAP file");
        }

        // Write PCAP global header
        write_pcap_global_header(file);

        // Write each VRT packet with PCAP record header and link-layer header
        for (const auto& vrt_packet : vrt_packets) {
            write_pcap_packet_record(file, vrt_packet, link_header_size);
        }

        file.close();
    }

    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
    bool big_endian_;

    /**
     * @brief Write PCAP global header using common format
     */
    void write_pcap_global_header(std::ofstream& file) {
        PCAPGlobalHeader header{
            .magic = big_endian_ ? PCAP_MAGIC_MICROSEC_BE : PCAP_MAGIC_MICROSEC_LE,
            .version_major = PCAP_VERSION_MAJOR,
            .version_minor = PCAP_VERSION_MINOR,
            .thiszone = 0,
            .sigfigs = 0,
            .snaplen = DEFAULT_SNAPLEN,
            .network = PCAP_LINKTYPE_ETHERNET,
        };

        if (big_endian_) {
            header.version_major = vrtigo::detail::byteswap16(header.version_major);
            header.version_minor = vrtigo::detail::byteswap16(header.version_minor);
            header.thiszone = vrtigo::detail::byteswap32(header.thiszone);
            header.sigfigs = vrtigo::detail::byteswap32(header.sigfigs);
            header.snaplen = vrtigo::detail::byteswap32(header.snaplen);
            header.network = vrtigo::detail::byteswap32(header.network);
        }

        file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    }

    /**
     * @brief Write PCAP packet record with VRT payload
     */
    void write_pcap_packet_record(std::ofstream& file, const std::vector<uint8_t>& vrt_packet,
                                  size_t link_header_size) {
        // PCAP packet record header
        PCAPRecordHeader record_header{
            .ts_sec = 1234567890, // Arbitrary timestamp for testing
            .ts_usec = 123456,
            .incl_len = static_cast<uint32_t>(link_header_size + vrt_packet.size()),
            .orig_len = static_cast<uint32_t>(link_header_size + vrt_packet.size()),
        };

        if (big_endian_) {
            record_header.ts_sec = vrtigo::detail::byteswap32(record_header.ts_sec);
            record_header.ts_usec = vrtigo::detail::byteswap32(record_header.ts_usec);
            record_header.incl_len = vrtigo::detail::byteswap32(record_header.incl_len);
            record_header.orig_len = vrtigo::detail::byteswap32(record_header.orig_len);
        }

        file.write(reinterpret_cast<const char*>(&record_header), sizeof(record_header));

        // Write dummy link-layer header (all zeros)
        if (link_header_size > 0) {
            std::vector<uint8_t> dummy_header(link_header_size, 0);
            file.write(reinterpret_cast<const char*>(dummy_header.data()), link_header_size);
        }

        // Write VRT packet
        file.write(reinterpret_cast<const char*>(vrt_packet.data()), vrt_packet.size());
    }
};

} // namespace vrtigo::utils::pcapio::test
