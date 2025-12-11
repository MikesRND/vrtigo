#pragma once

#include <array>
#include <optional>
#include <string>

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "../../detail/endian.hpp"

namespace vrtigo::utils::pcapio {

// Re-export endian helpers for convenient use in pcapio
using vrtigo::detail::host_to_network16;
using vrtigo::detail::host_to_network32;
using vrtigo::detail::network_to_host16;

// =============================================================================
// PCAP File Format Constants
// =============================================================================

/**
 * @brief PCAP magic numbers for file format identification
 *
 * The magic number indicates both byte order and timestamp precision:
 * - 0xa1b2c3d4: Microsecond precision, little-endian (most common)
 * - 0xd4c3b2a1: Microsecond precision, big-endian
 * - 0xa1b23c4d: Nanosecond precision, little-endian
 * - 0x4d3cb2a1: Nanosecond precision, big-endian
 */
constexpr uint32_t PCAP_MAGIC_MICROSEC_LE = 0xa1b2c3d4;
constexpr uint32_t PCAP_MAGIC_MICROSEC_BE = 0xd4c3b2a1;
constexpr uint32_t PCAP_MAGIC_NANOSEC_LE = 0xa1b23c4d;
constexpr uint32_t PCAP_MAGIC_NANOSEC_BE = 0x4d3cb2a1;

/**
 * @brief PCAP file format version
 *
 * Current stable version is 2.4 (established in 1998)
 */
constexpr uint16_t PCAP_VERSION_MAJOR = 2;
constexpr uint16_t PCAP_VERSION_MINOR = 4;

/**
 * @brief PCAP link-layer types (network field)
 *
 * Common types:
 * - 1: Ethernet (DIX/802.3)
 * - 101: Raw IP (no link layer)
 * - 113: Linux cooked capture (SLL)
 * - 147: User-defined
 *
 * See: https://www.tcpdump.org/linktypes.html
 */
constexpr uint32_t PCAP_LINKTYPE_ETHERNET = 1;
constexpr uint32_t PCAP_LINKTYPE_RAW = 101;
constexpr uint32_t PCAP_LINKTYPE_LINUX_SLL = 113;
constexpr uint32_t PCAP_LINKTYPE_USER0 = 147;

/**
 * @brief PCAP header sizes (fixed by spec)
 */
constexpr size_t PCAP_GLOBAL_HEADER_SIZE = 24; ///< File header size
constexpr size_t PCAP_RECORD_HEADER_SIZE = 16; ///< Per-packet record header size

/**
 * @brief Default values for PCAP files
 */
constexpr uint32_t DEFAULT_SNAPLEN = 65535;     ///< Maximum packet capture length
constexpr size_t DEFAULT_LINK_HEADER_SIZE = 14; ///< Ethernet header size
constexpr size_t MAX_LINK_HEADER_SIZE = 256;    ///< Maximum supported link header

// =============================================================================
// PCAP Header Structures
// =============================================================================

/**
 * @brief PCAP global file header (24 bytes)
 *
 * Appears once at the beginning of every PCAP file.
 * All fields are in host byte order (determined by magic number).
 *
 * @see https://wiki.wireshark.org/Development/LibpcapFileFormat
 */
struct PCAPGlobalHeader {
    uint32_t magic;         ///< Magic number (determines byte order and precision)
    uint16_t version_major; ///< Major version (always 2)
    uint16_t version_minor; ///< Minor version (always 4)
    uint32_t thiszone;      ///< GMT to local correction (usually 0)
    uint32_t sigfigs;       ///< Timestamp accuracy (usually 0)
    uint32_t snaplen;       ///< Max length of captured packets
    uint32_t network;       ///< Link-layer type (1 = Ethernet)
};
static_assert(sizeof(PCAPGlobalHeader) == 24, "PCAPGlobalHeader must be 24 bytes");

/**
 * @brief PCAP packet record header (16 bytes)
 *
 * Precedes each captured packet in the file.
 * All fields are in host byte order (determined by file magic).
 */
struct PCAPRecordHeader {
    uint32_t ts_sec;   ///< Timestamp: seconds since epoch
    uint32_t ts_usec;  ///< Timestamp: microseconds (or nanoseconds if magic indicates)
    uint32_t incl_len; ///< Number of octets of packet saved in file
    uint32_t orig_len; ///< Actual length of packet on wire
};
static_assert(sizeof(PCAPRecordHeader) == 16, "PCAPRecordHeader must be 16 bytes");

// =============================================================================
// Network Header Structures (for UDP encapsulation)
// =============================================================================

/**
 * @brief Ethernet header (14 bytes)
 */
struct EthernetHeader {
    std::array<uint8_t, 6> dst_mac; ///< Destination MAC address
    std::array<uint8_t, 6> src_mac; ///< Source MAC address
    uint16_t ethertype;             ///< EtherType (network byte order)
};
static_assert(sizeof(EthernetHeader) == 14, "EthernetHeader must be 14 bytes");

constexpr uint16_t ETHERTYPE_IPV4_HOST = 0x0800; ///< IPv4 EtherType in host byte order

/**
 * @brief IPv4 header (20 bytes, no options)
 */
struct IPv4Header {
    uint8_t version_ihl;     ///< Version (4) and IHL (5) = 0x45
    uint8_t dscp_ecn;        ///< DSCP and ECN (typically 0)
    uint16_t total_length;   ///< Total packet length (network byte order)
    uint16_t identification; ///< Identification (for fragmentation)
    uint16_t flags_fragment; ///< Flags and fragment offset
    uint8_t ttl;             ///< Time to live
    uint8_t protocol;        ///< Protocol (17 = UDP)
    uint16_t checksum;       ///< Header checksum
    uint32_t src_ip;         ///< Source IP address (network byte order)
    uint32_t dst_ip;         ///< Destination IP address (network byte order)
};
static_assert(sizeof(IPv4Header) == 20, "IPv4Header must be 20 bytes");

constexpr uint8_t IP_PROTOCOL_UDP = 17;
constexpr uint8_t IP_DEFAULT_TTL = 64;

/**
 * @brief UDP header (8 bytes)
 */
struct UDPHeader {
    uint16_t src_port; ///< Source port (network byte order)
    uint16_t dst_port; ///< Destination port (network byte order)
    uint16_t length;   ///< Length including header (network byte order)
    uint16_t checksum; ///< Checksum (0 = disabled)
};
static_assert(sizeof(UDPHeader) == 8, "UDPHeader must be 8 bytes");

/// Total size of Ethernet + IPv4 + UDP headers
constexpr size_t UDP_ENCAP_HEADER_SIZE =
    sizeof(EthernetHeader) + sizeof(IPv4Header) + sizeof(UDPHeader);
static_assert(UDP_ENCAP_HEADER_SIZE == 42, "UDP encapsulation headers must be 42 bytes");

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * @brief Calculate IPv4 header checksum
 *
 * @param header Pointer to IPv4 header (checksum field should be 0)
 * @return Checksum value in network byte order
 */
inline uint16_t calculate_ip_checksum(const IPv4Header* header) noexcept {
    const uint16_t* data = reinterpret_cast<const uint16_t*>(header);
    uint32_t sum = 0;

    // Sum all 16-bit words (10 words = 20 bytes)
    // Convert from network to host order for proper arithmetic
    for (int i = 0; i < 10; ++i) {
        sum += vrtigo::detail::network_to_host16(data[i]);
    }

    // Fold 32-bit sum to 16 bits
    while (sum >> 16) {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }

    // Return in network byte order
    return host_to_network16(static_cast<uint16_t>(~sum));
}

/**
 * @brief Parse IPv4 address string to network byte order uint32_t
 *
 * @param ip_str IP address string (e.g., "10.0.0.1")
 * @return IP address in network byte order, or std::nullopt on parse error
 */
inline std::optional<uint32_t> parse_ipv4(const std::string& ip_str) noexcept {
    uint32_t a, b, c, d;
    if (std::sscanf(ip_str.c_str(), "%u.%u.%u.%u", &a, &b, &c, &d) != 4) {
        return std::nullopt;
    }
    if (a > 255 || b > 255 || c > 255 || d > 255) {
        return std::nullopt;
    }
    // Build IP in host byte order, then convert to network byte order
    uint32_t host_order = (a << 24) | (b << 16) | (c << 8) | d;
    return host_to_network32(host_order);
}

/**
 * @brief Check if magic number is valid PCAP format
 *
 * @param magic The magic number from file header
 * @return true if magic indicates valid PCAP file
 */
constexpr bool is_valid_pcap_magic(uint32_t magic) noexcept {
    return magic == PCAP_MAGIC_MICROSEC_LE || magic == PCAP_MAGIC_MICROSEC_BE ||
           magic == PCAP_MAGIC_NANOSEC_LE || magic == PCAP_MAGIC_NANOSEC_BE;
}

/**
 * @brief Check if PCAP file uses big-endian byte order
 *
 * @param magic The magic number from file header
 * @return true if file uses big-endian format
 */
constexpr bool is_big_endian_pcap(uint32_t magic) noexcept {
    return magic == PCAP_MAGIC_MICROSEC_BE || magic == PCAP_MAGIC_NANOSEC_BE;
}

/**
 * @brief Check if PCAP file uses nanosecond precision
 *
 * @param magic The magic number from file header
 * @return true if timestamps are nanosecond precision (else microsecond)
 */
constexpr bool is_nanosecond_precision(uint32_t magic) noexcept {
    return magic == PCAP_MAGIC_NANOSEC_LE || magic == PCAP_MAGIC_NANOSEC_BE;
}

} // namespace vrtigo::utils::pcapio
