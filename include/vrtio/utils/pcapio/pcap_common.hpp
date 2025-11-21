#pragma once

#include <cstdint>

namespace vrtio::utils::pcapio {

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
// Helper Functions
// =============================================================================

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

} // namespace vrtio::utils::pcapio
