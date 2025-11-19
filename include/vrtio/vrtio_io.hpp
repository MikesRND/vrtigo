#pragma once

/**
 * @file vrtio_io.hpp
 * @brief Convenience header for VRT I/O utilities
 *
 * This header provides I/O types for reading and parsing VRT files with automatic
 * validation and type-safe packet access.
 *
 * Primary types:
 * - VRTFileReader: High-level reader returning validated, type-safe PacketVariant (RECOMMENDED)
 * - RawVRTFileReader: Low-level reader returning raw packet bytes
 * - PacketVariant: Type-safe union of RuntimeDataPacket, RuntimeContextPacket, or InvalidPacket
 */

#include "detail/packet_parser.hpp"
#include "detail/packet_variant.hpp"
#include "utils/fileio/raw_vrt_file_reader.hpp"
#include "utils/fileio/vrt_file_reader.hpp"

namespace vrtio {

// Primary VRT file reader - returns validated, type-safe packet views (RECOMMENDED)
template <uint16_t MaxPacketWords = 65535>
using VRTFileReader = utils::fileio::VRTFileReader<MaxPacketWords>;

// Low-level reader - returns raw packet bytes (for advanced use)
template <uint16_t MaxPacketWords = 65535>
using RawVRTFileReader = utils::fileio::RawVRTFileReader<MaxPacketWords>;

/**
 * @brief Parse and validate a VRT packet from raw bytes
 *
 * Automatically detects packet type and returns a validated packet view.
 * This is the recommended way to parse packets when the packet type is
 * unknown at compile time.
 *
 * Supported packet types:
 * - Signal Data (types 0-1) → RuntimeDataPacket
 * - Extension Data (types 2-3) → RuntimeDataPacket
 * - Context (types 4-5) → RuntimeContextPacket
 * - Command (types 6-7) → InvalidPacket (not yet supported)
 *
 * @param bytes Raw packet bytes (must remain valid while using returned view)
 * @return PacketVariant containing validated view or error information
 *
 * @example
 * std::span<const uint8_t> bytes = get_packet_bytes();
 * auto pkt = vrtio::parse_packet(bytes);
 * if (vrtio::is_data_packet(pkt)) {
 *     const auto& view = std::get<vrtio::RuntimeDataPacket>(pkt);
 *     // Process packet...
 * }
 */
inline PacketVariant parse_packet(std::span<const uint8_t> bytes) noexcept {
    return detail::parse_packet(bytes);
}

} // namespace vrtio
