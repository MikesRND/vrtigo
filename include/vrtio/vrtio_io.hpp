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
 * - PacketVariant: Type-safe union of DataPacketView, ContextPacketView, or InvalidPacket
 */

#include "utils/fileio/packet_variant.hpp"
#include "utils/fileio/raw_vrt_file_reader.hpp"
#include "utils/fileio/vrt_file_reader.hpp"

namespace vrtio {

// Primary VRT file reader - returns validated, type-safe packet views (RECOMMENDED)
template <uint16_t MaxPacketWords = 65535>
using VRTFileReader = utils::fileio::VRTFileReader<MaxPacketWords>;

// Low-level reader - returns raw packet bytes (for advanced use)
template <uint16_t MaxPacketWords = 65535>
using RawVRTFileReader = utils::fileio::RawVRTFileReader<MaxPacketWords>;

// Packet variant types
using PacketVariant = utils::fileio::PacketVariant;
using InvalidPacket = utils::fileio::InvalidPacket;

// Helper functions for working with PacketVariant
using utils::fileio::is_context_packet;
using utils::fileio::is_data_packet;
using utils::fileio::is_valid;
using utils::fileio::packet_type;
using utils::fileio::stream_id;

} // namespace vrtio
