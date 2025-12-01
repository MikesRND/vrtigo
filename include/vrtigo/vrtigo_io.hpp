#pragma once

/**
 * @file vrtigo_io.hpp
 * @brief Convenience header for VRT I/O utilities
 *
 * This header provides I/O types for reading and parsing VRT files and PCAP captures
 * with automatic validation and type-safe packet access.
 *
 * Primary types:
 * - VRTFileReader: High-level reader returning ParseResult<PacketVariant> (RECOMMENDED)
 * - RawVRTFileReader: Low-level reader returning raw packet bytes
 * - PCAPVRTReader: Read VRT packets from PCAP capture files (for testing/validation)
 * - PCAPVRTWriter: Write VRT packets to PCAP capture files (for testing/validation)
 * - PacketVariant: Type-safe union of dynamic::DataPacketView or dynamic::ContextPacketView
 * - ParseResult: Result wrapper with either valid packet or ParseError
 *
 * Parsing functions (defined in packet_parser.hpp, exported here):
 * - parse_packet(): Parse any packet type, returns ParseResult<PacketVariant>
 * - parse_data_packet(): Parse data packet, returns ParseResult<dynamic::DataPacketView>
 * - parse_context_packet(): Parse context packet, returns ParseResult<dynamic::ContextPacketView>
 */

#include "detail/packet_parser.hpp"
#include "detail/packet_variant.hpp"
#include "detail/parse_error.hpp"
#include "detail/parse_result.hpp"
#include "utils/fileio/raw_vrt_file_reader.hpp"
#include "utils/fileio/vrt_file_reader.hpp"
#include "utils/pcapio/pcap_vrt_reader.hpp"
#include "utils/pcapio/pcap_vrt_writer.hpp"

namespace vrtigo {

// Primary VRT file reader - returns validated, type-safe packet views (RECOMMENDED)
template <uint16_t MaxPacketWords = 65535>
using VRTFileReader = utils::fileio::VRTFileReader<MaxPacketWords>;

// Low-level reader - returns raw packet bytes (for advanced use)
template <uint16_t MaxPacketWords = 65535>
using RawVRTFileReader = utils::fileio::RawVRTFileReader<MaxPacketWords>;

// PCAP reader - extracts VRT packets from PCAP capture files (for testing/validation)
template <uint16_t MaxPacketWords = 65535>
using PCAPVRTReader = utils::pcapio::PCAPVRTReader<MaxPacketWords>;

// PCAP writer - writes VRT packets to PCAP capture files (for testing/validation)
using PCAPVRTWriter = utils::pcapio::PCAPVRTWriter;

// parse_packet(), parse_data_packet(), and parse_context_packet() are
// defined in packet_parser.hpp and are in the vrtigo namespace

} // namespace vrtigo
