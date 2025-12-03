#pragma once

// VRTIGO Utilities
// Optional utilities that may allocate memory and use exceptions

// File I/O
#include "vrtigo/utils/fileio/raw_vrt_file_writer.hpp"
#include "vrtigo/utils/fileio/vrt_file_reader.hpp"
#include "vrtigo/utils/fileio/vrt_file_writer.hpp"

// PCAP I/O
#include "vrtigo/utils/pcapio/pcap_vrt_reader.hpp"
#include "vrtigo/utils/pcapio/pcap_vrt_writer.hpp"

// Network I/O (Linux/POSIX)
#if defined(__linux__) || defined(__unix__) || defined(__APPLE__)
    #include "vrtigo/utils/netio/udp_vrt_reader.hpp"
    #include "vrtigo/utils/netio/udp_vrt_writer.hpp"
#endif
// Synthetic sample clock helper (throws on invalid input)
#include "vrtigo/utils/sample_clock.hpp"
// Deferred start time specification for SampleClock
#include "vrtigo/utils/start_time.hpp"

#include "vrtigo.hpp"

namespace vrtigo {
// Import utilities into main namespace for convenience
template <uint16_t MaxPacketWords = 65535>
using VRTFileReader = utils::fileio::VRTFileReader<MaxPacketWords>;

template <size_t MaxPacketWords = 65535>
using VRTFileWriter = utils::fileio::VRTFileWriter<MaxPacketWords>;

template <size_t MaxPacketWords = 65535>
using RawVRTFileWriter = utils::fileio::RawVRTFileWriter<MaxPacketWords>;

template <uint16_t MaxPacketWords = 65535>
using PCAPVRTReader = utils::pcapio::PCAPVRTReader<MaxPacketWords>;

using PCAPVRTWriter = utils::pcapio::PCAPVRTWriter;

#if defined(__linux__) || defined(__unix__) || defined(__APPLE__)
template <uint16_t MaxPacketWords = 65535>
using UDPVRTReader = utils::netio::UDPVRTReader<MaxPacketWords>;

using UDPVRTWriter = utils::netio::UDPVRTWriter;
#endif
} // namespace vrtigo
