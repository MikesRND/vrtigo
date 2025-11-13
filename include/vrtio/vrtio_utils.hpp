#pragma once

// VRTIO Utilities
// Optional utilities that may allocate memory and use exceptions

#include "vrtio/utils/fileio/vrt_file_reader.hpp"

#include "vrtio.hpp"

namespace vrtio {
// Import VRTFileReader into main namespace for convenience
template <uint16_t MaxPacketWords = 65535>
using VRTFileReader = utils::fileio::VRTFileReader<MaxPacketWords>;
} // namespace vrtio
