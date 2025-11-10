#pragma once

// VRTIO I/O Helpers
// This header provides I/O utilities for VRTIO

#include "vrtio/io/vrt_file_reader.hpp"

#include "vrtio.hpp"

namespace vrtio {
// Import VRTFileReader into main namespace for convenience
template <uint16_t MaxPacketWords = 65535>
using VRTFileReader = io::VRTFileReader<MaxPacketWords>;
} // namespace vrtio
