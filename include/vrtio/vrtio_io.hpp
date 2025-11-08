#pragma once

// VRTIO I/O Helpers - Optional Layer
// This header provides optional I/O utilities for VRTIO
// Only available when VRTIO_BUILD_IO_HELPERS is enabled in CMake

#include "vrtio.hpp"

#ifdef VRTIO_HAS_IO_HELPERS
#include "vrtio/io/vrt_file_reader.hpp"

namespace vrtio {
    // Import VRTFileReader into main namespace for convenience
    template<uint16_t MaxPacketWords = 65535>
    using VRTFileReader = io::VRTFileReader<MaxPacketWords>;
}
#else
#error "VRTIO I/O helpers not enabled. Set VRTIO_BUILD_IO_HELPERS=ON in CMake."
#endif
