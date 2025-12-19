#pragma once

/**
 * @file vrtigo_gpu.hpp
 * @brief GPU extensions for vrtigo - opt-in aggregator header.
 *
 * This header provides GPU-compatible utilities for working with VRT packets:
 * - POD complex types that work in CUDA device code
 * - Device-side sample I/O with optional endian conversion
 * - GPU memory management utilities (DeviceBuffer, PinnedBuffer)
 * - Device-side packet parsing
 *
 * REQUIREMENTS:
 * - Must define VRTIGO_ENABLE_CUDA before including, or link with vrtigo::gpu target
 * - Requires CUDA toolkit for device code compilation
 *
 * INCLUDE ORDER:
 * - For SampleSpan<POD complex> support, include sample_traits_ext.hpp BEFORE sample_span.hpp:
 *   @code
 *   #include <vrtigo/gpu/sample_traits_ext.hpp>  // MUST come first
 *   #include <vrtigo/sample_span.hpp>
 *   @endcode
 *
 * - Or include this header before vrtigo.hpp for automatic setup:
 *   @code
 *   #include <vrtigo/vrtigo_gpu.hpp>
 *   #include <vrtigo.hpp>
 *   @endcode
 */

#ifndef VRTIGO_ENABLE_CUDA
    #error "vrtigo_gpu.hpp requires VRTIGO_ENABLE_CUDA to be defined. " \
       "Either define it before including, or use CMake with -DVRTIGO_ENABLE_CUDA=ON"
#endif

// Foundation
#include "vrtigo/gpu/complex.hpp"
#include "vrtigo/gpu/complex_traits.hpp"
#include "vrtigo/gpu/detail/cuda_macros.hpp"

// Host-side extensions (include before sample_span.hpp for SampleSpan<POD> support)
#include "vrtigo/gpu/pod_complex_adapter.hpp"
#include "vrtigo/gpu/raw_sample_io.hpp"
#include "vrtigo/gpu/sample_traits_ext.hpp"

// Device-side utilities
#include "vrtigo/gpu/endian.hpp"
#include "vrtigo/gpu/packet_parser.hpp"

// Only include device sample I/O when compiling with nvcc
#ifdef __CUDACC__
    #include "vrtigo/gpu/sample_span_device.hpp"
#endif

// Memory management (requires CUDA runtime)
#include "vrtigo/gpu/memory.hpp"

namespace vrtigo::gpu {

/**
 * @brief Convenience type aliases for common sample buffer types.
 */
template <typename T>
using ComplexSamples = DeviceBuffer<Complex<T>>;

using Complex8Samples = ComplexSamples<int8_t>;
using Complex16Samples = ComplexSamples<int16_t>;
using Complex32Samples = ComplexSamples<int32_t>;
using ComplexFSamples = ComplexSamples<float>;
using ComplexDSamples = ComplexSamples<double>;

} // namespace vrtigo::gpu
