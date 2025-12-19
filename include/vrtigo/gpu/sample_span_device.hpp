#pragma once

/**
 * @file sample_span_device.hpp
 * @brief Device-side sample read/write operations for GPU kernels
 *
 * This header provides functions for reading and writing VRT samples on CUDA devices.
 * It requires CUDA compilation (__CUDACC__) and provides:
 * - Endian-swapping read/write for VRT network byte order conversion
 * - Raw read/write for pre-swapped data (no conversion)
 * - Strided variants for pitched GPU memory
 * - Batch operations for processing multiple samples
 *
 * Note: std::complex<T> is not supported on CUDA device code. Use POD complex types
 * (e.g., vrtigo::gpu::Complex<T>) with the complex_traits system instead.
 */

#ifndef __CUDACC__
    #error "sample_span_device.hpp requires CUDA compilation (__CUDACC__)"
#endif

#include <type_traits>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "complex_traits.hpp"
#include "endian.hpp"

namespace vrtigo {
namespace gpu {

// ============================================================================
// ScalarSampleType concept (device-specific, complements complex_traits.hpp)
// ============================================================================

/**
 * @brief Concept for scalar sample types (non-complex)
 *
 * Valid types are:
 * - int8_t, int16_t, int32_t (signed integers)
 * - float, double (IEEE 754)
 */
template <typename T>
concept ScalarSampleType =
    std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t> ||
    std::is_same_v<T, float> || std::is_same_v<T, double>;

// ComplexType and GpuSampleType are defined in complex_traits.hpp

// ============================================================================
// Sample size traits
// ============================================================================

namespace detail {

template <typename T>
struct gpu_sample_size;

template <>
struct gpu_sample_size<int8_t> {
    static constexpr size_t value = 1;
    static constexpr size_t component_size = 1;
};

template <>
struct gpu_sample_size<int16_t> {
    static constexpr size_t value = 2;
    static constexpr size_t component_size = 2;
};

template <>
struct gpu_sample_size<int32_t> {
    static constexpr size_t value = 4;
    static constexpr size_t component_size = 4;
};

template <>
struct gpu_sample_size<float> {
    static constexpr size_t value = 4;
    static constexpr size_t component_size = 4;
};

template <>
struct gpu_sample_size<double> {
    static constexpr size_t value = 8;
    static constexpr size_t component_size = 8;
};

// Complex types - determined via complex_traits
template <ComplexType T>
struct gpu_sample_size<T> {
    using component_type = typename complex_traits<T>::value_type;
    static constexpr size_t component_size = gpu_sample_size<component_type>::value;
    static constexpr size_t value = component_size * 2;
};

} // namespace detail

// ============================================================================
// Endian swap helpers for sample types
// ============================================================================

namespace detail {

/**
 * @brief Swap bytes for a scalar sample type (device function)
 */
template <ScalarSampleType T>
__device__ __forceinline__ T swap_sample_bytes(T value) noexcept {
    if constexpr (sizeof(T) == 1) {
        return value; // No swap needed for 8-bit
    } else if constexpr (sizeof(T) == 2) {
        uint16_t tmp;
        memcpy(&tmp, &value, sizeof(T));
        tmp = byteswap16(tmp);
        T result;
        memcpy(&result, &tmp, sizeof(T));
        return result;
    } else if constexpr (sizeof(T) == 4) {
        uint32_t tmp;
        memcpy(&tmp, &value, sizeof(T));
        tmp = byteswap32(tmp);
        T result;
        memcpy(&result, &tmp, sizeof(T));
        return result;
    } else if constexpr (sizeof(T) == 8) {
        uint64_t tmp;
        memcpy(&tmp, &value, sizeof(T));
        tmp = byteswap64(tmp);
        T result;
        memcpy(&result, &tmp, sizeof(T));
        return result;
    }
}

/**
 * @brief Swap bytes for a complex sample type (device function)
 *
 * Swaps each component (real, imag) independently.
 */
template <ComplexType T>
__device__ __forceinline__ T swap_sample_bytes(T value) noexcept {
    using component_type = typename complex_traits<T>::value_type;
    component_type r = complex_traits<T>::real(value);
    component_type i = complex_traits<T>::imag(value);
    r = swap_sample_bytes(r);
    i = swap_sample_bytes(i);
    return complex_traits<T>::make(r, i);
}

} // namespace detail

// ============================================================================
// Read operations
// ============================================================================

/**
 * @brief Read a sample from buffer with endian swap (network to host)
 *
 * @tparam T Sample type (must satisfy GpuSampleType)
 * @param src Pointer to source buffer (network byte order)
 * @return Sample in host byte order
 */
template <GpuSampleType T>
__device__ __forceinline__ T read_sample(const uint8_t* src) noexcept {
    T value;
    memcpy(&value, src, sizeof(T));
    return detail::swap_sample_bytes(value);
}

/**
 * @brief Read a sample from buffer without endian swap (raw)
 *
 * Use when data is already in host byte order or when
 * endian conversion will be done separately.
 *
 * @tparam T Sample type (must satisfy GpuSampleType)
 * @param src Pointer to source buffer
 * @return Sample (no byte order conversion)
 */
template <GpuSampleType T>
__device__ __forceinline__ T read_sample_raw(const uint8_t* src) noexcept {
    T value;
    memcpy(&value, src, sizeof(T));
    return value;
}

/**
 * @brief Read a sample from strided buffer with endian swap
 *
 * @tparam T Sample type (must satisfy GpuSampleType)
 * @param src Base pointer to source buffer
 * @param index Sample index
 * @param stride Byte stride between samples (0 = use sizeof(T))
 * @return Sample in host byte order
 */
template <GpuSampleType T>
__device__ __forceinline__ T read_sample_strided(const uint8_t* src, size_t index,
                                                 size_t stride) noexcept {
    size_t actual_stride = (stride == 0) ? sizeof(T) : stride;
    return read_sample<T>(src + index * actual_stride);
}

/**
 * @brief Read a sample from strided buffer without endian swap
 *
 * @tparam T Sample type (must satisfy GpuSampleType)
 * @param src Base pointer to source buffer
 * @param index Sample index
 * @param stride Byte stride between samples (0 = use sizeof(T))
 * @return Sample (no byte order conversion)
 */
template <GpuSampleType T>
__device__ __forceinline__ T read_sample_strided_raw(const uint8_t* src, size_t index,
                                                     size_t stride) noexcept {
    size_t actual_stride = (stride == 0) ? sizeof(T) : stride;
    return read_sample_raw<T>(src + index * actual_stride);
}

// ============================================================================
// Write operations
// ============================================================================

/**
 * @brief Write a sample to buffer with endian swap (host to network)
 *
 * @tparam T Sample type (must satisfy GpuSampleType)
 * @param dest Pointer to destination buffer
 * @param sample Sample in host byte order
 */
template <GpuSampleType T>
__device__ __forceinline__ void write_sample(uint8_t* dest, T sample) noexcept {
    T swapped = detail::swap_sample_bytes(sample);
    memcpy(dest, &swapped, sizeof(T));
}

/**
 * @brief Write a sample to buffer without endian swap (raw)
 *
 * Use when data is already in network byte order or when
 * endian conversion will be done separately.
 *
 * @tparam T Sample type (must satisfy GpuSampleType)
 * @param dest Pointer to destination buffer
 * @param sample Sample (no byte order conversion applied)
 */
template <GpuSampleType T>
__device__ __forceinline__ void write_sample_raw(uint8_t* dest, T sample) noexcept {
    memcpy(dest, &sample, sizeof(T));
}

/**
 * @brief Write a sample to strided buffer with endian swap
 *
 * @tparam T Sample type (must satisfy GpuSampleType)
 * @param dest Base pointer to destination buffer
 * @param sample Sample in host byte order
 * @param index Sample index
 * @param stride Byte stride between samples (0 = use sizeof(T))
 */
template <GpuSampleType T>
__device__ __forceinline__ void write_sample_strided(uint8_t* dest, T sample, size_t index,
                                                     size_t stride) noexcept {
    size_t actual_stride = (stride == 0) ? sizeof(T) : stride;
    write_sample<T>(dest + index * actual_stride, sample);
}

/**
 * @brief Write a sample to strided buffer without endian swap
 *
 * @tparam T Sample type (must satisfy GpuSampleType)
 * @param dest Base pointer to destination buffer
 * @param sample Sample (no byte order conversion applied)
 * @param index Sample index
 * @param stride Byte stride between samples (0 = use sizeof(T))
 */
template <GpuSampleType T>
__device__ __forceinline__ void write_sample_strided_raw(uint8_t* dest, T sample, size_t index,
                                                         size_t stride) noexcept {
    size_t actual_stride = (stride == 0) ? sizeof(T) : stride;
    write_sample_raw<T>(dest + index * actual_stride, sample);
}

// ============================================================================
// Batch operations
// ============================================================================

/**
 * @brief Read multiple samples from buffer with endian swap
 *
 * @tparam T Sample type (must satisfy GpuSampleType)
 * @param dest Pointer to destination array (host byte order output)
 * @param src Pointer to source buffer (network byte order)
 * @param count Number of samples to read
 * @param src_stride Source byte stride (0 = contiguous)
 */
template <GpuSampleType T>
__device__ void read_samples(T* dest, const uint8_t* src, size_t count,
                             size_t src_stride = 0) noexcept {
    size_t actual_stride = (src_stride == 0) ? sizeof(T) : src_stride;
    for (size_t i = 0; i < count; ++i) {
        dest[i] = read_sample<T>(src + i * actual_stride);
    }
}

/**
 * @brief Read multiple samples from buffer without endian swap
 *
 * @tparam T Sample type (must satisfy GpuSampleType)
 * @param dest Pointer to destination array
 * @param src Pointer to source buffer
 * @param count Number of samples to read
 * @param src_stride Source byte stride (0 = contiguous)
 */
template <GpuSampleType T>
__device__ void read_samples_raw(T* dest, const uint8_t* src, size_t count,
                                 size_t src_stride = 0) noexcept {
    size_t actual_stride = (src_stride == 0) ? sizeof(T) : src_stride;
    for (size_t i = 0; i < count; ++i) {
        dest[i] = read_sample_raw<T>(src + i * actual_stride);
    }
}

/**
 * @brief Write multiple samples to buffer with endian swap
 *
 * @tparam T Sample type (must satisfy GpuSampleType)
 * @param dest Pointer to destination buffer (network byte order output)
 * @param src Pointer to source array (host byte order)
 * @param count Number of samples to write
 * @param dest_stride Destination byte stride (0 = contiguous)
 */
template <GpuSampleType T>
__device__ void write_samples(uint8_t* dest, const T* src, size_t count,
                              size_t dest_stride = 0) noexcept {
    size_t actual_stride = (dest_stride == 0) ? sizeof(T) : dest_stride;
    for (size_t i = 0; i < count; ++i) {
        write_sample<T>(dest + i * actual_stride, src[i]);
    }
}

/**
 * @brief Write multiple samples to buffer without endian swap
 *
 * @tparam T Sample type (must satisfy GpuSampleType)
 * @param dest Pointer to destination buffer
 * @param src Pointer to source array
 * @param count Number of samples to write
 * @param dest_stride Destination byte stride (0 = contiguous)
 */
template <GpuSampleType T>
__device__ void write_samples_raw(uint8_t* dest, const T* src, size_t count,
                                  size_t dest_stride = 0) noexcept {
    size_t actual_stride = (dest_stride == 0) ? sizeof(T) : dest_stride;
    for (size_t i = 0; i < count; ++i) {
        write_sample_raw<T>(dest + i * actual_stride, src[i]);
    }
}

} // namespace gpu
} // namespace vrtigo
