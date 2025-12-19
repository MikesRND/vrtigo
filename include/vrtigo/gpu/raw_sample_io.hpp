#pragma once

// Include guard for extra ODR safety
#ifndef VRTIGO_GPU_RAW_SAMPLE_IO_HPP
    #define VRTIGO_GPU_RAW_SAMPLE_IO_HPP

    #include <algorithm>
    #include <span>

    #include <cstddef>
    #include <cstdint>
    #include <cstring>
    #include <vrtigo/gpu/complex_traits.hpp>

namespace vrtigo::gpu {

/**
 * Read a sample from payload without endian conversion.
 *
 * Use this when the payload is already in host byte order (e.g., GPU kernel
 * output that was written in host byte order, or data that has already been
 * converted).
 *
 * For network byte order payloads, use vrtigo::SampleSpan instead.
 *
 * @tparam T Sample type (scalar or ComplexType)
 * @param payload Raw payload buffer
 * @param index Sample index (0-based)
 * @return Sample value (no byte swap performed)
 */
template <GpuSampleType T>
T read_sample_raw(std::span<const uint8_t> payload, size_t index) noexcept {
    T value;
    if constexpr (ComplexType<T>) {
        // Complex type: sizeof(T) covers both components
        std::memcpy(&value, payload.data() + index * sizeof(T), sizeof(T));
    } else {
        // Scalar type
        std::memcpy(&value, payload.data() + index * sizeof(T), sizeof(T));
    }
    return value;
}

/**
 * Write a sample to payload without endian conversion.
 *
 * Use this when the payload should remain in host byte order (e.g., preparing
 * data for GPU processing where the GPU expects host byte order).
 *
 * For VRT packet payloads requiring network byte order, use vrtigo::SampleSpan.
 *
 * @tparam T Sample type (scalar or ComplexType)
 * @param payload Mutable payload buffer
 * @param index Sample index (0-based)
 * @param sample Sample value to write
 */
template <GpuSampleType T>
void write_sample_raw(std::span<uint8_t> payload, size_t index, T sample) noexcept {
    if constexpr (ComplexType<T>) {
        // Complex type: sizeof(T) covers both components
        std::memcpy(payload.data() + index * sizeof(T), &sample, sizeof(T));
    } else {
        // Scalar type
        std::memcpy(payload.data() + index * sizeof(T), &sample, sizeof(T));
    }
}

/**
 * Bulk copy raw payload bytes without any conversion.
 *
 * Copies bytes directly from source to destination with no endian conversion.
 * Use when payload data is already in the correct byte order for the target.
 *
 * Common use case: GPU kernel has already produced network byte order output,
 * copy directly to VRT packet payload without host-side conversion.
 *
 * @param dest Destination buffer
 * @param src Source buffer
 * @return Number of bytes copied (min of dest.size() and src.size())
 */
inline size_t copy_raw_payload(std::span<uint8_t> dest, std::span<const uint8_t> src) noexcept {
    size_t bytes_to_copy = std::min(dest.size(), src.size());
    std::memcpy(dest.data(), src.data(), bytes_to_copy);
    return bytes_to_copy;
}

/**
 * Read multiple samples from payload without endian conversion.
 *
 * Bulk read variant of read_sample_raw for better performance when
 * reading many consecutive samples.
 *
 * @tparam T Sample type (scalar or ComplexType)
 * @param dest Destination buffer for samples
 * @param payload Source payload buffer
 * @param start_index First sample index to read
 * @return Number of samples read
 */
template <GpuSampleType T>
size_t read_samples_raw(std::span<T> dest, std::span<const uint8_t> payload,
                        size_t start_index = 0) noexcept {
    size_t offset_bytes = start_index * sizeof(T);
    if (offset_bytes >= payload.size()) {
        return 0; // start_index out of bounds
    }
    size_t available_bytes = payload.size() - offset_bytes;
    size_t available_samples = available_bytes / sizeof(T);
    size_t to_read = std::min(dest.size(), available_samples);

    if (to_read > 0) {
        std::memcpy(dest.data(), payload.data() + offset_bytes, to_read * sizeof(T));
    }
    return to_read;
}

/**
 * Write multiple samples to payload without endian conversion.
 *
 * Bulk write variant of write_sample_raw for better performance when
 * writing many consecutive samples.
 *
 * @tparam T Sample type (scalar or ComplexType)
 * @param payload Destination payload buffer
 * @param src Source samples
 * @param start_index First sample index to write
 * @return Number of samples written
 */
template <GpuSampleType T>
size_t write_samples_raw(std::span<uint8_t> payload, std::span<const T> src,
                         size_t start_index = 0) noexcept {
    size_t offset_bytes = start_index * sizeof(T);
    if (offset_bytes >= payload.size()) {
        return 0; // start_index out of bounds
    }
    size_t available_bytes = payload.size() - offset_bytes;
    size_t available_samples = available_bytes / sizeof(T);
    size_t to_write = std::min(src.size(), available_samples);

    if (to_write > 0) {
        std::memcpy(payload.data() + offset_bytes, src.data(), to_write * sizeof(T));
    }
    return to_write;
}

} // namespace vrtigo::gpu

#endif // VRTIGO_GPU_RAW_SAMPLE_IO_HPP
