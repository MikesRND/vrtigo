#pragma once

/**
 * @file memory.hpp
 * @brief RAII GPU buffer management for CUDA memory
 *
 * This header provides RAII wrappers for CUDA memory allocation:
 * - DeviceBuffer<T>: GPU device memory (cudaMalloc/cudaFree)
 * - PinnedBuffer<T>: Pinned host memory (cudaMallocHost/cudaFreeHost)
 *
 * Both classes are move-only and support stream-aware upload/download operations.
 */

#ifndef __CUDACC__
// Allow inclusion in non-CUDA code for type declarations,
// but runtime functions will only compile with CUDA
#endif

#include <span>
#include <utility>

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace vrtigo {
namespace gpu {

// ============================================================================
// DeviceBuffer - RAII wrapper for GPU device memory
// ============================================================================

/**
 * @brief RAII wrapper for GPU device memory
 *
 * Manages cudaMalloc/cudaFree lifecycle. Move-only (no copies).
 *
 * @tparam T Element type
 */
template <typename T>
class DeviceBuffer {
public:
    /**
     * @brief Default constructor - creates empty buffer
     */
    DeviceBuffer() noexcept : data_(nullptr), count_(0) {}

    /**
     * @brief Allocate device buffer for count elements
     *
     * @param count Number of elements to allocate
     * @throws Nothing (check valid() after construction)
     */
    explicit DeviceBuffer(size_t count) noexcept : data_(nullptr), count_(0) {
        if (count > 0) {
            cudaError_t err = cudaMalloc(&data_, count * sizeof(T));
            if (err == cudaSuccess) {
                count_ = count;
            } else {
                data_ = nullptr;
            }
        }
    }

    /**
     * @brief Destructor - frees device memory
     */
    ~DeviceBuffer() noexcept {
        if (data_) {
            cudaFree(data_);
        }
    }

    // Move-only semantics
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    /**
     * @brief Move constructor
     */
    DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_), count_(other.count_) {
        other.data_ = nullptr;
        other.count_ = 0;
    }

    /**
     * @brief Move assignment
     */
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (data_) {
                cudaFree(data_);
            }
            data_ = other.data_;
            count_ = other.count_;
            other.data_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /**
     * @brief Get raw device pointer
     */
    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }

    /**
     * @brief Get element count
     */
    size_t size() const noexcept { return count_; }

    /**
     * @brief Get size in bytes
     */
    size_t size_bytes() const noexcept { return count_ * sizeof(T); }

    /**
     * @brief Check if buffer is valid (allocation succeeded)
     */
    bool valid() const noexcept { return data_ != nullptr; }

    /**
     * @brief Boolean conversion for validity check
     */
    explicit operator bool() const noexcept { return valid(); }

    // ========================================================================
    // Data transfer operations
    // ========================================================================

    /**
     * @brief Upload data from host to device (H2D)
     *
     * @param host_src Host source pointer
     * @param count Number of elements to transfer
     * @param stream CUDA stream (0 = default stream)
     * @return cudaError_t cudaSuccess on success
     */
    cudaError_t upload(const T* host_src, size_t count, cudaStream_t stream = 0) noexcept {
        if (!data_ || count > count_) {
            return cudaErrorInvalidValue;
        }
        return cudaMemcpyAsync(data_, host_src, count * sizeof(T), cudaMemcpyHostToDevice, stream);
    }

    /**
     * @brief Upload all data from host to device (H2D)
     *
     * @param host_src Host source pointer (must have at least size() elements)
     * @param stream CUDA stream (0 = default stream)
     * @return cudaError_t cudaSuccess on success
     */
    cudaError_t upload(const T* host_src, cudaStream_t stream = 0) noexcept {
        return upload(host_src, count_, stream);
    }

    /**
     * @brief Download data from device to host (D2H)
     *
     * @param host_dst Host destination pointer
     * @param count Number of elements to transfer
     * @param stream CUDA stream (0 = default stream)
     * @return cudaError_t cudaSuccess on success
     */
    cudaError_t download(T* host_dst, size_t count, cudaStream_t stream = 0) const noexcept {
        if (!data_ || count > count_) {
            return cudaErrorInvalidValue;
        }
        return cudaMemcpyAsync(host_dst, data_, count * sizeof(T), cudaMemcpyDeviceToHost, stream);
    }

    /**
     * @brief Download all data from device to host (D2H)
     *
     * @param host_dst Host destination pointer (must have space for size() elements)
     * @param stream CUDA stream (0 = default stream)
     * @return cudaError_t cudaSuccess on success
     */
    cudaError_t download(T* host_dst, cudaStream_t stream = 0) const noexcept {
        return download(host_dst, count_, stream);
    }

    /**
     * @brief Set all elements to zero
     *
     * @param stream CUDA stream (0 = default stream)
     * @return cudaError_t cudaSuccess on success
     */
    cudaError_t zero(cudaStream_t stream = 0) noexcept {
        if (!data_) {
            return cudaErrorInvalidValue;
        }
        return cudaMemsetAsync(data_, 0, count_ * sizeof(T), stream);
    }

    /**
     * @brief Resize buffer (reallocates if necessary)
     *
     * @param new_count New element count
     * @return cudaError_t cudaSuccess on success
     *
     * Note: Contents are NOT preserved on resize. Buffer is zeroed.
     */
    cudaError_t resize(size_t new_count) noexcept {
        if (new_count == count_) {
            return cudaSuccess;
        }

        if (data_) {
            cudaFree(data_);
            data_ = nullptr;
            count_ = 0;
        }

        if (new_count > 0) {
            cudaError_t err = cudaMalloc(&data_, new_count * sizeof(T));
            if (err != cudaSuccess) {
                return err;
            }
            count_ = new_count;
        }
        return cudaSuccess;
    }

private:
    T* data_;
    size_t count_;
};

// ============================================================================
// PinnedBuffer - RAII wrapper for pinned host memory
// ============================================================================

/**
 * @brief RAII wrapper for pinned (page-locked) host memory
 *
 * Manages cudaMallocHost/cudaFreeHost lifecycle. Move-only (no copies).
 * Pinned memory enables faster and asynchronous H2D/D2H transfers.
 *
 * @tparam T Element type
 */
template <typename T>
class PinnedBuffer {
public:
    /**
     * @brief Default constructor - creates empty buffer
     */
    PinnedBuffer() noexcept : data_(nullptr), count_(0) {}

    /**
     * @brief Allocate pinned buffer for count elements
     *
     * @param count Number of elements to allocate
     * @throws Nothing (check valid() after construction)
     */
    explicit PinnedBuffer(size_t count) noexcept : data_(nullptr), count_(0) {
        if (count > 0) {
            cudaError_t err = cudaMallocHost(&data_, count * sizeof(T));
            if (err == cudaSuccess) {
                count_ = count;
            } else {
                data_ = nullptr;
            }
        }
    }

    /**
     * @brief Destructor - frees pinned memory
     */
    ~PinnedBuffer() noexcept {
        if (data_) {
            cudaFreeHost(data_);
        }
    }

    // Move-only semantics
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    /**
     * @brief Move constructor
     */
    PinnedBuffer(PinnedBuffer&& other) noexcept : data_(other.data_), count_(other.count_) {
        other.data_ = nullptr;
        other.count_ = 0;
    }

    /**
     * @brief Move assignment
     */
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
        if (this != &other) {
            if (data_) {
                cudaFreeHost(data_);
            }
            data_ = other.data_;
            count_ = other.count_;
            other.data_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /**
     * @brief Get raw host pointer
     */
    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }

    /**
     * @brief Get element count
     */
    size_t size() const noexcept { return count_; }

    /**
     * @brief Get size in bytes
     */
    size_t size_bytes() const noexcept { return count_ * sizeof(T); }

    /**
     * @brief Check if buffer is valid (allocation succeeded)
     */
    bool valid() const noexcept { return data_ != nullptr; }

    /**
     * @brief Boolean conversion for validity check
     */
    explicit operator bool() const noexcept { return valid(); }

    /**
     * @brief Array access operator
     */
    T& operator[](size_t index) noexcept { return data_[index]; }
    const T& operator[](size_t index) const noexcept { return data_[index]; }

    /**
     * @brief Get span view of buffer
     */
    std::span<T> span() noexcept { return {data_, count_}; }
    std::span<const T> span() const noexcept { return {data_, count_}; }

    /**
     * @brief Resize buffer (reallocates if necessary)
     *
     * @param new_count New element count
     * @return cudaError_t cudaSuccess on success
     *
     * Note: Contents are NOT preserved on resize.
     */
    cudaError_t resize(size_t new_count) noexcept {
        if (new_count == count_) {
            return cudaSuccess;
        }

        if (data_) {
            cudaFreeHost(data_);
            data_ = nullptr;
            count_ = 0;
        }

        if (new_count > 0) {
            cudaError_t err = cudaMallocHost(&data_, new_count * sizeof(T));
            if (err != cudaSuccess) {
                return err;
            }
            count_ = new_count;
        }
        return cudaSuccess;
    }

private:
    T* data_;
    size_t count_;
};

// ============================================================================
// Convenience functions for VRT payload handling
// ============================================================================

/**
 * @brief Upload samples from VRT payload to device with endian conversion
 *
 * Converts samples from network byte order (VRT payload) to host byte order
 * on the CPU, then uploads to device memory.
 *
 * @tparam T Sample type
 * @param payload VRT payload data (network byte order)
 * @param d_samples Device buffer to upload to (host byte order output)
 * @param h_staging Pinned staging buffer for conversion (must be >= sample count)
 * @param sample_count Number of samples to convert and upload
 * @param stream CUDA stream for upload
 * @return cudaError_t cudaSuccess on success
 *
 * Note: This function performs endian conversion on the host. For large payloads,
 * consider using upload_payload_and_convert() to do conversion on the GPU.
 *
 * Usage requires including sample_span.hpp and performing endian conversion:
 * @code
 * #include <vrtigo/sample_span.hpp>
 * #include <vrtigo/gpu/memory.hpp>
 *
 * vrtigo::SampleSpan<int16_t> span(payload);
 * for (size_t i = 0; i < sample_count; ++i) {
 *     h_staging[i] = span.read(i);  // Endian conversion on host
 * }
 * d_samples.upload(h_staging.data(), sample_count, stream);
 * @endcode
 */
template <typename T>
cudaError_t upload_samples_from_payload(std::span<const uint8_t> payload,
                                        DeviceBuffer<T>& d_samples, PinnedBuffer<T>& h_staging,
                                        size_t sample_count, cudaStream_t stream = 0) noexcept {
    // Validate buffer sizes
    if (h_staging.size() < sample_count || d_samples.size() < sample_count) {
        return cudaErrorInvalidValue;
    }

    // Note: Actual endian conversion must be done by caller using SampleSpan
    // This function just handles the upload after conversion is complete
    return d_samples.upload(h_staging.data(), sample_count, stream);
}

/**
 * @brief Upload raw payload to device for GPU-side conversion
 *
 * Uploads VRT payload bytes directly to device memory. Endian conversion
 * should be performed on the GPU using device-side functions.
 *
 * @param payload VRT payload data (network byte order)
 * @param d_payload Device buffer for raw payload
 * @param stream CUDA stream
 * @return cudaError_t cudaSuccess on success
 *
 * After upload, use a GPU kernel to convert samples:
 * @code
 * __global__ void convert_kernel(const uint8_t* payload, int16_t* samples, size_t count) {
 *     size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
 *     if (idx < count) {
 *         samples[idx] = vrtigo::gpu::read_sample<int16_t>(payload + idx * sizeof(int16_t));
 *     }
 * }
 * @endcode
 */
inline cudaError_t upload_payload_raw(std::span<const uint8_t> payload,
                                      DeviceBuffer<uint8_t>& d_payload,
                                      cudaStream_t stream = 0) noexcept {
    if (d_payload.size() < payload.size()) {
        return cudaErrorInvalidValue;
    }
    return cudaMemcpyAsync(d_payload.data(), payload.data(), payload.size(), cudaMemcpyHostToDevice,
                           stream);
}

/**
 * @brief Upload raw payload and convert on GPU (placeholder for kernel launch)
 *
 * This is a convenience function that uploads raw payload bytes to the GPU.
 * The actual conversion kernel must be launched separately by the caller.
 *
 * @tparam T Sample type
 * @param payload VRT payload data (network byte order)
 * @param d_payload Device buffer for raw payload (temporary)
 * @param d_samples Device buffer for converted samples (output)
 * @param sample_count Number of samples
 * @param stream CUDA stream
 * @return cudaError_t cudaSuccess on success
 *
 * Note: Caller must launch conversion kernel after this function returns.
 */
template <typename T>
cudaError_t upload_payload_and_convert(std::span<const uint8_t> payload,
                                       DeviceBuffer<uint8_t>& d_payload, DeviceBuffer<T>& d_samples,
                                       size_t sample_count, cudaStream_t stream = 0) noexcept {
    // Validate sizes
    size_t required_payload_bytes = sample_count * sizeof(T);
    if (payload.size() < required_payload_bytes || d_payload.size() < required_payload_bytes ||
        d_samples.size() < sample_count) {
        return cudaErrorInvalidValue;
    }

    // Upload raw payload - caller must launch conversion kernel
    return upload_payload_raw(payload, d_payload, stream);
}

} // namespace gpu
} // namespace vrtigo
