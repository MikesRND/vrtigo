#pragma once

/**
 * @file endian.hpp
 * @brief Device-compatible byte swap operations for GPU kernels
 *
 * This header provides endian conversion functions that work on both host and device.
 * On device, manual byte swap implementations are used since __builtin_bswap*
 * intrinsics are not available. On CUDA devices with __CUDA_ARCH__ >= 200,
 * __byte_perm() intrinsic is used for efficient 32-bit swaps.
 */

#include <cstdint>

#include <vrtigo/gpu/detail/cuda_macros.hpp>

namespace vrtigo {
namespace gpu {

// ============================================================================
// Byte swap implementations
// ============================================================================

/**
 * @brief Swap bytes in a 16-bit value
 *
 * Manual implementation for device compatibility.
 */
VRTIGO_HOST_DEVICE inline uint16_t byteswap16(uint16_t value) noexcept {
#if defined(__CUDA_ARCH__)
    // Manual swap on device
    return static_cast<uint16_t>((value >> 8) | (value << 8));
#else
    // Use builtin on host for potential optimization
    return __builtin_bswap16(value);
#endif
}

/**
 * @brief Swap bytes in a 32-bit value
 *
 * Uses __byte_perm() CUDA intrinsic when available for efficient hardware swap.
 */
VRTIGO_HOST_DEVICE inline uint32_t byteswap32(uint32_t value) noexcept {
#if defined(__CUDA_ARCH__)
    // __byte_perm(x, y, s) selects bytes from x and y based on selector s
    // Selector 0x0123 means: byte0=x[0], byte1=x[1], byte2=x[2], byte3=x[3]
    // Selector 0x3210 reverses the order
    return __byte_perm(value, 0, 0x0123);
#else
    // Use builtin on host
    return __builtin_bswap32(value);
#endif
}

/**
 * @brief Swap bytes in a 64-bit value
 *
 * Manual implementation for device compatibility.
 */
VRTIGO_HOST_DEVICE inline uint64_t byteswap64(uint64_t value) noexcept {
#if defined(__CUDA_ARCH__)
    // Split into two 32-bit halves, swap each, then swap halves
    uint32_t lo = static_cast<uint32_t>(value);
    uint32_t hi = static_cast<uint32_t>(value >> 32);
    lo = __byte_perm(lo, 0, 0x0123);
    hi = __byte_perm(hi, 0, 0x0123);
    return (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
#else
    // Use builtin on host
    return __builtin_bswap64(value);
#endif
}

// ============================================================================
// Platform endianness detection
// ============================================================================

// Note: CUDA devices are little-endian. Host detection uses compile-time check.
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
inline constexpr bool is_little_endian = (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__);
#else
// Fallback assumption: most modern systems are little-endian
inline constexpr bool is_little_endian = true;
#endif

inline constexpr bool is_big_endian = !is_little_endian;

// ============================================================================
// Network byte order conversion functions
// ============================================================================

/**
 * @brief Convert 16-bit value from host to network byte order (big-endian)
 */
VRTIGO_HOST_DEVICE inline uint16_t host_to_network16(uint16_t value) noexcept {
    // CUDA devices are always little-endian, so always swap on device
#if defined(__CUDA_ARCH__)
    return byteswap16(value);
#else
    if constexpr (is_little_endian) {
        return byteswap16(value);
    } else {
        return value;
    }
#endif
}

/**
 * @brief Convert 32-bit value from host to network byte order (big-endian)
 */
VRTIGO_HOST_DEVICE inline uint32_t host_to_network32(uint32_t value) noexcept {
#if defined(__CUDA_ARCH__)
    return byteswap32(value);
#else
    if constexpr (is_little_endian) {
        return byteswap32(value);
    } else {
        return value;
    }
#endif
}

/**
 * @brief Convert 64-bit value from host to network byte order (big-endian)
 */
VRTIGO_HOST_DEVICE inline uint64_t host_to_network64(uint64_t value) noexcept {
#if defined(__CUDA_ARCH__)
    return byteswap64(value);
#else
    if constexpr (is_little_endian) {
        return byteswap64(value);
    } else {
        return value;
    }
#endif
}

/**
 * @brief Convert 16-bit value from network byte order (big-endian) to host
 */
VRTIGO_HOST_DEVICE inline uint16_t network_to_host16(uint16_t value) noexcept {
    return host_to_network16(value); // Same operation (symmetric)
}

/**
 * @brief Convert 32-bit value from network byte order (big-endian) to host
 */
VRTIGO_HOST_DEVICE inline uint32_t network_to_host32(uint32_t value) noexcept {
    return host_to_network32(value); // Same operation (symmetric)
}

/**
 * @brief Convert 64-bit value from network byte order (big-endian) to host
 */
VRTIGO_HOST_DEVICE inline uint64_t network_to_host64(uint64_t value) noexcept {
    return host_to_network64(value); // Same operation (symmetric)
}

} // namespace gpu
} // namespace vrtigo
