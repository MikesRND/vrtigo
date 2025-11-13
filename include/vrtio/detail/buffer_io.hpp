#pragma once

#include <cstdint>
#include <cstring>

#include "endian.hpp"

namespace vrtio::detail {

/**
 * @brief Endian-safe buffer read/write helpers
 *
 * These functions provide a single source of truth for reading and writing
 * 32-bit and 64-bit values from/to packet buffers with proper network byte
 * order conversion.
 *
 * All functions use std::memcpy for alignment safety and compiler optimization.
 */

/**
 * Read a 32-bit value from buffer with network-to-host conversion
 * @param buffer Pointer to buffer
 * @param offset Byte offset into buffer
 * @return Value in host byte order
 */
inline uint32_t read_u32(const uint8_t* buffer, size_t offset) noexcept {
    uint32_t value;
    std::memcpy(&value, buffer + offset, sizeof(value));
    return network_to_host32(value);
}

/**
 * Read a 64-bit value from buffer with network-to-host conversion
 * @param buffer Pointer to buffer
 * @param offset Byte offset into buffer
 * @return Value in host byte order
 */
inline uint64_t read_u64(const uint8_t* buffer, size_t offset) noexcept {
    uint64_t value;
    std::memcpy(&value, buffer + offset, sizeof(value));
    return network_to_host64(value);
}

/**
 * Write a 32-bit value to buffer with host-to-network conversion
 * @param buffer Pointer to buffer
 * @param offset Byte offset into buffer
 * @param value Value in host byte order
 */
inline void write_u32(uint8_t* buffer, size_t offset, uint32_t value) noexcept {
    value = host_to_network32(value);
    std::memcpy(buffer + offset, &value, sizeof(value));
}

/**
 * Write a 64-bit value to buffer with host-to-network conversion
 * @param buffer Pointer to buffer
 * @param offset Byte offset into buffer
 * @param value Value in host byte order
 */
inline void write_u64(uint8_t* buffer, size_t offset, uint64_t value) noexcept {
    value = host_to_network64(value);
    std::memcpy(buffer + offset, &value, sizeof(value));
}

} // namespace vrtio::detail
