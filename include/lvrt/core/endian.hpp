#pragma once

#include <cstdint>
#include <bit>

namespace vrtio {
namespace detail {

// Platform endianness detection
inline constexpr bool is_little_endian = (std::endian::native == std::endian::little);
inline constexpr bool is_big_endian = (std::endian::native == std::endian::big);

static_assert(is_little_endian || is_big_endian,
              "Mixed endianness not supported");

// Byte swap operations (constexpr for compile-time use)
constexpr uint16_t byteswap16(uint16_t value) noexcept {
    return __builtin_bswap16(value);
}

constexpr uint32_t byteswap32(uint32_t value) noexcept {
    return __builtin_bswap32(value);
}

constexpr uint64_t byteswap64(uint64_t value) noexcept {
    return __builtin_bswap64(value);
}

// Convert to network byte order (big-endian)
constexpr uint16_t host_to_network16(uint16_t value) noexcept {
    if constexpr (is_little_endian) {
        return byteswap16(value);
    } else {
        return value;
    }
}

constexpr uint32_t host_to_network32(uint32_t value) noexcept {
    if constexpr (is_little_endian) {
        return byteswap32(value);
    } else {
        return value;
    }
}

constexpr uint64_t host_to_network64(uint64_t value) noexcept {
    if constexpr (is_little_endian) {
        return byteswap64(value);
    } else {
        return value;
    }
}

// Convert from network byte order (big-endian) to host
constexpr uint16_t network_to_host16(uint16_t value) noexcept {
    return host_to_network16(value);  // Same operation
}

constexpr uint32_t network_to_host32(uint32_t value) noexcept {
    return host_to_network32(value);  // Same operation
}

constexpr uint64_t network_to_host64(uint64_t value) noexcept {
    return host_to_network64(value);  // Same operation
}

}  // namespace detail
}  // namespace vrtio
