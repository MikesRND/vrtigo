#pragma once

#include <concepts>
#include <span>
#include <type_traits>

#include <cstddef>
#include <cstdint>

#include "types.hpp"

namespace vrtio {

// Concept for types that can provide mutable buffer access
// Works with standard containers (std::vector, std::array) and custom buffer types
template <typename T>
concept MutableBuffer = requires(T t) {
    { t.data() } -> std::convertible_to<uint8_t*>;
    { t.size() } -> std::convertible_to<size_t>;
};

// Concept for types that can provide const buffer access
// Works with standard containers (std::vector, std::array) and custom buffer types
template <typename T>
concept ConstBuffer = requires(const T t) {
    { t.data() } -> std::convertible_to<const uint8_t*>;
    { t.size() } -> std::convertible_to<size_t>;
};

// Concept for any buffer (mutable or const)
template <typename T>
concept AnyBuffer = MutableBuffer<T> || ConstBuffer<T>;

// Concept for valid payload size (must fit in packet with overhead)
// Max packet size is 65535 words (16-bit field), minus maximum overhead:
//   1 (header) + 1 (stream_id) + 1 (TSI) + 2 (TSF) + 1 (trailer) = 6 words
template <size_t N>
concept ValidPayloadWords = (N <= 65529); // 65535 - 6

// Concept for valid data packet types (Signal Data and Extension Data)
// Accepts packet types 0-3 (both Signal and Extension variants, with and without stream ID)
template <auto Type>
concept DataPacketType =
    std::same_as<decltype(Type), const PacketType> &&
    (Type == PacketType::SignalDataNoId || Type == PacketType::SignalData ||
     Type == PacketType::ExtensionDataNoId || Type == PacketType::ExtensionData);

} // namespace vrtio
