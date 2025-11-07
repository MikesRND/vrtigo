#pragma once

#include "types.hpp"
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace vrtio {

// Concept for types that can provide mutable buffer access
template<typename T>
concept MutableBuffer = requires(T t) {
    { t.data() } -> std::convertible_to<uint8_t*>;
    { t.size() } -> std::convertible_to<size_t>;
};

// Concept for types that can provide const buffer access
template<typename T>
concept ConstBuffer = requires(T t) {
    { t.data() } -> std::convertible_to<const uint8_t*>;
    { t.size() } -> std::convertible_to<size_t>;
};

// Concept for any buffer (mutable or const)
template<typename T>
concept AnyBuffer = MutableBuffer<T> || ConstBuffer<T>;

// Concept for valid payload size (must fit in packet with overhead)
// Max packet size is 65535 words (16-bit field), minus maximum overhead:
//   1 (header) + 1 (stream_id) + 1 (TSI) + 2 (TSF) + 1 (trailer) = 6 words
template<size_t N>
concept ValidPayloadWords = (N <= 65529);  // 65535 - 6

// Concept for valid packet type in Phase 1 (Signal Data only)
template<auto Type>
concept SignalDataPacket =
    std::same_as<decltype(Type), const packet_type> &&
    (Type == packet_type::signal_data_no_stream ||
     Type == packet_type::signal_data_with_stream);

}  // namespace vrtio
