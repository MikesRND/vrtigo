// Copyright (c) 2025 Michael Smith
// SPDX-License-Identifier: MIT

#pragma once

#include "vrtio/detail/packet_variant.hpp"

#include <concepts>

#include <cstddef>

namespace vrtio::utils::detail {

/**
 * @brief Concept for types that can write VRT packets
 *
 * A PacketWriter must support writing PacketVariant objects and
 * tracking the number of packets written.
 *
 * @tparam T The type to check
 */
template <typename T>
concept PacketWriter = requires(T& writer, const vrtio::PacketVariant& pv) {
    { writer.write_packet(pv) } -> std::same_as<bool>;
    { writer.packets_written() } -> std::same_as<size_t>;
};

/**
 * @brief Concept for writers that support explicit flushing
 *
 * FlushableWriter extends PacketWriter to require a flush() operation.
 * Typically used for buffered file writers where pending data must be
 * explicitly written to disk.
 *
 * @tparam T The type to check
 */
template <typename T>
concept FlushableWriter = PacketWriter<T> && requires(T& writer) {
    { writer.flush() } -> std::same_as<bool>;
};

} // namespace vrtio::utils::detail
