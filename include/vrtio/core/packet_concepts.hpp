#pragma once

#include "types.hpp"
#include <concepts>
#include <span>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace vrtio {

/**
 * Base concept - all packet types must satisfy this
 *
 * Provides minimum interface for any packet-like object:
 * - Access to buffer via as_bytes() or context_buffer()
 * - Packet size information
 *
 * This specifically excludes std::vector and other container types.
 */
template<typename T>
concept PacketBase = (
    requires(const T& pkt) {
        { pkt.as_bytes() } -> std::convertible_to<std::span<const uint8_t>>;
        // Require packet-specific size method (not container .size())
        { pkt.packet_size_bytes() } -> std::convertible_to<size_t>;
    } ||
    requires(const T& pkt) {
        { pkt.as_bytes() } -> std::convertible_to<std::span<const uint8_t>>;
        // Or compile-time static size
        { T::size_bytes } -> std::convertible_to<size_t>;
    } ||
    requires(const T& pkt) {
        { pkt.context_buffer() } -> std::convertible_to<const uint8_t*>;
        { pkt.packet_size_bytes() } -> std::convertible_to<size_t>;
    }
);

/**
 * Fixed-structure compile-time packets (Signal, ExtData)
 *
 * Characteristics:
 * - Packet structure known at compile-time
 * - Fixed payload size (template parameter)
 * - No variable-length fields
 * - Zero-copy mutable view over user buffer
 * - Direct field accessors (stream_id(), getTimeStamp(), etc.)
 *
 * Examples: SignalPacket<...>
 */
template<typename T>
concept FixedPacketLike = PacketBase<T> && requires(const T& pkt, T& mut_pkt) {
    // Compile-time size
    { T::size_bytes } -> std::convertible_to<size_t>;

    // Validation
    { pkt.validate(size_t{}) } -> std::same_as<validation_error>;

    // Payload access (fixed-size span)
    { pkt.payload() } -> std::convertible_to<std::span<const uint8_t>>;
    { mut_pkt.payload() } -> std::convertible_to<std::span<uint8_t>>;

    // Mutable buffer access
    { mut_pkt.as_bytes() } -> std::convertible_to<std::span<uint8_t>>;
};

/**
 * Fixed-structure runtime parsers (SignalPacketView, ExtDataPacketView)
 *
 * Characteristics:
 * - Type-erased runtime parsing
 * - Automatically validates on construction
 * - Const-only (read-only) view
 * - Size calculated from header at runtime
 * - Returns std::optional for optional fields
 *
 * Examples: SignalPacketView
 */
template<typename T>
concept FixedPacketViewLike = PacketBase<T> && requires(const T& view) {
    // Validation status
    { view.is_valid() } -> std::same_as<bool>;
    { view.error() } -> std::same_as<validation_error>;

    // Runtime size
    { view.packet_size_bytes() } -> std::convertible_to<size_t>;
    { view.packet_size_words() } -> std::convertible_to<size_t>;

    // Payload access
    { view.payload() } -> std::convertible_to<std::span<const uint8_t>>;
    { view.payload_size_bytes() } -> std::convertible_to<size_t>;

    // Buffer access
    { view.buffer_size() } -> std::convertible_to<size_t>;
};

/**
 * Variable-structure compile-time packets (Context, Command)
 *
 * Characteristics:
 * - Packet structure defined by CIF (Context Indicator Fields)
 * - May contain variable-length fields
 * - Zero-copy mutable view over user buffer
 * - Field access via generic API (get/set/has in field_access.hpp)
 * - Size calculation based on CIF template parameters
 *
 * Examples: ContextPacket<...>
 */
template<typename T>
concept VariablePacketLike = PacketBase<T> && requires(const T& pkt, T& mut_pkt) {
    // Compile-time size
    { T::size_bytes } -> std::convertible_to<size_t>;

    // CIF access (compile-time constants)
    { pkt.cif0() } -> std::same_as<uint32_t>;

    // Buffer and offset access for field_access API
    { pkt.context_buffer() } -> std::same_as<const uint8_t*>;
    { mut_pkt.mutable_context_buffer() } -> std::convertible_to<uint8_t*>;
    { pkt.context_base_offset() } -> std::convertible_to<size_t>;

    // Mutable buffer access
    { mut_pkt.as_bytes() } -> std::convertible_to<std::span<uint8_t>>;

    // Buffer size
    { pkt.buffer_size() } -> std::convertible_to<size_t>;
};

/**
 * Variable-structure runtime parsers (ContextPacketView, CommandPacketView)
 *
 * Characteristics:
 * - Type-erased runtime parsing
 * - Automatically validates on construction
 * - Const-only (read-only) view
 * - Handles variable-length fields (GPS ASCII, Context Association)
 * - Size calculated from CIF at runtime
 * - Field access via generic API
 *
 * Examples: ContextPacketView
 */
template<typename T>
concept VariablePacketViewLike = PacketBase<T> && requires(const T& view) {
    // Validation status
    { view.is_valid() } -> std::same_as<bool>;

    // CIF access (runtime values)
    { view.cif0() } -> std::same_as<uint32_t>;

    // Buffer and offset access for field_access API
    { view.context_buffer() } -> std::same_as<const uint8_t*>;
    { view.context_base_offset() } -> std::convertible_to<size_t>;

    // Runtime size
    { view.packet_size_bytes() } -> std::convertible_to<size_t>;
    { view.packet_size_words() } -> std::convertible_to<size_t>;

    // Buffer access
    { view.buffer_size() } -> std::convertible_to<size_t>;
};

/**
 * Compile-time packet (any type - fixed or variable structure)
 *
 * Union of FixedPacketLike and VariablePacketLike concepts.
 * Use this when you need to accept any compile-time packet type.
 */
template<typename T>
concept CompileTimePacket = FixedPacketLike<T> || VariablePacketLike<T>;

/**
 * Runtime packet parser (any type - fixed or variable structure)
 *
 * Union of FixedPacketViewLike and VariablePacketViewLike concepts.
 * Use this when you need to accept any runtime parser type.
 */
template<typename T>
concept RuntimePacketView = FixedPacketViewLike<T> || VariablePacketViewLike<T>;

/**
 * Any packet or view (compile-time or runtime)
 *
 * Use this when you need to accept any packet-like object.
 */
template<typename T>
concept AnyPacketLike = CompileTimePacket<T> || RuntimePacketView<T>;

// ========================================================================
// Helper concepts for PacketBuilder
// ========================================================================

/**
 * Packet has stream ID field
 */
template<typename T>
concept HasStreamId = requires(T& pkt) {
    { pkt.set_stream_id(uint32_t{}) } -> std::same_as<void>;
    { std::as_const(pkt).stream_id() } -> std::same_as<uint32_t>;
};

template<typename T>
concept MutableTrailerProxy = requires(T proxy) {
    { proxy.raw() } -> std::same_as<uint32_t>;
    { proxy.set_raw(uint32_t{}) } -> std::same_as<void>;
};

template<typename T>
concept ConstTrailerProxy = requires(T proxy) {
    { proxy.raw() } -> std::same_as<uint32_t>;
};

/**
 * Packet has trailer field
 */
template<typename T>
concept HasTrailer = requires(T& pkt) {
    { pkt.trailer() } -> MutableTrailerProxy;
    { std::as_const(pkt).trailer() } -> ConstTrailerProxy;
};

/**
 * Packet has payload field
 */
template<typename T>
concept HasPayload = requires(T& pkt, const uint8_t* data, size_t size) {
    { pkt.set_payload(data, size) } -> std::same_as<void>;
    { std::as_const(pkt).payload() } -> std::convertible_to<std::span<const uint8_t>>;
};

/**
 * Packet has packet count field
 */
template<typename T>
concept HasPacketCount = requires(T& pkt) {
    { pkt.set_packet_count(uint8_t{}) } -> std::same_as<void>;
    { std::as_const(pkt).packet_count() } -> std::same_as<uint8_t>;
};

}  // namespace vrtio
