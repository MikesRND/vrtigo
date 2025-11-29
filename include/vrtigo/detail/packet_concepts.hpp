#pragma once

#include <concepts>
#include <optional>
#include <span>
#include <utility>

#include <cstddef>
#include <cstdint>
#include <vrtigo/types.hpp>

#include "packet_header_accessor.hpp"
#include "parse_result.hpp"
#include "trailer_view.hpp"

namespace vrtigo {

// ============================================================================
// Shared Read-Only Metadata (common CT/RT surface)
// ============================================================================

/**
 * Minimal shared interface for packet metadata access.
 *
 * Both compile-time and runtime packets satisfy this concept.
 * This is the common read-only surface for generic packet handling.
 */
template <typename T>
concept PacketMetadataLike = requires(const T& pkt) {
    { pkt.header() } -> std::same_as<HeaderView>;
    { pkt.packet_count() } -> std::same_as<uint8_t>;
    { pkt.type() } -> std::same_as<PacketType>;
    // Note: convertible_to because CT returns span<const uint8_t, N>, RT returns span<const
    // uint8_t>
    { pkt.as_bytes() } -> std::convertible_to<std::span<const uint8_t>>;
};

// ============================================================================
// Compile-Time Packet Concepts (mutable, static structure)
// ============================================================================

/**
 * Compile-time packets - mutable, static size known at compile time.
 *
 * These are packets where the structure is known at compile time through
 * template parameters (DataPacket<...>, ContextPacket<...>).
 *
 * Key characteristics:
 * - Static size functions available (T::size_bytes(), T::size_words())
 * - Static feature functions available (T::has_stream_id(), etc.)
 * - Mutable access (can modify packet contents)
 */
template <typename T>
concept CompileTimePacketLike = PacketMetadataLike<T> && requires(T& mut) {
    // Static size (callable without instance)
    { T::size_bytes() } -> std::same_as<size_t>;
    { T::size_words() } -> std::same_as<size_t>;

    // Static feature queries
    { T::has_stream_id() } -> std::same_as<bool>;
    { T::has_class_id() } -> std::same_as<bool>;
    { T::has_timestamp() } -> std::same_as<bool>;
    { T::has_trailer() } -> std::same_as<bool>;

    // Mutable operations
    { mut.set_packet_count(uint8_t{}) } -> std::same_as<void>;
    // Note: convertible_to because returns span<uint8_t, N> (fixed extent)
    { mut.as_bytes() } -> std::convertible_to<std::span<uint8_t>>;
};

/**
 * Data packets (compile-time) - has payload.
 *
 * Examples: DataPacket<...>
 */
template <typename T>
concept DataPacketLike =
    CompileTimePacketLike<T> && requires(T& mut, const T& pkt, const uint8_t* d, size_t s) {
        // Note: convertible_to because returns span<T, N> (fixed extent)
        { pkt.payload() } -> std::convertible_to<std::span<const uint8_t>>;
        { mut.payload() } -> std::convertible_to<std::span<uint8_t>>;
        { mut.set_payload(d, s) } -> std::same_as<void>;
    };

/**
 * Context packets (compile-time) - has CIF fields.
 *
 * Examples: ContextPacket<...>, ContextPacketBase<...>
 */
template <typename T>
concept ContextPacketLike = CompileTimePacketLike<T> && requires(const T& pkt) {
    { pkt.change_indicator() } -> std::same_as<bool>;
};

// ============================================================================
// Runtime Packet Concepts (read-only, parsed)
// ============================================================================

/**
 * Runtime packets - read-only, guaranteed valid.
 *
 * These are packets where the structure is determined at runtime by parsing
 * the header and CIF fields (dynamic::DataPacket, dynamic::ContextPacket).
 *
 * Key characteristics:
 * - Constructed only via static parse() method returning ParseResult<T>
 * - If you have a RuntimePacket, it's guaranteed to be valid
 * - Read-only access (cannot modify packet contents)
 */
template <typename T>
concept RuntimePacketLike =
    PacketMetadataLike<T> && requires(const T& pkt, std::span<const uint8_t> buf) {
        // Static parse method - distinguishes RT from CT
        { T::parse(buf) } -> std::same_as<ParseResult<T>>;

        // Instance-only size (determined at parse time)
        { pkt.size_bytes() } -> std::same_as<size_t>;
        { pkt.size_words() } -> std::same_as<size_t>;
        { pkt.buffer_size() } -> std::same_as<size_t>;

        // Instance feature queries (determined at parse time)
        { pkt.has_stream_id() } -> std::same_as<bool>;
        { pkt.has_class_id() } -> std::same_as<bool>;
        { pkt.has_timestamp() } -> std::same_as<bool>;
        { pkt.has_trailer() } -> std::same_as<bool>;
    };

/**
 * Runtime data packets - has payload.
 *
 * Examples: dynamic::DataPacket
 */
template <typename T>
concept RuntimeDataPacketLike = RuntimePacketLike<T> && requires(const T& pkt) {
    { pkt.payload() } -> std::same_as<std::span<const uint8_t>>;
    { pkt.payload_size_bytes() } -> std::same_as<size_t>;
};

/**
 * Runtime context packets - has CIF field access.
 *
 * Examples: dynamic::ContextPacket
 */
template <typename T>
concept RuntimeContextPacketLike = RuntimePacketLike<T> && requires(const T& pkt) {
    { pkt.change_indicator() } -> std::same_as<bool>;
};

// ============================================================================
// Helper Concepts for PacketBuilder
// ============================================================================

/**
 * Packet has stream ID field (for builder constraint)
 */
template <typename T>
concept HasStreamId = requires(T& pkt) {
    { pkt.set_stream_id(uint32_t{}) } -> std::same_as<void>;
    { std::as_const(pkt).stream_id() } -> std::same_as<uint32_t>;
};

template <typename T>
concept MutableTrailerProxy = requires(T proxy) {
    { proxy.raw() } -> std::same_as<uint32_t>;
    { proxy.set_raw(uint32_t{}) } -> std::same_as<void>;
};

template <typename T>
concept ConstTrailerProxy = requires(T proxy) {
    { proxy.raw() } -> std::same_as<uint32_t>;
};

/**
 * Packet has mutable trailer field (for builder constraint)
 *
 * Compile-time packets with trailer support satisfy this concept.
 * Non-const access returns MutableTrailerProxy, const access returns ConstTrailerProxy.
 */
template <typename T>
concept HasMutableTrailer = requires(T& pkt) {
    { pkt.trailer() } -> MutableTrailerProxy;
    { std::as_const(pkt).trailer() } -> ConstTrailerProxy;
};

/**
 * Packet has read-only trailer field
 *
 * Runtime packets with trailer support satisfy this concept.
 * Returns std::optional<TrailerView> (ConstTrailerProxy wrapped in optional).
 */
template <typename T>
concept HasConstTrailer = requires(const T& pkt) {
    { pkt.trailer() } -> std::same_as<std::optional<TrailerView>>;
};

/**
 * Packet has payload field (for builder constraint)
 */
template <typename T>
concept HasPayload = requires(T& pkt, const uint8_t* data, size_t size) {
    { pkt.set_payload(data, size) } -> std::same_as<void>;
    { std::as_const(pkt).payload() } -> std::convertible_to<std::span<const uint8_t>>;
};

/**
 * Packet has packet count field (for builder constraint)
 */
template <typename T>
concept HasPacketCount = requires(T& pkt) {
    { pkt.set_packet_count(uint8_t{}) } -> std::same_as<void>;
    { std::as_const(pkt).packet_count() } -> std::same_as<uint8_t>;
};

// ============================================================================
// Legacy Aliases (deprecated, will be removed)
// ============================================================================

// PacketLike is now PacketMetadataLike
template <typename T>
concept PacketLike = PacketMetadataLike<T>;

// AnyPacket is redundant with PacketMetadataLike
template <typename T>
concept AnyPacket = PacketMetadataLike<T>;

} // namespace vrtigo
