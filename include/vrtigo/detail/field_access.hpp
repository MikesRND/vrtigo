#pragma once

#include <optional>
#include <type_traits>

#include <vrtigo/field_tags.hpp>

#include "cif.hpp"
#include "field_proxy.hpp"
#include "field_traits.hpp"
#include "variable_field_dispatch.hpp"

namespace vrtigo {

namespace detail {

/// Concept: CIF-enabled packets that provide context buffer and CIF word access
/// Note: This is distinct from the global CompileTimePacket concept in packet_concepts.hpp
/// This concept is specifically for CIF field operations on context packets
template <typename T>
concept CifPacketBase = requires(const T& pkt) {
    { pkt.context_buffer() } -> std::same_as<const uint8_t*>;
    { pkt.context_base_offset() } -> std::same_as<size_t>;
    { pkt.cif0() } -> std::same_as<uint32_t>;
    { pkt.cif1() } -> std::same_as<uint32_t>;
    { pkt.cif2() } -> std::same_as<uint32_t>;
    { pkt.cif3() } -> std::same_as<uint32_t>;
};

/// Concept: Mutable CIF packets (for write operations on CIF fields)
template <typename T>
concept MutableCifPacket = CifPacketBase<T> && requires(T& pkt) {
    { pkt.mutable_context_buffer() } -> std::same_as<uint8_t*>;
};

/// Concept: Compile-time CIF packets with static CIF values (for zero-overhead field access)
/// Note: This is distinct from the global CompileTimePacket concept in packet_concepts.hpp
/// This concept specifically requires static CIF values for compile-time field optimization
template <typename T>
concept CifPacketLike = CifPacketBase<T> && requires {
    { T::cif0_value } -> std::convertible_to<uint32_t>;
    { T::cif1_value } -> std::convertible_to<uint32_t>;
    { T::cif2_value } -> std::convertible_to<uint32_t>;
    { T::cif3_value } -> std::convertible_to<uint32_t>;
};

/// Check if a field is present in the packet's CIF words
template <uint8_t CifWord, uint8_t Bit>
constexpr bool is_field_present(uint32_t cif0, uint32_t cif1, uint32_t cif2,
                                uint32_t cif3) noexcept {
    if constexpr (CifWord == 0) {
        return (cif0 & (1U << Bit)) != 0;
    } else if constexpr (CifWord == 1) {
        return (cif1 & (1U << Bit)) != 0;
    } else if constexpr (CifWord == 2) {
        return (cif2 & (1U << Bit)) != 0;
    } else if constexpr (CifWord == 3) {
        return (cif3 & (1U << Bit)) != 0;
    }
    return false;
}

/// Internal implementation of field access (used by both free functions and operator[])
template <typename Packet, uint8_t CifWord, uint8_t Bit>
    requires CifPacketBase<Packet>
auto make_field_proxy(Packet& packet, field::field_tag_t<CifWord, Bit>) noexcept
    -> FieldProxy<field::field_tag_t<CifWord, Bit>, Packet> {
    using Trait = FieldTraits<CifWord, Bit>;
    using Tag = field::field_tag_t<CifWord, Bit>;

    // Check if field is present
    bool present =
        is_field_present<CifWord, Bit>(packet.cif0(), packet.cif1(), packet.cif2(), packet.cif3());

    if (!present) {
        // Return proxy with present=false
        return FieldProxy<Tag, Packet>(packet, 0, 0, false);
    }

    // Calculate field offset (compile-time or runtime)
    size_t field_offset;
    if constexpr (CifPacketLike<Packet>) {
        // Compile-time packet: fold offset to constant at compile time
        constexpr size_t ct_offset =
            cif::calculate_field_offset_ct<Packet::cif0_value, Packet::cif1_value,
                                           Packet::cif2_value, Packet::cif3_value, CifWord, Bit>();
        field_offset = packet.context_base_offset() + ct_offset;
    } else {
        // Runtime packet: calculate offset dynamically
        field_offset = calculate_field_offset_runtime(
            packet.cif0(), packet.cif1(), packet.cif2(), packet.cif3(), CifWord, Bit,
            packet.context_buffer(), packet.context_base_offset(), packet.buffer_size());

        // Check for error sentinel (bounds check failed)
        if (field_offset == SIZE_MAX) {
            return FieldProxy<Tag, Packet>(packet, 0, 0, false);
        }
    }

    // Calculate field size in bytes
    size_t field_size_bytes;

    // Determine which CIF field table to use
    constexpr const cif::FieldInfo* field_table = CifWord == 0   ? cif::CIF0_FIELDS
                                                  : CifWord == 1 ? cif::CIF1_FIELDS
                                                  : CifWord == 2 ? cif::CIF2_FIELDS
                                                                 : cif::CIF3_FIELDS;

    // Check if field is variable-length (needs runtime size calculation)
    if constexpr (VariableFieldTrait<Trait>) {
        // Variable-length field - compute size from buffer
        size_t size_words = Trait::compute_size_words(packet.context_buffer(), field_offset);
        field_size_bytes = size_words * 4;
    } else {
        // Fixed-length field - use size from CIF field table
        field_size_bytes = field_table[Bit].size_words * 4;
    }

    // Return proxy with cached offset, size, and presence
    return FieldProxy<Tag, Packet>(packet, field_offset, field_size_bytes, true);
}

} // namespace detail

// ============================================================================
// Convenience: make_field_proxy_unchecked() - Read without presence check (faster)
// ============================================================================

/// Read a field value without checking presence (undefined behavior if not present)
/// Use this only if you've already verified presence with has() or other means
/// IMPORTANT: Only available for compile-time packets (ContextPacket<>).
/// For runtime packets (dynamic::ContextPacketView), use the checked get() API instead.
/// @tparam Tag Field tag type
/// @param packet Compile-time context packet
/// @param tag Field tag
/// @return Field value (undefined if field not present)
template <typename Packet, uint8_t CifWord, uint8_t Bit>
    requires detail::CifPacketLike<Packet>
auto make_field_proxy_unchecked(const Packet& packet, field::field_tag_t<CifWord, Bit>) noexcept ->
    typename detail::FieldTraits<CifWord, Bit>::value_type {
    using Trait = detail::FieldTraits<CifWord, Bit>;

    // Compile-time packet: fold offset to constant at compile time (zero overhead)
    constexpr size_t ct_offset =
        cif::calculate_field_offset_ct<Packet::cif0_value, Packet::cif1_value, Packet::cif2_value,
                                       Packet::cif3_value, CifWord, Bit>();
    size_t field_offset = packet.context_base_offset() + ct_offset;

    return Trait::read(packet.context_buffer(), field_offset);
}

} // namespace vrtigo
