#pragma once

#include "fields.hpp"
#include "detail/field_traits.hpp"
#include "detail/variable_field_dispatch.hpp"
#include "cif.hpp"
#include <optional>
#include <type_traits>

namespace vrtio {

namespace detail {

/// Concept: Types that provide packet buffer access
template<typename T>
concept PacketLike = requires(const T& pkt) {
    { pkt.context_buffer() } -> std::same_as<const uint8_t*>;
    { pkt.context_base_offset() } -> std::same_as<size_t>;
    { pkt.cif0() } -> std::same_as<uint32_t>;
    { pkt.cif1() } -> std::same_as<uint32_t>;
    { pkt.cif2() } -> std::same_as<uint32_t>;
};

/// Concept: Mutable packets (for write operations)
template<typename T>
concept MutablePacketLike = PacketLike<T> && requires(T& pkt) {
    { pkt.mutable_context_buffer() } -> std::same_as<uint8_t*>;
};

/// Concept: Compile-time packets with static CIF values (for zero-overhead field access)
template<typename T>
concept CompileTimePacket = PacketLike<T> && requires {
    { T::cif0_value } -> std::convertible_to<uint32_t>;
    { T::cif1_value } -> std::convertible_to<uint32_t>;
    { T::cif2_value } -> std::convertible_to<uint32_t>;
};

/// Check if a field is present in the packet's CIF words
template<uint8_t CifWord, uint8_t Bit>
constexpr bool is_field_present(uint32_t cif0, uint32_t cif1, uint32_t cif2) noexcept {
    if constexpr (CifWord == 0) {
        return (cif0 & (1U << Bit)) != 0;
    } else if constexpr (CifWord == 1) {
        return (cif1 & (1U << Bit)) != 0;
    } else if constexpr (CifWord == 2) {
        return (cif2 & (1U << Bit)) != 0;
    }
    return false;
}

} // namespace detail

// ============================================================================
// Public API: get() - Read field value from packet
// ============================================================================

/// Read a field value from a context packet
/// Returns std::nullopt if field is not present in packet
/// @tparam Tag Field tag type (e.g., field_tag_t<0, 29>)
/// @param packet Context packet (ContextPacket or ContextPacketView)
/// @param tag Field tag (e.g., field::bandwidth)
/// @return Field value or std::nullopt if not present
template<typename Packet, uint8_t CifWord, uint8_t Bit>
    requires detail::PacketLike<Packet>
auto get(const Packet& packet, field::field_tag_t<CifWord, Bit>) noexcept
    -> std::optional<typename detail::FieldTraits<CifWord, Bit>::value_type>
{
    using Trait = detail::FieldTraits<CifWord, Bit>;

    // Check if field is present
    if (!detail::is_field_present<CifWord, Bit>(
            packet.cif0(), packet.cif1(), packet.cif2())) {
        return std::nullopt;
    }

    // Use compile-time offset calculation for ContextPacket (zero overhead)
    // or runtime calculation for ContextPacketView
    size_t field_offset;
    if constexpr (detail::CompileTimePacket<Packet>) {
        // Compile-time packet: fold offset to constant at compile time
        constexpr size_t ct_offset = cif::calculate_field_offset_ct<
            Packet::cif0_value, Packet::cif1_value, Packet::cif2_value,
            CifWord, Bit>();
        field_offset = packet.context_base_offset() + ct_offset;
    } else {
        // Runtime packet: calculate offset dynamically
        field_offset = detail::calculate_field_offset_runtime(
            packet.cif0(), packet.cif1(), packet.cif2(),
            CifWord, Bit,
            packet.context_buffer(),
            packet.context_base_offset(),
            packet.buffer_size()
        );

        // Check for error sentinel (bounds check failed)
        if (field_offset == SIZE_MAX) {
            return std::nullopt;
        }
    }

    // Read field using trait
    return Trait::read(packet.context_buffer(), field_offset);
}

// ============================================================================
// Public API: set() - Write field value to mutable packet
// ============================================================================

/// Write a field value to a mutable context packet
/// Field must be present in packet (determined at compile-time for ContextPacket)
/// For variable-length fields, this will fail at compile-time (write not supported)
/// @tparam Tag Field tag type
/// @param packet Mutable context packet
/// @param tag Field tag
/// @param value Value to write
/// @return true if write succeeded, false if field not present or not writable
template<typename Packet, uint8_t CifWord, uint8_t Bit>
    requires detail::MutablePacketLike<Packet>
bool set(Packet& packet,
         field::field_tag_t<CifWord, Bit>,
         const typename detail::FieldTraits<CifWord, Bit>::value_type& value) noexcept
{
    using Trait = detail::FieldTraits<CifWord, Bit>;

    // Check if field is present
    if (!detail::is_field_present<CifWord, Bit>(
            packet.cif0(), packet.cif1(), packet.cif2())) {
        return false;
    }

    // Variable-length fields don't support write (no write() method in trait)
    if constexpr (!detail::FixedFieldTrait<Trait>) {
        return false;  // Cannot write variable-length fields
    } else {
        // Calculate field offset (compile-time for ContextPacket, runtime for ContextPacketView)
        size_t field_offset;
        if constexpr (detail::CompileTimePacket<Packet>) {
            // Compile-time packet: fold offset to constant at compile time
            constexpr size_t ct_offset = cif::calculate_field_offset_ct<
                Packet::cif0_value, Packet::cif1_value, Packet::cif2_value,
                CifWord, Bit>();
            field_offset = packet.context_base_offset() + ct_offset;
        } else {
            // Runtime packet: calculate offset dynamically
            field_offset = detail::calculate_field_offset_runtime(
                packet.cif0(), packet.cif1(), packet.cif2(),
                CifWord, Bit,
                packet.mutable_context_buffer(),
                packet.context_base_offset(),
                packet.buffer_size()
            );

            // Check for error sentinel (bounds check failed)
            if (field_offset == SIZE_MAX) {
                return false;
            }
        }

        // Write field using trait
        Trait::write(packet.mutable_context_buffer(), field_offset, value);
        return true;
    }
}

// ============================================================================
// Convenience: has() - Check if field is present
// ============================================================================

/// Check if a field is present in the packet
/// @tparam Tag Field tag type
/// @param packet Context packet
/// @param tag Field tag
/// @return true if field is present
template<typename Packet, uint8_t CifWord, uint8_t Bit>
    requires detail::PacketLike<Packet>
constexpr bool has(const Packet& packet, field::field_tag_t<CifWord, Bit>) noexcept {
    return detail::is_field_present<CifWord, Bit>(
        packet.cif0(), packet.cif1(), packet.cif2()
    );
}

// ============================================================================
// Convenience: get_unchecked() - Read without presence check (faster)
// ============================================================================

/// Read a field value without checking presence (undefined behavior if not present)
/// Use this only if you've already verified presence with has() or other means
/// IMPORTANT: Only available for compile-time packets (ContextPacket<>).
/// For runtime packets (ContextPacketView), use the checked get() API instead.
/// @tparam Tag Field tag type
/// @param packet Compile-time context packet
/// @param tag Field tag
/// @return Field value (undefined if field not present)
template<typename Packet, uint8_t CifWord, uint8_t Bit>
    requires detail::CompileTimePacket<Packet>
auto get_unchecked(const Packet& packet, field::field_tag_t<CifWord, Bit>) noexcept
    -> typename detail::FieldTraits<CifWord, Bit>::value_type
{
    using Trait = detail::FieldTraits<CifWord, Bit>;

    // Compile-time packet: fold offset to constant at compile time (zero overhead)
    constexpr size_t ct_offset = cif::calculate_field_offset_ct<
        Packet::cif0_value, Packet::cif1_value, Packet::cif2_value,
        CifWord, Bit>();
    size_t field_offset = packet.context_base_offset() + ct_offset;

    return Trait::read(packet.context_buffer(), field_offset);
}

} // namespace vrtio
