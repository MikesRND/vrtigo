#pragma once

#include <span>

#include <cstring>
#include <vrtigo/class_id.hpp>
#include <vrtigo/field_tags.hpp>
#include <vrtigo/types.hpp>

#include "../detail/cif.hpp"
#include "../detail/field_access.hpp"
#include "../detail/field_mask.hpp"
#include "../detail/prologue.hpp"
#include "../detail/timestamp_traits.hpp"
#include "packet_base.hpp"

namespace vrtigo::typed {

// Base compile-time context packet template (low-level API)
// Creates packets with known structure at compile time using CIF bitmasks
// Variable-length fields are NOT supported in this template
//
// IMPORTANT: Per VITA 49.2 spec:
//   - Context packets ALWAYS have a Stream ID (no HasStreamId parameter)
//   - Context packets NEVER have a Trailer (bit 26 is reserved, must be 0)
//
// NOTE: Most users should use the field-based ContextPacket template instead,
// which automatically computes CIF bitmasks from field tags.
template <typename TimestampType = NoTimestamp, typename ClassIdType = NoClassId, uint32_t CIF0 = 0,
          uint32_t CIF1 = 0, uint32_t CIF2 = 0, uint32_t CIF3 = 0>
    requires vrtigo::ValidTimestampType<TimestampType> && vrtigo::ValidClassIdType<ClassIdType>
class ContextPacketBase
    : public detail::PacketBase<
          ContextPacketBase<TimestampType, ClassIdType, CIF0, CIF1, CIF2, CIF3>,
          vrtigo::Prologue<PacketType::context, ClassIdType, TimestampType, true>> {
private:
    using Base =
        detail::PacketBase<ContextPacketBase<TimestampType, ClassIdType, CIF0, CIF1, CIF2, CIF3>,
                           vrtigo::Prologue<PacketType::context, ClassIdType, TimestampType, true>>;

    friend Base;

    // Compute actual CIF0 with automatic CIF1/CIF2/CIF3 enable bits
    static constexpr uint32_t computed_cif0 = CIF0 |
                                              ((CIF1 != 0) ? (1U << cif::CIF1_ENABLE_BIT) : 0) |
                                              ((CIF2 != 0) ? (1U << cif::CIF2_ENABLE_BIT) : 0) |
                                              ((CIF3 != 0) ? (1U << cif::CIF3_ENABLE_BIT) : 0);

    // COMPLETE compile-time validation

    // User must NOT set CIF1/CIF2/CIF3 enable bits manually - they're auto-managed
    static_assert((CIF0 & cif::CIF_ENABLE_MASK) == 0,
                  "Do not set CIF1/CIF2/CIF3 enable bits (1,2,3) in CIF0 - they are auto-managed "
                  "based on CIF1/CIF2/CIF3 parameters");

    // Check CIF0 data fields: only supported non-variable bits allowed
    static_assert((CIF0 & ~cif::CIF0_COMPILETIME_SUPPORTED_MASK) == 0,
                  "CIF0 contains unsupported, reserved, or variable-length fields");

    // Check CIF1 if enabled: only supported bits allowed
    static_assert(CIF1 == 0 || (CIF1 & ~cif::CIF1_SUPPORTED_MASK) == 0,
                  "CIF1 contains unsupported or reserved fields");

    // Check CIF2 if enabled: only supported bits allowed
    static_assert(CIF2 == 0 || (CIF2 & ~cif::CIF2_SUPPORTED_MASK) == 0,
                  "CIF2 contains unsupported or reserved fields");

    // Check CIF3 if enabled: only supported bits allowed
    static_assert(CIF3 == 0 || (CIF3 & ~cif::CIF3_SUPPORTED_MASK) == 0,
                  "CIF3 contains unsupported or reserved fields");

    // Additional safety check: verify each set bit has non-zero size
    static constexpr bool validate_cif0_sizes() {
        for (int bit = 0; bit < 32; ++bit) {
            if (CIF0 & (1U << bit)) {
                // Skip control bits
                if (bit == 1 || bit == 2 || bit == 31)
                    continue;
                // Check for unsupported fields (size 0 means unsupported for fixed fields)
                if (!cif::CIF0_FIELDS[bit].is_variable && cif::CIF0_FIELDS[bit].size_words == 0 &&
                    cif::CIF0_FIELDS[bit].is_supported) {
                    return false;
                }
            }
        }
        return true;
    }

    static_assert(validate_cif0_sizes(), "CIF0 contains fields with undefined sizes");

    // Inherit prologue_type from base
    using prologue_type = typename Base::prologue_type;

    // CIF word count
    static constexpr size_t cif_words = 1 + ((CIF1 != 0) ? 1 : 0) + // CIF1
                                        ((CIF2 != 0) ? 1 : 0) +     // CIF2
                                        ((CIF3 != 0) ? 1 : 0);      // CIF3

    // Calculate context field size (compile-time, no variable fields)
    static constexpr size_t context_fields_words =
        cif::calculate_context_size_ct<CIF0, CIF1, CIF2, CIF3>();

    static constexpr size_t computed_size_words =
        prologue_type::size_words() + cif_words + context_fields_words;

    // Complete offset calculation for CIF words
    static constexpr size_t calculate_cif_offset() {
        // Use prologue's payload_offset which points right after all prologue fields
        return prologue_type::payload_offset * vrt_word_size; // Convert words to bytes
    }

    // Calculate offset to context fields (after CIF words)
    static constexpr size_t calculate_context_offset() {
        size_t offset = calculate_cif_offset();

        // Add CIF words
        offset += 4; // CIF0 always present
        if constexpr (CIF1 != 0) {
            offset += 4; // CIF1
        }
        if constexpr (CIF2 != 0) {
            offset += 4; // CIF2
        }
        if constexpr (CIF3 != 0) {
            offset += 4; // CIF3
        }

        return offset;
    }

public:
    // ========================================================================
    // Static constexpr property functions (packet-specific)
    // Inherited from base: has_stream_id(), has_class_id(), has_timestamp()
    // ========================================================================

    /// Get packet size in 32-bit words
    static constexpr size_t size_words() noexcept { return computed_size_words; }

    /// Get packet size in bytes
    static constexpr size_t size_bytes() noexcept { return size_words() * 4; }

    /// Check if packet has trailer (always false for context packets per spec)
    static constexpr bool has_trailer() noexcept { return false; }

    /// Get packet type
    static constexpr PacketType type() noexcept { return PacketType::context; }

    // CIF values for builder (with enable bits)
    static constexpr uint32_t cif0_value = computed_cif0;
    static constexpr uint32_t cif1_value = CIF1;
    static constexpr uint32_t cif2_value = CIF2;
    static constexpr uint32_t cif3_value = CIF3;

    explicit ContextPacketBase(std::span<uint8_t, size_bytes()> buffer, bool init = true) noexcept
        : Base(buffer.data()) {
        if (init) {
            init_header();
            init_stream_id();
            init_class_id();
            init_timestamps();
            write_cif_words();
        }
    }

    // ========================================================================
    // Inherited from typed::detail::PacketBase:
    //   - header(), header() const
    //   - packet_count(), set_packet_count()
    //   - stream_id(), set_stream_id()
    //   - class_id(), set_class_id() [requires has_class_id()]
    //   - timestamp(), set_timestamp() [requires has_timestamp()]
    //   - as_bytes(), as_bytes() const
    // ========================================================================

private:
    void init_header() noexcept {
        // Use prologue to initialize header (no trailer for context packets)
        this->prologue_.init_header(size_words(), 0, false);
    }

    void init_stream_id() noexcept {
        // Use prologue to initialize stream ID (always present per spec)
        this->prologue_.init_stream_id();
    }

    void init_class_id() noexcept {
        if constexpr (Base::has_class_id()) {
            this->prologue_.init_class_id();
        }
    }

    void init_timestamps() noexcept {
        if constexpr (Base::has_timestamp()) {
            this->prologue_.init_timestamps();
        }
    }

    void write_cif_words() noexcept {
        size_t offset = calculate_cif_offset();

        // Write CIF0 (with automatic CIF1/CIF2/CIF3 enable bits)
        cif::write_u32_safe(this->buffer_, offset, computed_cif0);
        offset += 4;

        // Write CIF1 if present
        if constexpr (CIF1 != 0) {
            cif::write_u32_safe(this->buffer_, offset, CIF1);
            offset += 4;
        }

        // Write CIF2 if present
        if constexpr (CIF2 != 0) {
            cif::write_u32_safe(this->buffer_, offset, CIF2);
            offset += 4;
        }

        // Write CIF3 if present
        if constexpr (CIF3 != 0) {
            cif::write_u32_safe(this->buffer_, offset, CIF3);
        }
    }

public:
    // Field access via subscript operator
    template <uint8_t CifWord, uint8_t Bit>
    auto operator[](field::field_tag_t<CifWord, Bit> tag) noexcept
        -> FieldProxy<field::field_tag_t<CifWord, Bit>, ContextPacketBase> {
        return vrtigo::detail::make_field_proxy(*this, tag);
    }

    template <uint8_t CifWord, uint8_t Bit>
    auto operator[](field::field_tag_t<CifWord, Bit> tag) const noexcept
        -> FieldProxy<field::field_tag_t<CifWord, Bit>, const ContextPacketBase> {
        return vrtigo::detail::make_field_proxy(*this, tag);
    }

    // Internal implementation details - DO NOT USE DIRECTLY
    // These methods are required by the field access implementation (CifPacketBase concept)
    // Users should access fields via operator[] (e.g., packet[bandwidth])
    const uint8_t* context_buffer() const noexcept { return this->buffer_; }
    uint8_t* mutable_context_buffer() noexcept { return this->buffer_; }
    static constexpr size_t context_base_offset() noexcept { return calculate_context_offset(); }
    static constexpr uint32_t cif0() noexcept { return computed_cif0; }
    static constexpr uint32_t cif1() noexcept { return CIF1; }
    static constexpr uint32_t cif2() noexcept { return CIF2; }
    static constexpr uint32_t cif3() noexcept { return CIF3; }
    static constexpr size_t buffer_size() noexcept { return size_bytes(); }

    /// Read the Context Field Change Indicator (CIF0 bit 31)
    /// Returns true if at least one context field has changed since last packet
    /// Returns false if all fields are unchanged from previous packets
    [[nodiscard]] bool change_indicator() const noexcept {
        const uint8_t* buf = context_buffer();
        size_t cif0_offset = calculate_cif_offset();
        uint32_t cif0_word = cif::read_u32_safe(buf, cif0_offset);
        return (cif0_word & (1U << 31)) != 0;
    }

    /// Set the Context Field Change Indicator (CIF0 bit 31)
    /// @param changed true = at least one field has new value
    ///                false = all fields unchanged from previous packets
    void set_change_indicator(bool changed) noexcept {
        uint8_t* buf = mutable_context_buffer();
        size_t cif0_offset = calculate_cif_offset();
        uint32_t cif0_word = cif::read_u32_safe(buf, cif0_offset);

        if (changed) {
            cif0_word |= (1U << 31); // Set bit 31
        } else {
            cif0_word &= ~(1U << 31); // Clear bit 31
        }

        cif::write_u32_safe(buf, cif0_offset, cif0_word);
    }
};

// Field-based ContextPacket template (user-friendly API)
// Automatically computes CIF bitmasks from field tags
//
// Example usage:
//     using namespace vrtigo::field;
//     using MyPacket = typed::ContextPacket<NoTimestamp, NoClassId,
//                                         bandwidth, sample_rate, gain>;
//
// This is the recommended API for most users.
template <typename TimestampType = NoTimestamp, typename ClassIdType = NoClassId, auto... Fields>
    requires vrtigo::ValidTimestampType<TimestampType> && vrtigo::ValidClassIdType<ClassIdType>
class ContextPacket
    : public ContextPacketBase<
          TimestampType, ClassIdType, vrtigo::detail::FieldMask<Fields...>::cif0,
          vrtigo::detail::FieldMask<Fields...>::cif1, vrtigo::detail::FieldMask<Fields...>::cif2,
          vrtigo::detail::FieldMask<Fields...>::cif3> {
public:
    using Base =
        ContextPacketBase<TimestampType, ClassIdType, vrtigo::detail::FieldMask<Fields...>::cif0,
                          vrtigo::detail::FieldMask<Fields...>::cif1,
                          vrtigo::detail::FieldMask<Fields...>::cif2,
                          vrtigo::detail::FieldMask<Fields...>::cif3>;

    // Inherit constructors
    using Base::Base;
};

} // namespace vrtigo::typed
