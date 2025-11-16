#pragma once

#include <span>

#include <cstring>
#include <vrtio/class_id.hpp>
#include <vrtio/field_tags.hpp>
#include <vrtio/types.hpp>

#include "cif.hpp"
#include "endian.hpp"
#include "field_access.hpp"
#include "field_mask.hpp"
#include "header_decode.hpp"
#include "header_init.hpp"
#include "packet_header_accessor.hpp"
#include "prologue.hpp"
#include "timestamp_traits.hpp"

namespace vrtio {

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
template <typename TimeStampType = NoTimeStamp, typename ClassIdType = NoClassId, uint32_t CIF0 = 0,
          uint32_t CIF1 = 0, uint32_t CIF2 = 0, uint32_t CIF3 = 0>
    requires ValidTimestampType<TimeStampType> && ValidClassIdType<ClassIdType>
class ContextPacketBase {
private:
    uint8_t* buffer_;
    // Use Prologue for common header fields (IsContext = true for always-present stream ID)
    using prologue_type = Prologue<PacketType::context, ClassIdType, TimeStampType, true>;
    mutable prologue_type prologue_;

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

    // CIF word count
    static constexpr size_t cif_words = 1 + ((CIF1 != 0) ? 1 : 0) + // CIF1
                                        ((CIF2 != 0) ? 1 : 0) +     // CIF2
                                        ((CIF3 != 0) ? 1 : 0);      // CIF3

    // Calculate context field size (compile-time, no variable fields)
    static constexpr size_t context_fields_words =
        cif::calculate_context_size_ct<CIF0, CIF1, CIF2, CIF3>();

    static constexpr size_t computed_size_words =
        prologue_type::size_words + cif_words + context_fields_words;

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
    // Packet size configuration
    static constexpr size_t size_words = computed_size_words;
    static constexpr size_t size_bytes = size_words * 4;
    static constexpr uint32_t cif0_value = computed_cif0; // For builder (with enable bits)
    static constexpr uint32_t cif1_value = CIF1;
    static constexpr uint32_t cif2_value = CIF2;
    static constexpr uint32_t cif3_value = CIF3;

    // Packet component presence (per VITA 49.2 spec)
    static constexpr bool has_stream_id = true; // Always present per spec
    static constexpr bool has_class_id = prologue_type::has_class_id;
    static constexpr bool has_timestamp = prologue_type::has_timestamp;
    static constexpr bool has_timestamp_integer = (prologue_type::tsi != TsiType::none);
    static constexpr bool has_timestamp_fractional = (prologue_type::tsf != TsfType::none);
    static constexpr bool has_trailer = false; // Always false per spec (bit 26 reserved)

    explicit ContextPacketBase(uint8_t* buffer, bool init = true) noexcept
        : buffer_(buffer),
          prologue_(buffer) {
        if (init) {
            init_header();
            init_stream_id();
            init_class_id();
            init_timestamps();
            write_cif_words();
        }
    }

    // Buffer access
    std::span<uint8_t, size_bytes> as_bytes() noexcept {
        return std::span<uint8_t, size_bytes>(buffer_, size_bytes);
    }

    std::span<const uint8_t, size_bytes> as_bytes() const noexcept {
        return std::span<const uint8_t, size_bytes>(buffer_, size_bytes);
    }

private:
    void init_header() noexcept {
        // Use prologue to initialize header (no trailer for context packets)
        prologue_.init_header(size_words, 0, false);
    }

    void init_stream_id() noexcept {
        // Use prologue to initialize stream ID (always present per spec)
        prologue_.init_stream_id();
    }

    void init_class_id() noexcept {
        if constexpr (has_class_id) {
            prologue_.init_class_id();
        }
    }

    void init_timestamps() noexcept {
        if constexpr (has_timestamp) {
            prologue_.init_timestamps();
        }
    }

    void write_cif_words() noexcept {
        size_t offset = calculate_cif_offset();

        // Write CIF0 (with automatic CIF1/CIF2/CIF3 enable bits)
        cif::write_u32_safe(buffer_, offset, computed_cif0);
        offset += 4;

        // Write CIF1 if present
        if constexpr (CIF1 != 0) {
            cif::write_u32_safe(buffer_, offset, CIF1);
            offset += 4;
        }

        // Write CIF2 if present
        if constexpr (CIF2 != 0) {
            cif::write_u32_safe(buffer_, offset, CIF2);
            offset += 4;
        }

        // Write CIF3 if present
        if constexpr (CIF3 != 0) {
            cif::write_u32_safe(buffer_, offset, CIF3);
        }
    }

public:
    /**
     * Get header accessor (mutable)
     * Provides access to header word fields (first 32 bits)
     * @return Mutable accessor for header fields
     */
    MutableHeaderView header() noexcept { return MutableHeaderView{&prologue_.header_word()}; }

    /**
     * Get header accessor (const)
     * Provides read-only access to header word fields (first 32 bits)
     * @return Const accessor for header fields
     */
    HeaderView header() const noexcept { return HeaderView{&prologue_.header_word()}; }

    // Stream ID accessor (always present per VITA 49.2 spec)
    uint32_t stream_id() const noexcept { return prologue_.stream_id(); }

    void set_stream_id(uint32_t id) noexcept { prologue_.set_stream_id(id); }

    // Packet count accessor (4-bit field in header, valid range 0-15)
    uint8_t packet_count() const noexcept { return prologue_.packet_count(); }

    void set_packet_count(uint8_t count) noexcept { prologue_.set_packet_count(count); }

    // Timestamp accessors

    TimeStampType timestamp() const noexcept
        requires(has_timestamp)
    {
        return prologue_.timestamp();
    }

    void set_timestamp(TimeStampType ts) noexcept
        requires(has_timestamp)
    {
        prologue_.set_timestamp(ts);
    }

    // Class ID accessors

    ClassIdValue class_id() const noexcept
        requires(has_class_id)
    {
        return prologue_.class_id();
    }

    void set_class_id(ClassIdValue cid) noexcept
        requires(has_class_id)
    {
        prologue_.set_class_id(cid);
    }

    // Field access via subscript operator
    template <uint8_t CifWord, uint8_t Bit>
    auto operator[](field::field_tag_t<CifWord, Bit> tag) noexcept
        -> FieldProxy<field::field_tag_t<CifWord, Bit>, ContextPacketBase> {
        return detail::make_field_proxy(*this, tag);
    }

    template <uint8_t CifWord, uint8_t Bit>
    auto operator[](field::field_tag_t<CifWord, Bit> tag) const noexcept
        -> FieldProxy<field::field_tag_t<CifWord, Bit>, const ContextPacketBase> {
        return detail::make_field_proxy(*this, tag);
    }

    // Internal implementation details - DO NOT USE DIRECTLY
    // These methods are required by the field access implementation (CifPacketBase concept)
    // Users should access fields via operator[] (e.g., packet[bandwidth])
    const uint8_t* context_buffer() const noexcept { return buffer_; }
    uint8_t* mutable_context_buffer() noexcept { return buffer_; }
    static constexpr size_t context_base_offset() noexcept { return calculate_context_offset(); }
    static constexpr uint32_t cif0() noexcept { return computed_cif0; }
    static constexpr uint32_t cif1() noexcept { return CIF1; }
    static constexpr uint32_t cif2() noexcept { return CIF2; }
    static constexpr uint32_t cif3() noexcept { return CIF3; }
    static constexpr size_t buffer_size() noexcept { return size_bytes; }

    // Validation (primarily for testing)
    ValidationError validate(size_t buffer_size) const noexcept {
        if (buffer_size < size_bytes) {
            return ValidationError::buffer_too_small;
        }

        // Read and decode header using shared utility
        uint32_t header = cif::read_u32_safe(buffer_, 0);
        auto decoded = detail::decode_header(header);

        // Check packet type (must be context: 4 or 5)
        if (decoded.type != PacketType::context && decoded.type != PacketType::extension_context) {
            return ValidationError::packet_type_mismatch;
        }

        // Check size field
        if (decoded.size_words != size_words) {
            return ValidationError::size_field_mismatch;
        }

        return ValidationError::none;
    }
};

// Field-based ContextPacket template (user-friendly API)
// Automatically computes CIF bitmasks from field tags
//
// Example usage:
//     using namespace vrtio::field;
//     using MyPacket = ContextPacket<NoTimeStamp, NoClassId,
//                                     bandwidth, sample_rate, gain>;
//
// This is the recommended API for most users.
template <typename TimeStampType = NoTimeStamp, typename ClassIdType = NoClassId, auto... Fields>
    requires ValidTimestampType<TimeStampType> && ValidClassIdType<ClassIdType>
class ContextPacket
    : public ContextPacketBase<TimeStampType, ClassIdType, detail::FieldMask<Fields...>::cif0,
                               detail::FieldMask<Fields...>::cif1,
                               detail::FieldMask<Fields...>::cif2,
                               detail::FieldMask<Fields...>::cif3> {
public:
    using Base =
        ContextPacketBase<TimeStampType, ClassIdType, detail::FieldMask<Fields...>::cif0,
                          detail::FieldMask<Fields...>::cif1, detail::FieldMask<Fields...>::cif2,
                          detail::FieldMask<Fields...>::cif3>;

    // Inherit constructors
    using Base::Base;
};

} // namespace vrtio
