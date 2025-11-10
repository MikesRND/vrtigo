#pragma once

#include <span>

#include <cstring>

#include "../core/cif.hpp"
#include "../core/class_id.hpp"
#include "../core/detail/field_mask.hpp"
#include "../core/detail/header_decode.hpp"
#include "../core/detail/header_init.hpp"
#include "../core/endian.hpp"
#include "../core/field_access.hpp"
#include "../core/fields.hpp"
#include "../core/timestamp_traits.hpp"
#include "../core/types.hpp"

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

    // Per spec: Context packets ALWAYS have stream ID
    static constexpr bool has_stream_id = true;

    // Extract timestamp information
    static constexpr bool has_timestamp = TimestampTraits<TimeStampType>::has_timestamp;
    static constexpr TsiType tsi = TimestampTraits<TimeStampType>::tsi;
    static constexpr TsfType tsf = TimestampTraits<TimeStampType>::tsf;

    // Extract class ID information
    static constexpr bool has_class_id = ClassIdTraits<ClassIdType>::has_class_id;
    static constexpr size_t class_id_words = ClassIdTraits<ClassIdType>::size_words;

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

    // Calculate packet structure sizes
    static constexpr size_t header_words = 1;
    static constexpr size_t stream_id_words = 1; // Always present per spec
    static constexpr size_t tsi_words = (tsi != TsiType::none) ? 1 : 0;
    static constexpr size_t tsf_words = (tsf != TsfType::none) ? 2 : 0;

    // CIF word count
    static constexpr size_t cif_words = 1 + ((CIF1 != 0) ? 1 : 0) + // CIF1
                                        ((CIF2 != 0) ? 1 : 0) +     // CIF2
                                        ((CIF3 != 0) ? 1 : 0);      // CIF3

    // Calculate context field size (compile-time, no variable fields)
    static constexpr size_t context_fields_words =
        cif::calculate_context_size_ct<CIF0, CIF1, CIF2, CIF3>();

    static constexpr size_t total_words = header_words + stream_id_words + class_id_words +
                                          tsi_words + tsf_words + cif_words + context_fields_words;

    // Complete offset calculation for CIF words
    static constexpr size_t calculate_cif_offset() {
        size_t offset_words = 1; // Header
        offset_words++;          // Stream ID (always present)

        if constexpr (has_class_id) {
            offset_words += 2; // 64-bit Class ID
        }

        // Add timestamp words
        if constexpr (tsi != TsiType::none) {
            offset_words++; // TSI
        }
        if constexpr (tsf != TsfType::none) {
            offset_words += 2; // TSF (64-bit)
        }

        return offset_words * 4; // Return bytes
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
    static constexpr size_t size_bytes = total_words * 4;
    static constexpr size_t size_words = total_words;
    static constexpr uint32_t cif0_value = computed_cif0; // For builder (with enable bits)
    static constexpr uint32_t cif1_value = CIF1;
    static constexpr uint32_t cif2_value = CIF2;
    static constexpr uint32_t cif3_value = CIF3;

    explicit ContextPacketBase(uint8_t* buffer, bool init = true) noexcept : buffer_(buffer) {
        if (init) {
            init_header();
            init_stream_id();
            init_class_id();
            init_timestamps();
            write_cif_words();
        }
    }

    // Buffer access
    std::span<uint8_t> as_bytes() noexcept { return std::span<uint8_t>(buffer_, size_bytes); }

    std::span<const uint8_t> as_bytes() const noexcept {
        return std::span<const uint8_t>(buffer_, size_bytes);
    }

private:
    void init_header() noexcept {
        // Build header using shared helper
        // Per VITA 49.2: Context packets always have Stream ID (bit 25 not used for indicator)
        uint32_t header =
            detail::build_header(4,            // Packet type: context
                                 has_class_id, // Class ID indicator
                                 false,        // Bit 26: reserved for context packets (no trailer)
                                 false,        // Bit 25: reserved for context packets
                                 false,        // Bit 24: reserved for context packets
                                 tsi,          // TSI field
                                 tsf,          // TSF field
                                 0,            // Packet count (initialized to 0)
                                 total_words   // Packet size in words
            );

        cif::write_u32_safe(buffer_, 0, header);
    }

    void init_stream_id() noexcept {
        // Always initialize stream ID to 0 (always present per spec)
        cif::write_u32_safe(buffer_, 4, 0);
    }

    void init_class_id() noexcept {
        if constexpr (has_class_id) {
            size_t offset = 8; // After header + stream ID (always present)

            // Proper VITA-49 encoding
            uint32_t word0 = ClassIdType::word0(0); // ICC = 0
            uint32_t word1 = ClassIdType::word1();  // Full 32-bit PCC

            cif::write_u32_safe(buffer_, offset, word0);
            cif::write_u32_safe(buffer_, offset + 4, word1);
        }
    }

    void init_timestamps() noexcept {
        // Calculate offset to timestamp fields
        size_t offset = 8; // After header + stream ID (always present)

        if constexpr (has_class_id) {
            offset += 8; // 64-bit class ID
        }

        // Zero-initialize TSI if present
        if constexpr (tsi != TsiType::none) {
            cif::write_u32_safe(buffer_, offset, 0);
            offset += 4;
        }

        // Zero-initialize TSF if present
        if constexpr (tsf != TsfType::none) {
            cif::write_u64_safe(buffer_, offset, 0);
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
    // Stream ID accessor (always present per VITA 49.2 spec)
    uint32_t stream_id() const noexcept { return cif::read_u32_safe(buffer_, 4); }

    void set_stream_id(uint32_t id) noexcept { cif::write_u32_safe(buffer_, 4, id); }

    // Timestamp accessors

    TimeStampType getTimeStamp() const noexcept
        requires(has_timestamp)
    {
        size_t tsi_offset = 8; // After header + stream ID (always present)
        if constexpr (has_class_id)
            tsi_offset += 8;

        size_t tsf_offset = tsi_offset;
        if constexpr (tsi != TsiType::none)
            tsf_offset += 4;

        uint32_t tsi_val = (tsi != TsiType::none) ? cif::read_u32_safe(buffer_, tsi_offset) : 0;
        uint64_t tsf_val = (tsf != TsfType::none) ? cif::read_u64_safe(buffer_, tsf_offset) : 0;

        return TimeStampType::fromComponents(tsi_val, tsf_val);
    }

    void setTimeStamp(const TimeStampType& ts) noexcept
        requires(has_timestamp)
    {
        size_t tsi_offset = 8; // After header + stream ID (always present)
        if constexpr (has_class_id)
            tsi_offset += 8;

        size_t tsf_offset = tsi_offset;
        if constexpr (tsi != TsiType::none)
            tsf_offset += 4;

        if constexpr (tsi != TsiType::none) {
            cif::write_u32_safe(buffer_, tsi_offset, ts.seconds());
        }
        if constexpr (tsf != TsfType::none) {
            cif::write_u64_safe(buffer_, tsf_offset, ts.fractional());
        }
    }

    // Field access API support - expose buffer and CIF state
    const uint8_t* context_buffer() const noexcept { return buffer_; }

    uint8_t* mutable_context_buffer() noexcept { return buffer_; }

    static constexpr size_t context_base_offset() noexcept { return calculate_context_offset(); }

    static constexpr uint32_t cif0() noexcept { return computed_cif0; }

    static constexpr uint32_t cif1() noexcept { return CIF1; }

    static constexpr uint32_t cif2() noexcept { return CIF2; }

    static constexpr uint32_t cif3() noexcept { return CIF3; }

    static constexpr size_t buffer_size() noexcept { return size_bytes; }

    // Field access via subscript operator
    template <uint8_t CifWord, uint8_t Bit>
    auto operator[](field::field_tag_t<CifWord, Bit> tag) noexcept
        -> FieldProxy<field::field_tag_t<CifWord, Bit>, ContextPacketBase> {
        return detail::get_impl(*this, tag);
    }

    template <uint8_t CifWord, uint8_t Bit>
    auto operator[](field::field_tag_t<CifWord, Bit> tag) const noexcept
        -> FieldProxy<field::field_tag_t<CifWord, Bit>, const ContextPacketBase> {
        return detail::get_impl(*this, tag);
    }

    // Validation (primarily for testing)
    ValidationError validate(size_t buffer_size) const noexcept {
        if (buffer_size < size_bytes) {
            return ValidationError::buffer_too_small;
        }

        // Read and decode header using shared utility
        uint32_t header = cif::read_u32_safe(buffer_, 0);
        auto decoded = detail::decode_header(header);

        // Check packet type (must be context: 4 or 5)
        if (decoded.type != PacketType::Context && decoded.type != PacketType::ExtensionContext) {
            return ValidationError::packet_type_mismatch;
        }

        // Check size field
        if (decoded.size_words != total_words) {
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
