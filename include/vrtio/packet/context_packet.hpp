#pragma once

#include "../core/types.hpp"
#include "../core/cif.hpp"
#include "../core/class_id.hpp"
#include "../core/timestamp_traits.hpp"
#include "../core/endian.hpp"
#include "../core/detail/header_decode.hpp"
#include <cstring>
#include <span>

namespace vrtio {

// Compile-time context packet template
// Creates packets with known structure at compile time
// Variable-length fields are NOT supported in this template
template<
    bool HasStreamId = true,
    typename TimeStampType = NoTimeStamp,
    typename ClassIdType = NoClassId,
    uint32_t CIF0 = 0,
    uint32_t CIF1 = 0,
    uint32_t CIF2 = 0,
    uint32_t CIF3 = 0,
    bool HasTrailer = false
>
    requires ValidTimestampType<TimeStampType> &&
             ValidClassIdType<ClassIdType>
class ContextPacket {
private:
    uint8_t* buffer_;

    // Extract timestamp information
    static constexpr bool has_timestamp = TimestampTraits<TimeStampType>::has_timestamp;
    static constexpr tsi_type tsi = TimestampTraits<TimeStampType>::tsi;
    static constexpr tsf_type tsf = TimestampTraits<TimeStampType>::tsf;

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
        "Do not set CIF1/CIF2/CIF3 enable bits (1,2,3) in CIF0 - they are auto-managed based on CIF1/CIF2/CIF3 parameters");

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
                if (bit == 1 || bit == 2 || bit == 31) continue;
                // Check for unsupported fields (size 0 means unsupported for fixed fields)
                if (!cif::CIF0_FIELDS[bit].is_variable &&
                    cif::CIF0_FIELDS[bit].size_words == 0 &&
                    cif::CIF0_FIELDS[bit].is_supported) {
                    return false;
                }
            }
        }
        return true;
    }

    static_assert(validate_cif0_sizes(),
        "CIF0 contains fields with undefined sizes");

    // Calculate packet structure sizes
    static constexpr size_t header_words = 1;
    static constexpr size_t stream_id_words = HasStreamId ? 1 : 0;
    static constexpr size_t tsi_words = (tsi != tsi_type::none) ? 1 : 0;
    static constexpr size_t tsf_words = (tsf != tsf_type::none) ? 2 : 0;

    // CIF word count
    static constexpr size_t cif_words = 1 +
        ((CIF1 != 0) ? 1 : 0) +  // CIF1
        ((CIF2 != 0) ? 1 : 0) +  // CIF2
        ((CIF3 != 0) ? 1 : 0);   // CIF3

    // Calculate context field size (compile-time, no variable fields)
    static constexpr size_t context_fields_words =
        cif::calculate_context_size_ct<CIF0, CIF1, CIF2, CIF3>();

    static constexpr size_t trailer_words = HasTrailer ? 1 : 0;

    static constexpr size_t total_words = header_words + stream_id_words +
        class_id_words + tsi_words + tsf_words + cif_words +
        context_fields_words + trailer_words;

    // Complete offset calculation for CIF words
    static constexpr size_t calculate_cif_offset() {
        size_t offset_words = 1;  // Header

        if constexpr (HasStreamId) {
            offset_words++;  // Stream ID
        }

        if constexpr (has_class_id) {
            offset_words += 2;  // 64-bit Class ID
        }

        // Add timestamp words
        if constexpr (tsi != tsi_type::none) {
            offset_words++;  // TSI
        }
        if constexpr (tsf != tsf_type::none) {
            offset_words += 2;  // TSF (64-bit)
        }

        return offset_words * 4;  // Return bytes
    }

    // Calculate offset to context fields (after CIF words)
    static constexpr size_t calculate_context_offset() {
        size_t offset = calculate_cif_offset();

        // Add CIF words
        offset += 4;  // CIF0 always present
        if constexpr (CIF1 != 0) {
            offset += 4;  // CIF1
        }
        if constexpr (CIF2 != 0) {
            offset += 4;  // CIF2
        }
        if constexpr (CIF3 != 0) {
            offset += 4;  // CIF3
        }

        return offset;
    }

public:
    static constexpr size_t size_bytes = total_words * 4;
    static constexpr size_t size_words = total_words;
    static constexpr uint32_t cif0_value = computed_cif0;  // For builder (with enable bits)
    static constexpr uint32_t cif1_value = CIF1;
    static constexpr uint32_t cif2_value = CIF2;
    static constexpr uint32_t cif3_value = CIF3;

    explicit ContextPacket(uint8_t* buffer, bool init = true) noexcept
        : buffer_(buffer) {
        if (init) {
            init_header();
            init_stream_id();
            init_class_id();
            init_timestamps();
            write_cif_words();
            init_trailer();
        }
    }

    // Buffer access
    uint8_t* data() noexcept { return buffer_; }
    const uint8_t* data() const noexcept { return buffer_; }

    std::span<uint8_t> as_bytes() noexcept {
        return std::span<uint8_t>(buffer_, size_bytes);
    }

    std::span<const uint8_t> as_bytes() const noexcept {
        return std::span<const uint8_t>(buffer_, size_bytes);
    }

private:
    void init_header() noexcept {
        uint32_t header = 0;

        // Packet type: context (4)
        header |= (4U << 28);

        // Class ID bit
        if constexpr (has_class_id) {
            header |= (1U << 27);
        }

        // Trailer bit
        if constexpr (HasTrailer) {
            header |= (1U << 26);
        }

        // Stream ID bit (bit 25 for context packets!)
        if constexpr (HasStreamId) {
            header |= (1U << 25);
        }

        // TSI/TSF bits (22-24)
        header |= (static_cast<uint8_t>(tsi) << 22);
        header |= (static_cast<uint8_t>(tsf) << 20);

        // Packet size
        header |= (total_words & 0xFFFF);

        cif::write_u32_safe(buffer_, 0, header);
    }

    void init_stream_id() noexcept {
        if constexpr (HasStreamId) {
            // Initialize to 0, user can set later
            cif::write_u32_safe(buffer_, 4, 0);
        }
    }

    void init_class_id() noexcept {
        if constexpr (has_class_id) {
            size_t offset = 4;  // After header
            if constexpr (HasStreamId) {
                offset += 4;
            }

            // Proper VITA-49 encoding
            uint32_t word0 = ClassIdType::word0(0);  // ICC = 0
            uint32_t word1 = ClassIdType::word1();   // Full 32-bit PCC

            cif::write_u32_safe(buffer_, offset, word0);
            cif::write_u32_safe(buffer_, offset + 4, word1);
        }
    }

    void init_timestamps() noexcept {
        // Calculate offset to timestamp fields
        size_t offset = 4;  // After header

        if constexpr (HasStreamId) {
            offset += 4;
        }

        if constexpr (has_class_id) {
            offset += 8;  // 64-bit class ID
        }

        // Zero-initialize TSI if present
        if constexpr (tsi != tsi_type::none) {
            cif::write_u32_safe(buffer_, offset, 0);
            offset += 4;
        }

        // Zero-initialize TSF if present
        if constexpr (tsf != tsf_type::none) {
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

    void init_trailer() noexcept {
        if constexpr (HasTrailer) {
            size_t offset = (total_words - 1) * 4;
            cif::write_u32_safe(buffer_, offset, 0);  // Initialize to 0
        }
    }

public:
    // Stream ID accessor
    uint32_t stream_id() const noexcept requires(HasStreamId) {
        return cif::read_u32_safe(buffer_, 4);
    }

    void set_stream_id(uint32_t id) noexcept requires(HasStreamId) {
        cif::write_u32_safe(buffer_, 4, id);
    }

    // Timestamp accessors

    // Integer timestamp accessors (only if TSI != none)
    uint32_t timestamp_integer() const noexcept requires(tsi != tsi_type::none) {
        size_t offset = 4;
        if constexpr (HasStreamId) offset += 4;
        if constexpr (has_class_id) offset += 8;
        return cif::read_u32_safe(buffer_, offset);
    }

    void set_timestamp_integer(uint32_t ts) noexcept requires(tsi != tsi_type::none) {
        size_t offset = 4;
        if constexpr (HasStreamId) offset += 4;
        if constexpr (has_class_id) offset += 8;
        cif::write_u32_safe(buffer_, offset, ts);
    }

    // Fractional timestamp accessors (only if TSF != none)
    uint64_t timestamp_fractional() const noexcept requires(tsf != tsf_type::none) {
        size_t offset = 4;
        if constexpr (HasStreamId) offset += 4;
        if constexpr (has_class_id) offset += 8;
        if constexpr (tsi != tsi_type::none) offset += 4;
        return cif::read_u64_safe(buffer_, offset);
    }

    void set_timestamp_fractional(uint64_t ts) noexcept requires(tsf != tsf_type::none) {
        size_t offset = 4;
        if constexpr (HasStreamId) offset += 4;
        if constexpr (has_class_id) offset += 8;
        if constexpr (tsi != tsi_type::none) offset += 4;
        cif::write_u64_safe(buffer_, offset, ts);
    }

    // Unified timestamp accessors
    TimeStampType getTimeStamp() const noexcept requires(has_timestamp) {
        return TimeStampType::fromComponents(
            timestamp_integer(),
            timestamp_fractional()
        );
    }

    void setTimeStamp(const TimeStampType& ts) noexcept requires(has_timestamp) {
        set_timestamp_integer(ts.seconds());
        set_timestamp_fractional(ts.fractional());
    }

    // Trailer accessors
    uint32_t trailer() const noexcept requires(HasTrailer) {
        size_t offset = (total_words - 1) * 4;
        return cif::read_u32_safe(buffer_, offset);
    }

    void set_trailer(uint32_t value) noexcept requires(HasTrailer) {
        size_t offset = (total_words - 1) * 4;
        cif::write_u32_safe(buffer_, offset, value);
    }

    // Packet size queries
    static constexpr size_t packet_size_bytes() noexcept {
        return size_bytes;
    }

    static constexpr size_t packet_size_words() noexcept {
        return total_words;
    }

    // Field access API support - expose buffer and CIF state
    const uint8_t* context_buffer() const noexcept {
        return buffer_;
    }

    uint8_t* mutable_context_buffer() noexcept {
        return buffer_;
    }

    static constexpr size_t context_base_offset() noexcept {
        return calculate_context_offset();
    }

    static constexpr uint32_t cif0() noexcept {
        return computed_cif0;
    }

    static constexpr uint32_t cif1() noexcept {
        return CIF1;
    }

    static constexpr uint32_t cif2() noexcept {
        return CIF2;
    }

    static constexpr uint32_t cif3() noexcept {
        return CIF3;
    }

    static constexpr size_t buffer_size() noexcept {
        return size_bytes;
    }

    // Validation (primarily for testing)
    validation_error validate(size_t buffer_size) const noexcept {
        if (buffer_size < size_bytes) {
            return validation_error::buffer_too_small;
        }

        // Read and decode header using shared utility
        uint32_t header = cif::read_u32_safe(buffer_, 0);
        auto decoded = detail::decode_header(header);

        // Check packet type (must be context: 4 or 5)
        if (decoded.type != packet_type::context && decoded.type != packet_type::ext_context) {
            return validation_error::packet_type_mismatch;
        }

        // Check size field
        if (decoded.size_words != total_words) {
            return validation_error::size_field_mismatch;
        }

        return validation_error::none;
    }
};

}  // namespace vrtio