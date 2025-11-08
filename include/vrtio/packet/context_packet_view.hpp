#pragma once

#include "../core/types.hpp"
#include "../core/cif.hpp"
#include "../core/endian.hpp"
#include "../core/detail/header_decode.hpp"
#include "../core/detail/variable_field_dispatch.hpp"
#include <optional>
#include <span>
#include <cstring>

namespace vrtio {

// Runtime parser for context packets
// Can parse arbitrary context packets from network with any CIF configuration
class ContextPacketView {
private:
    const uint8_t* buffer_;
    size_t buffer_size_;

    struct ParsedStructure {
        // Header fields
        uint16_t packet_size_words = 0;
        uint8_t packet_type = 0;
        bool has_stream_id = false;
        bool has_class_id = false;
        tsi_type tsi = tsi_type::none;
        tsf_type tsf = tsf_type::none;
        bool has_trailer = false;

        // CIF words
        uint32_t cif0 = 0;
        uint32_t cif1 = 0;
        uint32_t cif2 = 0;
        uint32_t cif3 = 0;

        // Variable field info (populated during validation)
        struct VariableFieldInfo {
            bool present = false;
            size_t offset_bytes = 0;
            size_t size_words = 0;
        };
        VariableFieldInfo gps_ascii;
        VariableFieldInfo context_assoc;

        // Store base offset for context fields
        size_t context_base_bytes = 0;

        // Calculated after parsing variable fields
        size_t calculated_size_words = 0;
    } structure_;

    bool validated_ = false;

public:
    explicit ContextPacketView(const uint8_t* buffer, size_t size) noexcept
        : buffer_(buffer), buffer_size_(size) {}

    validation_error validate() noexcept {
        if (!buffer_ || buffer_size_ < 4) {
            return validation_error::buffer_too_small;
        }

        // 1. Read and decode header using shared utility
        uint32_t header = cif::read_u32_safe(buffer_, 0);
        auto decoded = detail::decode_header(header);

        // 2. Store decoded header fields
        structure_.packet_type = static_cast<uint8_t>(decoded.type);
        structure_.has_class_id = decoded.has_class_id;
        structure_.has_trailer = decoded.has_trailer;
        structure_.has_stream_id = decoded.has_stream_id;  // Bit 25 for context packets
        structure_.tsi = decoded.tsi;
        structure_.tsf = decoded.tsf;
        structure_.packet_size_words = decoded.size_words;

        // 3. Validate packet type (must be context: 4 or 5)
        if (decoded.type != packet_type::context && decoded.type != packet_type::ext_context) {
            return validation_error::invalid_packet_type;
        }

        // 4. Initial buffer size check
        size_t required_bytes = structure_.packet_size_words * 4;
        if (buffer_size_ < required_bytes) {
            return validation_error::buffer_too_small;
        }

        // 5. Calculate position after header fields
        size_t offset_words = 1;  // Header

        if (structure_.has_stream_id) {
            if ((offset_words + 1) * 4 > buffer_size_) {
                return validation_error::buffer_too_small;
            }
            offset_words++;  // Skip stream ID
        }

        if (structure_.has_class_id) {
            if ((offset_words + 2) * 4 > buffer_size_) {
                return validation_error::buffer_too_small;
            }
            offset_words += 2;  // Skip 64-bit class ID
        }

        // Skip timestamp fields based on TSI/TSF
        size_t tsi_words = (structure_.tsi != tsi_type::none) ? 1 : 0;
        size_t tsf_words = (structure_.tsf != tsf_type::none) ? 2 : 0;
        offset_words += tsi_words + tsf_words;

        // 6. Read CIF words
        if ((offset_words + 1) * 4 > buffer_size_) {
            return validation_error::buffer_too_small;
        }
        structure_.cif0 = cif::read_u32_safe(buffer_, offset_words * 4);
        offset_words++;

        if (structure_.cif0 & (1U << cif::CIF1_ENABLE_BIT)) {
            if ((offset_words + 1) * 4 > buffer_size_) {
                return validation_error::buffer_too_small;
            }
            structure_.cif1 = cif::read_u32_safe(buffer_, offset_words * 4);
            offset_words++;
        }

        if (structure_.cif0 & (1U << cif::CIF2_ENABLE_BIT)) {
            if ((offset_words + 1) * 4 > buffer_size_) {
                return validation_error::buffer_too_small;
            }
            structure_.cif2 = cif::read_u32_safe(buffer_, offset_words * 4);
            offset_words++;
        }

        if (structure_.cif0 & (1U << cif::CIF3_ENABLE_BIT)) {
            if ((offset_words + 1) * 4 > buffer_size_) {
                return validation_error::buffer_too_small;
            }
            structure_.cif3 = cif::read_u32_safe(buffer_, offset_words * 4);
            offset_words++;
        }

        // COMPLETE validation: reject ANY unsupported bits
        // CIF0_SUPPORTED_MASK now includes the CIF1/CIF2 enable bits (1 and 2)
        if (structure_.cif0 & ~cif::CIF0_SUPPORTED_MASK) {
            return validation_error::unsupported_field;
        }

        // Check CIF1 if present
        if (structure_.cif0 & (1U << cif::CIF1_ENABLE_BIT)) {
            if (structure_.cif1 & ~cif::CIF1_SUPPORTED_MASK) {
                return validation_error::unsupported_field;
            }
        }

        // Check CIF2 if present
        if (structure_.cif0 & (1U << cif::CIF2_ENABLE_BIT)) {
            if (structure_.cif2 & ~cif::CIF2_SUPPORTED_MASK) {
                return validation_error::unsupported_field;
            }
        }

        // Check CIF3 if present
        if (structure_.cif0 & (1U << cif::CIF3_ENABLE_BIT)) {
            if (structure_.cif3 & ~cif::CIF3_SUPPORTED_MASK) {
                return validation_error::unsupported_field;
            }
        }

        // Store context field base offset
        structure_.context_base_bytes = offset_words * 4;

        // 7. Calculate context field sizes with variable field handling
        size_t context_fields_words = 0;

        // Process all fixed fields first (from MSB to LSB)
        for (int bit = 31; bit >= 0; bit--) {
            if (bit == cif::GPS_ASCII_BIT || bit == cif::CONTEXT_ASSOC_BIT) continue;  // Skip variable fields for now

            if (structure_.cif0 & (1U << bit)) {
                context_fields_words += cif::CIF0_FIELDS[bit].size_words;
            }
        }

        // Now handle variable fields IN ORDER (bit 10 before bit 9)
        if (structure_.cif0 & (1U << cif::GPS_ASCII_BIT)) {
            structure_.gps_ascii.present = true;
            structure_.gps_ascii.offset_bytes = (offset_words + context_fields_words) * 4;

            // Read the length from the buffer
            if ((offset_words + context_fields_words + 1) * 4 > buffer_size_) {
                return validation_error::buffer_too_small;
            }
            structure_.gps_ascii.size_words = cif::read_gps_ascii_length_words(
                buffer_, structure_.gps_ascii.offset_bytes);

            // Check entire field fits in buffer!
            if ((offset_words + context_fields_words + structure_.gps_ascii.size_words) * 4
                > buffer_size_) {
                return validation_error::buffer_too_small;
            }

            context_fields_words += structure_.gps_ascii.size_words;
        }

        if (structure_.cif0 & (1U << cif::CONTEXT_ASSOC_BIT)) {
            structure_.context_assoc.present = true;
            structure_.context_assoc.offset_bytes = (offset_words + context_fields_words) * 4;

            // Check counts word is present
            if ((offset_words + context_fields_words + 1) * 4 > buffer_size_) {
                return validation_error::buffer_too_small;
            }

            // Read length with CORRECTED format
            structure_.context_assoc.size_words = cif::read_context_assoc_length_words(
                buffer_, structure_.context_assoc.offset_bytes);

            // Check entire field fits!
            if ((offset_words + context_fields_words + structure_.context_assoc.size_words) * 4
                > buffer_size_) {
                return validation_error::buffer_too_small;
            }

            context_fields_words += structure_.context_assoc.size_words;
        }

        // Process CIF1 fields
        if (structure_.cif0 & (1U << cif::CIF1_ENABLE_BIT)) {
            for (int bit = 31; bit >= 0; --bit) {
                if (structure_.cif1 & (1U << bit)) {
                    context_fields_words += cif::CIF1_FIELDS[bit].size_words;
                }
            }
        }

        // Process CIF2 fields
        if (structure_.cif0 & (1U << cif::CIF2_ENABLE_BIT)) {
            for (int bit = 31; bit >= 0; --bit) {
                if (structure_.cif2 & (1U << bit)) {
                    context_fields_words += cif::CIF2_FIELDS[bit].size_words;
                }
            }
        }

        // Process CIF3 fields
        if (structure_.cif0 & (1U << cif::CIF3_ENABLE_BIT)) {
            for (int bit = 31; bit >= 0; --bit) {
                if (structure_.cif3 & (1U << bit)) {
                    context_fields_words += cif::CIF3_FIELDS[bit].size_words;
                }
            }
        }

        // 8. Calculate total expected size
        structure_.calculated_size_words = offset_words + context_fields_words;
        if (structure_.has_trailer) {
            structure_.calculated_size_words++;
        }

        // 9. Final validation: calculated size must match header
        if (structure_.calculated_size_words != structure_.packet_size_words) {
            return validation_error::size_field_mismatch;
        }

        validated_ = true;
        return validation_error::none;
    }

    // Query methods
    bool is_valid() const noexcept { return validated_; }

    uint32_t cif0() const noexcept { return structure_.cif0; }
    uint32_t cif1() const noexcept { return structure_.cif1; }
    uint32_t cif2() const noexcept { return structure_.cif2; }
    uint32_t cif3() const noexcept { return structure_.cif3; }

    bool has_stream_id() const noexcept { return structure_.has_stream_id; }
    bool has_class_id() const noexcept { return structure_.has_class_id; }
    bool has_trailer() const noexcept { return structure_.has_trailer; }

    // Stream ID accessor
    std::optional<uint32_t> stream_id() const noexcept {
        if (!validated_ || !structure_.has_stream_id) {
            return std::nullopt;
        }
        return cif::read_u32_safe(buffer_, 4);  // Right after header
    }

    // Class ID accessor
    struct ClassIdValue {
        uint32_t oui;
        uint8_t icc;
        uint32_t pcc;
    };

    std::optional<ClassIdValue> class_id() const noexcept {
        if (!validated_ || !structure_.has_class_id) {
            return std::nullopt;
        }

        size_t offset = 4;  // After header
        if (structure_.has_stream_id) {
            offset += 4;
        }

        uint32_t word0 = cif::read_u32_safe(buffer_, offset);
        uint32_t word1 = cif::read_u32_safe(buffer_, offset + 4);

        ClassIdValue result;
        result.oui = (word0 >> 8) & 0xFFFFFF;
        result.icc = word0 & 0xFF;
        result.pcc = word1;  // Full 32 bits

        return result;
    }

    // Runtime field presence query helpers
    template<uint32_t Bit>
    bool has_cif0_field() const noexcept {
        return validated_ && (structure_.cif0 & Bit);
    }

    template<uint32_t Bit>
    bool has_cif1_field() const noexcept {
        return validated_ && (structure_.cif1 & Bit);
    }

    template<uint32_t Bit>
    bool has_cif2_field() const noexcept {
        return validated_ && (structure_.cif2 & Bit);
    }

    // CIF2 field accessors

    // Controller UUID accessor (CIF2 bit 22) - returns pointer to 128-bit UUID
    std::optional<std::span<const uint8_t>> controller_uuid() const noexcept {
        constexpr uint8_t bit = 22;

        if (!validated_ || !(structure_.cif2 & (1U << bit))) {
            return std::nullopt;
        }

        size_t offset = detail::calculate_field_offset_runtime(
            structure_.cif0, structure_.cif1, structure_.cif2, structure_.cif3,
            2, bit,
            buffer_,
            structure_.context_base_bytes,
            buffer_size_);

        if (offset == SIZE_MAX) {
            return std::nullopt;
        }

        // Controller UUID is 4 words (16 bytes / 128 bits)
        return std::span<const uint8_t>(buffer_ + offset, 16);
    }

    // Variable field accessors
    bool has_gps_ascii() const noexcept {
        return validated_ && structure_.gps_ascii.present;
    }

    std::span<const uint8_t> gps_ascii_data() const noexcept {
        if (!has_gps_ascii()) return {};
        return std::span<const uint8_t>(
            buffer_ + structure_.gps_ascii.offset_bytes,
            structure_.gps_ascii.size_words * 4);
    }

    bool has_context_association() const noexcept {
        return validated_ && structure_.context_assoc.present;
    }

    std::span<const uint8_t> context_association_data() const noexcept {
        if (!has_context_association()) return {};
        return std::span<const uint8_t>(
            buffer_ + structure_.context_assoc.offset_bytes,
            structure_.context_assoc.size_words * 4);
    }

    // Size queries
    size_t packet_size_bytes() const noexcept {
        return structure_.packet_size_words * 4;
    }

    size_t packet_size_words() const noexcept {
        return structure_.packet_size_words;
    }

    // Field access API support - expose buffer and offsets
    const uint8_t* context_buffer() const noexcept {
        return buffer_;
    }

    size_t context_base_offset() const noexcept {
        return structure_.context_base_bytes;
    }

    size_t buffer_size() const noexcept {
        return buffer_size_;
    }
};

}  // namespace vrtio