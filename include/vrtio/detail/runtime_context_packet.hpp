#pragma once

#include <optional>
#include <span>

#include <cstring>
#include <vrtio/class_id.hpp>
#include <vrtio/types.hpp>

#include "cif.hpp"
#include "endian.hpp"
#include "field_access.hpp"
#include "header_decode.hpp"
#include "packet_header_accessor.hpp"
#include "variable_field_dispatch.hpp"

namespace vrtio {

/**
 * Runtime parser for context packets
 *
 * Provides safe, type-erased parsing of context packets with automatic
 * validation. Unlike ContextPacket<...>, this class doesn't require compile-time
 * knowledge of the packet structure and automatically validates on construction.
 *
 * Safety:
 * - Validates automatically on construction (no manual validate() call needed)
 * - All accessors check validation state and return std::optional for optional fields
 * - Const-only view (cannot modify packet)
 * - Makes unsafe parsing patterns impossible
 *
 * Usage:
 *   RuntimeContextPacket view(rx_buffer, buffer_size);
 *   if (view.is_valid()) {
 *       if (auto id = view.stream_id()) {
 *           std::cout << "Stream ID: " << *id << "\n";
 *       }
 *       if (auto bw = view[field::bandwidth]) {
 *           std::cout << "Bandwidth: " << bw.value() << " Hz\n";
 *       }
 *       // Process fields...
 *   }
 */
class RuntimeContextPacket {
private:
    const uint8_t* buffer_;
    size_t buffer_size_;
    ValidationError error_;

    struct ParsedStructure {
        // Header data (consolidated from decode_header)
        detail::DecodedHeader header{}; // Value-initialize to zero

        // Additional derived fields not in header
        bool has_stream_id = false; // Derived from packet type

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

    ValidationError validate_internal() noexcept {
        if (!buffer_ || buffer_size_ < 4) {
            return ValidationError::buffer_too_small;
        }

        // 1. Read and decode header using shared utility
        uint32_t header = cif::read_u32_safe(buffer_, 0);
        auto decoded = detail::decode_header(header);

        // 2. Store decoded header
        structure_.header = decoded;

        // For Context packets, stream ID presence is determined by packet type
        // Per VITA 49.2: types 1,3,4,5,6,7 have stream ID; types 0,2 do not
        structure_.has_stream_id = detail::has_stream_id_field(decoded.type);

        // 3. Validate packet type (must be context: 4 or 5)
        if (decoded.type != PacketType::context && decoded.type != PacketType::extension_context) {
            return ValidationError::invalid_packet_type;
        }

        // 4. Validate reserved bit 26 is 0 for Context packets
        // Per VITA 49.2 Table 5.1.1.1-1, bit 26 is Reserved for Context packets (must be 0)
        if (decoded.bit_26) {
            return ValidationError::unsupported_field;
        }

        // 5. Initial buffer size check
        size_t required_bytes = structure_.header.size_words * 4;
        if (buffer_size_ < required_bytes) {
            return ValidationError::buffer_too_small;
        }

        // 6. Calculate position after header fields
        size_t offset_words = 1; // Header

        if (structure_.has_stream_id) {
            if ((offset_words + 1) * 4 > buffer_size_) {
                return ValidationError::buffer_too_small;
            }
            offset_words++; // Skip stream ID
        }

        if (structure_.header.has_class_id) {
            if ((offset_words + 2) * 4 > buffer_size_) {
                return ValidationError::buffer_too_small;
            }
            offset_words += 2; // Skip 64-bit class ID
        }

        // Skip timestamp fields based on TSI/TSF
        size_t tsi_words = (structure_.header.tsi != TsiType::none) ? 1 : 0;
        size_t tsf_words = (structure_.header.tsf != TsfType::none) ? 2 : 0;
        offset_words += tsi_words + tsf_words;

        // 7. Read CIF words
        if ((offset_words + 1) * 4 > buffer_size_) {
            return ValidationError::buffer_too_small;
        }
        structure_.cif0 = cif::read_u32_safe(buffer_, offset_words * 4);
        offset_words++;

        if (structure_.cif0 & (1U << cif::CIF1_ENABLE_BIT)) {
            if ((offset_words + 1) * 4 > buffer_size_) {
                return ValidationError::buffer_too_small;
            }
            structure_.cif1 = cif::read_u32_safe(buffer_, offset_words * 4);
            offset_words++;
        }

        if (structure_.cif0 & (1U << cif::CIF2_ENABLE_BIT)) {
            if ((offset_words + 1) * 4 > buffer_size_) {
                return ValidationError::buffer_too_small;
            }
            structure_.cif2 = cif::read_u32_safe(buffer_, offset_words * 4);
            offset_words++;
        }

        if (structure_.cif0 & (1U << cif::CIF3_ENABLE_BIT)) {
            if ((offset_words + 1) * 4 > buffer_size_) {
                return ValidationError::buffer_too_small;
            }
            structure_.cif3 = cif::read_u32_safe(buffer_, offset_words * 4);
            offset_words++;
        }

        // COMPLETE validation: reject ANY unsupported bits
        // CIF0_SUPPORTED_MASK now includes the CIF1/CIF2 enable bits (1 and 2)
        if (structure_.cif0 & ~cif::CIF0_SUPPORTED_MASK) {
            return ValidationError::unsupported_field;
        }

        // Check CIF1 if present
        if (structure_.cif0 & (1U << cif::CIF1_ENABLE_BIT)) {
            if (structure_.cif1 & ~cif::CIF1_SUPPORTED_MASK) {
                return ValidationError::unsupported_field;
            }
        }

        // Check CIF2 if present
        if (structure_.cif0 & (1U << cif::CIF2_ENABLE_BIT)) {
            if (structure_.cif2 & ~cif::CIF2_SUPPORTED_MASK) {
                return ValidationError::unsupported_field;
            }
        }

        // Check CIF3 if present
        if (structure_.cif0 & (1U << cif::CIF3_ENABLE_BIT)) {
            if (structure_.cif3 & ~cif::CIF3_SUPPORTED_MASK) {
                return ValidationError::unsupported_field;
            }
        }

        // Store context field base offset
        structure_.context_base_bytes = offset_words * 4;

        // 8. Calculate context field sizes with variable field handling
        size_t context_fields_words = 0;

        // Process all fixed fields first (from MSB to LSB)
        for (int bit = 31; bit >= 0; bit--) {
            if (bit == cif::GPS_ASCII_BIT || bit == cif::CONTEXT_ASSOC_BIT)
                continue; // Skip variable fields for now

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
                return ValidationError::buffer_too_small;
            }
            structure_.gps_ascii.size_words =
                cif::read_gps_ascii_length_words(buffer_, structure_.gps_ascii.offset_bytes);

            // Check entire field fits in buffer!
            if ((offset_words + context_fields_words + structure_.gps_ascii.size_words) * 4 >
                buffer_size_) {
                return ValidationError::buffer_too_small;
            }

            context_fields_words += structure_.gps_ascii.size_words;
        }

        if (structure_.cif0 & (1U << cif::CONTEXT_ASSOC_BIT)) {
            structure_.context_assoc.present = true;
            structure_.context_assoc.offset_bytes = (offset_words + context_fields_words) * 4;

            // Check counts word is present
            if ((offset_words + context_fields_words + 1) * 4 > buffer_size_) {
                return ValidationError::buffer_too_small;
            }

            // Read length with CORRECTED format
            structure_.context_assoc.size_words = cif::read_context_assoc_length_words(
                buffer_, structure_.context_assoc.offset_bytes);

            // Check entire field fits!
            if ((offset_words + context_fields_words + structure_.context_assoc.size_words) * 4 >
                buffer_size_) {
                return ValidationError::buffer_too_small;
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

        // 9. Calculate total expected size
        // Note: Context packets do not support trailer fields (bit 26 is Reserved)
        structure_.calculated_size_words = offset_words + context_fields_words;

        // 10. Final validation: calculated size must match header
        if (structure_.calculated_size_words != structure_.header.size_words) {
            return ValidationError::size_field_mismatch;
        }

        return ValidationError::none;
    }

public:
    /**
     * Construct runtime parser and automatically validate
     * @param buffer Pointer to packet buffer
     * @param buffer_size Size of buffer in bytes
     */
    explicit RuntimeContextPacket(const uint8_t* buffer, size_t buffer_size) noexcept
        : buffer_(buffer),
          buffer_size_(buffer_size),
          error_(ValidationError::none),
          structure_{} {
        // structure_{} zero-initializes all members including padding
        error_ = validate_internal();
    }

    /**
     * Get validation error
     * @return ValidationError::none if packet is valid, otherwise specific error
     */
    ValidationError error() const noexcept { return error_; }

    /**
     * Check if packet is valid
     * @return true if validation passed
     */
    bool is_valid() const noexcept { return error_ == ValidationError::none; }

    /**
     * Get header accessor
     * @return Const accessor for header word fields
     */
    HeaderView header() const noexcept { return HeaderView{&structure_.header}; }

    /**
     * Get packet type decoded from the header
     */
    PacketType type() const noexcept { return structure_.header.type; }

    // Header field accessors

    /**
     * Get timestamp integer format type (TSI field)
     *
     * Returns the format of the integer timestamp component.
     * This is metadata ABOUT the timestamp, not the timestamp data itself.
     *
     * @return TSI type from header
     */
    TsiType tsi_type() const noexcept { return structure_.header.tsi; }

    /**
     * Get timestamp fractional format type (TSF field)
     *
     * Returns the format of the fractional timestamp component.
     * This is metadata ABOUT the timestamp, not the timestamp data itself.
     *
     * @return TSF type from header
     */
    TsfType tsf_type() const noexcept { return structure_.header.tsf; }

    /**
     * Check if packet has stream ID
     * @return true (always present per VITA 49.2 spec for context packets)
     */
    bool has_stream_id() const noexcept { return true; }

    /**
     * Check if packet has class ID
     * @return true if class ID indicator is set
     */
    bool has_class_id() const noexcept { return header().has_class_id(); }

    /**
     * Check if packet has trailer
     * @return false (always false per VITA 49.2 spec - bit 26 reserved for context packets)
     */
    bool has_trailer() const noexcept { return false; }

    /**
     * Check if packet has integer timestamp
     * @return true if TSI != none
     */
    bool has_timestamp_integer() const noexcept { return header().has_timestamp_integer(); }

    /**
     * Check if packet has fractional timestamp
     * @return true if TSF != none
     */
    bool has_timestamp_fractional() const noexcept { return header().has_timestamp_fractional(); }

    /**
     * Get integer timestamp
     * @return Integer timestamp if packet has TSI and is valid, otherwise std::nullopt
     */
    std::optional<uint32_t> timestamp_integer() const noexcept {
        if (!is_valid() || structure_.header.tsi == TsiType::none) {
            return std::nullopt;
        }

        // Calculate offset to integer timestamp
        size_t offset = 4; // After header
        if (structure_.has_stream_id) {
            offset += 4;
        }
        if (structure_.header.has_class_id) {
            offset += 8; // 64-bit class ID
        }

        return cif::read_u32_safe(buffer_, offset);
    }

    /**
     * Get fractional timestamp
     * @return Fractional timestamp if packet has TSF and is valid, otherwise std::nullopt
     */
    std::optional<uint64_t> timestamp_fractional() const noexcept {
        if (!is_valid() || structure_.header.tsf == TsfType::none) {
            return std::nullopt;
        }

        // Calculate offset to fractional timestamp
        size_t offset = 4; // After header
        if (structure_.has_stream_id) {
            offset += 4;
        }
        if (structure_.header.has_class_id) {
            offset += 8; // 64-bit class ID
        }
        if (structure_.header.tsi != TsiType::none) {
            offset += 4; // Integer timestamp
        }

        // Read fractional timestamp (size depends on TSF type per VITA 49.2)
        switch (structure_.header.tsf) {
            case TsfType::none:
                return std::nullopt;
            case TsfType::sample_count:
                // 64-bit sample count
                return cif::read_u64_safe(buffer_, offset);
            case TsfType::real_time:
                // 64-bit picoseconds
                return cif::read_u64_safe(buffer_, offset);
            case TsfType::free_running:
                // 32-bit free running count (pad to 64-bit)
                return static_cast<uint64_t>(cif::read_u32_safe(buffer_, offset));
        }
        return std::nullopt;
    }

    // CIF accessors

    uint32_t cif0() const noexcept { return structure_.cif0; }
    uint32_t cif1() const noexcept { return structure_.cif1; }
    uint32_t cif2() const noexcept { return structure_.cif2; }
    uint32_t cif3() const noexcept { return structure_.cif3; }

    // Note: Context packets do not support trailers (bit 26 is Reserved, validated during parse)

    // Stream ID accessor
    std::optional<uint32_t> stream_id() const noexcept {
        if (!is_valid() || !structure_.has_stream_id) {
            return std::nullopt;
        }
        return cif::read_u32_safe(buffer_, 4); // Right after header
    }

    // Packet count accessor (4-bit field in header, always present)
    uint8_t packet_count() const noexcept { return structure_.header.packet_count; }

    // Class ID accessor
    [[nodiscard]] std::optional<ClassIdValue> class_id() const noexcept {
        if (!is_valid() || !structure_.header.has_class_id) {
            return std::nullopt;
        }

        size_t offset = 4; // After header
        if (structure_.has_stream_id) {
            offset += 4;
        }

        uint32_t word0 = cif::read_u32_safe(buffer_, offset);
        uint32_t word1 = cif::read_u32_safe(buffer_, offset + 4);

        return ClassIdValue::fromWords(word0, word1);
    }

    // Note: Field access has been standardized through the operator[] API.
    // Use view[field::field_name] for all CIF field access, including:
    //   - view[field::controller_uuid] - instead of controller_uuid()
    //   - view[field::gps_ascii] - instead of has_gps_ascii() / gps_ascii_data()
    //   - view[field::context_association_lists] - instead of has_context_association() /
    //   context_association_data()
    //   - if (view[field::field_name]) - for presence checks instead of has_cifN_field<>()
    //
    // This provides a uniform API with FieldProxy for raw access (.raw_bytes()) and
    // interpreted access (.value()).

    // Size queries
    size_t packet_size_bytes() const noexcept { return structure_.header.size_words * 4; }

    size_t packet_size_words() const noexcept { return structure_.header.size_words; }

    // Field access API support - expose buffer and offsets
    const uint8_t* context_buffer() const noexcept { return buffer_; }

    size_t context_base_offset() const noexcept { return structure_.context_base_bytes; }

    size_t buffer_size() const noexcept { return buffer_size_; }

    /**
     * Get entire packet as bytes
     * @return Span of entire packet if valid, otherwise empty span
     */
    std::span<const uint8_t> as_bytes() const noexcept {
        if (!is_valid()) {
            return {};
        }
        return std::span<const uint8_t>(buffer_, packet_size_bytes());
    }

    // Field access via subscript operator
    template <uint8_t CifWord, uint8_t Bit>
    auto operator[](field::field_tag_t<CifWord, Bit> tag) const noexcept
        -> FieldProxy<field::field_tag_t<CifWord, Bit>, const RuntimeContextPacket> {
        return detail::make_field_proxy(*this, tag);
    }
};

} // namespace vrtio
