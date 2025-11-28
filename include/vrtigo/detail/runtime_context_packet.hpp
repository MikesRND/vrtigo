#pragma once

#include <optional>
#include <span>

#include <vrtigo/types.hpp>

#include "cif.hpp"
#include "field_access.hpp"
#include "parse_result.hpp"
#include "runtime_packet_base.hpp"
#include "variable_field_dispatch.hpp"

namespace vrtigo {

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
 *   auto result = RuntimeContextPacket::parse(rx_buffer);  // rx_buffer is std::span<const uint8_t>
 *   if (result.ok()) {
 *       const auto& view = result.value();
 *       if (auto id = view.stream_id()) {
 *           std::cout << "Stream ID: " << *id << "\n";
 *       }
 *       if (auto bw = view[field::bandwidth]) {
 *           std::cout << "Bandwidth: " << bw.value() << " Hz\n";
 *       }
 *       // Process fields...
 *   } else {
 *       std::cerr << "Parse error: " << result.error().message() << "\n";
 *   }
 */
class RuntimeContextPacket : public detail::RuntimePacketBase {
private:
    struct ContextFields {
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

        // Store base offset for context fields (after CIF words)
        size_t context_base_bytes = 0;

        // Calculated after parsing variable fields
        size_t calculated_size_words = 0;
    } context_fields_;

    ValidationError validate_context_packet() noexcept {
        // 1. Parse common prologue (header, stream_id, class_id, timestamp)
        ValidationError prologue_error = parse_prologue();
        if (prologue_error != ValidationError::none) {
            return prologue_error;
        }

        // 2. Validate packet type (must be context: 4 or 5)
        if (prologue_.header.type != PacketType::context &&
            prologue_.header.type != PacketType::extension_context) {
            return ValidationError::packet_type_mismatch;
        }

        // 3. Validate reserved bit 26 is 0 for Context packets
        // Per VITA 49.2 Table 5.1.1.1-1, bit 26 is Reserved for Context packets (must be 0)
        if (prologue_.header.bit_26) {
            return ValidationError::unsupported_field;
        }

        // 4. Read CIF words (they start at payload_offset)
        size_t offset_words = prologue_.payload_offset / vrt_word_size;

        if ((offset_words + 1) * 4 > buffer_size_) {
            return ValidationError::buffer_too_small;
        }
        context_fields_.cif0 = cif::read_u32_safe(buffer_, offset_words * 4);
        offset_words++;

        if (context_fields_.cif0 & (1U << cif::CIF1_ENABLE_BIT)) {
            if ((offset_words + 1) * 4 > buffer_size_) {
                return ValidationError::buffer_too_small;
            }
            context_fields_.cif1 = cif::read_u32_safe(buffer_, offset_words * 4);
            offset_words++;
        }

        if (context_fields_.cif0 & (1U << cif::CIF2_ENABLE_BIT)) {
            if ((offset_words + 1) * 4 > buffer_size_) {
                return ValidationError::buffer_too_small;
            }
            context_fields_.cif2 = cif::read_u32_safe(buffer_, offset_words * 4);
            offset_words++;
        }

        if (context_fields_.cif0 & (1U << cif::CIF3_ENABLE_BIT)) {
            if ((offset_words + 1) * 4 > buffer_size_) {
                return ValidationError::buffer_too_small;
            }
            context_fields_.cif3 = cif::read_u32_safe(buffer_, offset_words * 4);
            offset_words++;
        }

        // Store context field base offset (after all CIF words)
        context_fields_.context_base_bytes = offset_words * 4;

        // 5. COMPLETE validation: reject ANY unsupported bits
        if (context_fields_.cif0 & ~cif::CIF0_SUPPORTED_MASK) {
            return ValidationError::unsupported_field;
        }

        if (context_fields_.cif0 & (1U << cif::CIF1_ENABLE_BIT)) {
            if (context_fields_.cif1 & ~cif::CIF1_SUPPORTED_MASK) {
                return ValidationError::unsupported_field;
            }
        }

        if (context_fields_.cif0 & (1U << cif::CIF2_ENABLE_BIT)) {
            if (context_fields_.cif2 & ~cif::CIF2_SUPPORTED_MASK) {
                return ValidationError::unsupported_field;
            }
        }

        if (context_fields_.cif0 & (1U << cif::CIF3_ENABLE_BIT)) {
            if (context_fields_.cif3 & ~cif::CIF3_SUPPORTED_MASK) {
                return ValidationError::unsupported_field;
            }
        }

        // 6. Calculate context field sizes with variable field handling
        size_t context_fields_words = 0;

        // Process all fixed fields first (from MSB to LSB)
        for (int bit = 31; bit >= 0; bit--) {
            if (bit == cif::GPS_ASCII_BIT || bit == cif::CONTEXT_ASSOC_BIT)
                continue; // Skip variable fields for now

            if (context_fields_.cif0 & (1U << bit)) {
                context_fields_words += cif::CIF0_FIELDS[bit].size_words;
            }
        }

        // Now handle variable fields IN ORDER (bit 10 before bit 9)
        if (context_fields_.cif0 & (1U << cif::GPS_ASCII_BIT)) {
            context_fields_.gps_ascii.present = true;
            context_fields_.gps_ascii.offset_bytes = (offset_words + context_fields_words) * 4;

            // Read the length from the buffer
            if ((offset_words + context_fields_words + 1) * 4 > buffer_size_) {
                return ValidationError::buffer_too_small;
            }
            context_fields_.gps_ascii.size_words =
                cif::read_gps_ascii_length_words(buffer_, context_fields_.gps_ascii.offset_bytes);

            // Check entire field fits in buffer!
            if ((offset_words + context_fields_words + context_fields_.gps_ascii.size_words) * 4 >
                buffer_size_) {
                return ValidationError::buffer_too_small;
            }

            context_fields_words += context_fields_.gps_ascii.size_words;
        }

        if (context_fields_.cif0 & (1U << cif::CONTEXT_ASSOC_BIT)) {
            context_fields_.context_assoc.present = true;
            context_fields_.context_assoc.offset_bytes = (offset_words + context_fields_words) * 4;

            // Check counts word is present
            if ((offset_words + context_fields_words + 1) * 4 > buffer_size_) {
                return ValidationError::buffer_too_small;
            }

            // Read length with CORRECTED format
            context_fields_.context_assoc.size_words = cif::read_context_assoc_length_words(
                buffer_, context_fields_.context_assoc.offset_bytes);

            // Check entire field fits!
            if ((offset_words + context_fields_words + context_fields_.context_assoc.size_words) *
                    4 >
                buffer_size_) {
                return ValidationError::buffer_too_small;
            }

            context_fields_words += context_fields_.context_assoc.size_words;
        }

        // Process CIF1 fields
        if (context_fields_.cif0 & (1U << cif::CIF1_ENABLE_BIT)) {
            for (int bit = 31; bit >= 0; --bit) {
                if (context_fields_.cif1 & (1U << bit)) {
                    context_fields_words += cif::CIF1_FIELDS[bit].size_words;
                }
            }
        }

        // Process CIF2 fields
        if (context_fields_.cif0 & (1U << cif::CIF2_ENABLE_BIT)) {
            for (int bit = 31; bit >= 0; --bit) {
                if (context_fields_.cif2 & (1U << bit)) {
                    context_fields_words += cif::CIF2_FIELDS[bit].size_words;
                }
            }
        }

        // Process CIF3 fields
        if (context_fields_.cif0 & (1U << cif::CIF3_ENABLE_BIT)) {
            for (int bit = 31; bit >= 0; --bit) {
                if (context_fields_.cif3 & (1U << bit)) {
                    context_fields_words += cif::CIF3_FIELDS[bit].size_words;
                }
            }
        }

        // 7. Calculate total expected size
        // Note: Context packets do not support trailer fields (bit 26 is Reserved)
        context_fields_.calculated_size_words = offset_words + context_fields_words;

        // 8. Final validation: calculated size must match header
        if (context_fields_.calculated_size_words != prologue_.header.size_words) {
            return ValidationError::size_field_mismatch;
        }

        return ValidationError::none;
    }

    // Private constructor - use parse() to construct
    explicit RuntimeContextPacket(std::span<const uint8_t> buffer) noexcept
        : RuntimePacketBase(buffer),
          context_fields_{} {
        error_ = validate_context_packet();
    }

public:
    /**
     * @brief Parse a context packet from raw bytes
     *
     * This is the only way to construct a RuntimeContextPacket. Returns
     * a ParseResult that either contains the valid packet or error information.
     *
     * @param buffer Raw packet bytes
     * @return ParseResult<RuntimeContextPacket> containing either the packet or error
     */
    [[nodiscard]] static ParseResult<RuntimeContextPacket>
    parse(std::span<const uint8_t> buffer) noexcept {
        RuntimeContextPacket packet(buffer);
        if (packet.error_ == ValidationError::none) {
            return packet;
        }

        // Build ParseError with available info
        ParseError err{};
        err.code = packet.error_;
        err.attempted_type = packet.prologue_.header.type;
        err.header = packet.prologue_.header;
        err.raw_bytes = buffer;
        return err;
    }

    // CIF accessors

    uint32_t cif0() const noexcept { return context_fields_.cif0; }
    uint32_t cif1() const noexcept { return context_fields_.cif1; }
    uint32_t cif2() const noexcept { return context_fields_.cif2; }
    uint32_t cif3() const noexcept { return context_fields_.cif3; }

    /// Read the Context Field Change Indicator (CIF0 bit 31)
    /// Returns true if at least one context field has changed since last packet
    /// Returns false if all fields are unchanged from previous packets
    bool change_indicator() const noexcept { return (cif0() & (1U << 31)) != 0; }

    // Field access API support - expose buffer and offsets
    const uint8_t* context_buffer() const noexcept { return buffer_; }

    /// Get byte offset where context fields begin (after CIF words)
    size_t context_base_offset() const noexcept { return context_fields_.context_base_bytes; }

    // Field access via subscript operator
    template <uint8_t CifWord, uint8_t Bit>
    auto operator[](field::field_tag_t<CifWord, Bit> tag) const noexcept
        -> FieldProxy<field::field_tag_t<CifWord, Bit>, const RuntimeContextPacket> {
        return detail::make_field_proxy(*this, tag);
    }
};

} // namespace vrtigo
