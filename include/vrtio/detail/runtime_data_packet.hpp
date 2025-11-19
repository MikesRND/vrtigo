#pragma once

#include <optional>
#include <span>

#include <cstring>
#include <vrtio/class_id.hpp>
#include <vrtio/types.hpp>

#include "buffer_io.hpp"
#include "endian.hpp"
#include "header_decode.hpp"
#include "packet_header_accessor.hpp"

namespace vrtio {

/**
 * Runtime parser for data packets (signal and extension data)
 *
 * Provides safe, type-erased parsing of data packets with automatic
 * validation. Unlike DataPacket<...>, this class doesn't require compile-time
 * knowledge of the packet structure and automatically validates on construction.
 *
 * Design Pattern: Runtime Parser vs Compile-Time Template
 * - DataPacket<...>: Compile-time template for building/modifying packets with known structure
 * - RuntimeDataPacket: Runtime parser for reading received packets with unknown structure
 * This separation ensures type safety while allowing flexible packet parsing.
 *
 * Safety:
 * - Validates automatically on construction (no manual validate() call needed)
 * - All accessors return std::optional (safe even if validation failed)
 * - Const-only view (cannot modify packet)
 * - Makes unsafe parsing patterns impossible
 *
 * API Differences from DataPacket:
 * - Returns std::optional for all fields (vs direct values)
 * - timestamp_integer()/timestamp_fractional() instead of timestamp() (no compile-time type)
 * - Read-only access (no setters)
 *
 * Usage:
 *   RuntimeDataPacket view(rx_buffer, buffer_size);
 *   if (view.is_valid()) {
 *       if (auto id = view.stream_id()) {
 *           std::cout << "Stream ID: " << *id << "\n";
 *       }
 *       auto payload = view.payload();
 *       // Process payload...
 *   }
 */
class RuntimeDataPacket {
private:
    const uint8_t* buffer_;
    size_t buffer_size_;
    ValidationError error_;

    struct ParsedStructure {
        // Header data (consolidated from decode_header)
        detail::DecodedHeader header{}; // Value-initialize to zero

        // Additional derived fields not in header
        bool has_stream_id = false; // Derived from packet type

        // Field offsets (in bytes)
        size_t stream_id_offset = 0;
        size_t class_id_offset = 0;
        size_t tsi_offset = 0;
        size_t tsf_offset = 0;
        size_t payload_offset = 0;
        size_t trailer_offset = 0;

        // Sizes
        size_t payload_size_bytes = 0;
    } structure_;

public:
    /**
     * Construct runtime parser and automatically validate
     * @param buffer Pointer to packet buffer
     * @param buffer_size Size of buffer in bytes
     */
    explicit RuntimeDataPacket(const uint8_t* buffer, size_t buffer_size) noexcept
        : buffer_(buffer),
          buffer_size_(buffer_size),
          error_(ValidationError::none),
          structure_{} {
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
     * Get packet type
     * @return PacketType if valid, otherwise SignalDataNoId
     */
    PacketType type() const noexcept { return structure_.header.type; }

    /**
     * Check if packet has stream ID
     * @return true if packet type is signal_data_with_stream
     */
    bool has_stream_id() const noexcept { return structure_.has_stream_id; }

    /**
     * Check if packet has class ID
     * @return true if class ID indicator is set
     */
    bool has_class_id() const noexcept { return header().has_class_id(); }

    /**
     * Check if packet has trailer
     * @return true if trailer indicator is set
     */
    bool has_trailer() const noexcept { return header().has_trailer(); }

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
     * Get packet count field
     * @return packet count (0-15) if valid, otherwise 0
     */
    uint8_t packet_count() const noexcept { return structure_.header.packet_count; }

    /**
     * Get stream ID
     * @return Stream ID if packet has stream_id and is valid, otherwise std::nullopt
     */
    std::optional<uint32_t> stream_id() const noexcept {
        if (!is_valid() || !structure_.has_stream_id) {
            return std::nullopt;
        }
        return detail::read_u32(buffer_, structure_.stream_id_offset);
    }

    /**
     * Get class ID
     * @return ClassIdValue if packet has class_id and is valid, otherwise std::nullopt
     */
    [[nodiscard]] std::optional<ClassIdValue> class_id() const noexcept {
        if (!is_valid() || !structure_.header.has_class_id) {
            return std::nullopt;
        }
        uint32_t word0 = detail::read_u32(buffer_, structure_.class_id_offset);
        uint32_t word1 = detail::read_u32(buffer_, structure_.class_id_offset + 4);
        return ClassIdValue::fromWords(word0, word1);
    }

    /**
     * Get integer timestamp
     * @return Integer timestamp if packet has TSI and is valid, otherwise std::nullopt
     */
    std::optional<uint32_t> timestamp_integer() const noexcept {
        if (!is_valid() || structure_.header.tsi == TsiType::none) {
            return std::nullopt;
        }
        return detail::read_u32(buffer_, structure_.tsi_offset);
    }

    /**
     * Get fractional timestamp
     * @return Fractional timestamp if packet has TSF and is valid, otherwise std::nullopt
     */
    std::optional<uint64_t> timestamp_fractional() const noexcept {
        if (!is_valid() || structure_.header.tsf == TsfType::none) {
            return std::nullopt;
        }
        return detail::read_u64(buffer_, structure_.tsf_offset);
    }

    /**
     * Get trailer
     * @return Trailer word if packet has trailer and is valid, otherwise std::nullopt
     */
    std::optional<uint32_t> trailer() const noexcept {
        if (!is_valid() || !has_trailer()) {
            return std::nullopt;
        }
        return detail::read_u32(buffer_, structure_.trailer_offset);
    }

    /**
     * Get payload data
     * @return Span of payload bytes if valid, otherwise empty span
     */
    std::span<const uint8_t> payload() const noexcept {
        if (!is_valid()) {
            return {};
        }
        return std::span<const uint8_t>(buffer_ + structure_.payload_offset,
                                        structure_.payload_size_bytes);
    }

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

    /**
     * Get packet size in bytes (from header size field)
     * @return Packet size in bytes
     */
    size_t packet_size_bytes() const noexcept {
        return structure_.header.size_words * vrt_word_size;
    }

    /**
     * Get packet size in words (from header size field)
     * @return Packet size in words
     */
    size_t packet_size_words() const noexcept { return structure_.header.size_words; }

    /**
     * Get payload size in bytes
     * @return Payload size in bytes
     */
    size_t payload_size_bytes() const noexcept { return structure_.payload_size_bytes; }

    /**
     * Get payload size in words
     * @return Payload size in words
     */
    size_t payload_size_words() const noexcept {
        return structure_.payload_size_bytes / vrt_word_size;
    }

    /**
     * Get buffer size
     * @return Size of buffer in bytes
     */
    size_t buffer_size() const noexcept { return buffer_size_; }

private:
    ValidationError validate_internal() noexcept {
        // 1. Check minimum buffer size for header
        if (!buffer_ || buffer_size_ < vrt_word_size) {
            return ValidationError::buffer_too_small;
        }

        // 2. Read and decode header
        uint32_t header = detail::read_u32(buffer_, 0);
        auto decoded = detail::decode_header(header);

        // 3. Validate packet type (must be signal or extension data)
        if (decoded.type != PacketType::signal_data_no_id &&
            decoded.type != PacketType::signal_data &&
            decoded.type != PacketType::extension_data_no_id &&
            decoded.type != PacketType::extension_data) {
            return ValidationError::packet_type_mismatch;
        }

        // 4. Determine stream ID presence from packet type (odd types have stream ID)
        // Per VITA 49.2 Table 5.1.1.1-1, stream ID presence is indicated by packet type:
        // - Type 0 (signal_data_no_stream) → no stream ID field
        // - Type 1 (signal_data_with_stream) → stream ID field present
        // Note: Bit 25 is the "Not a V49.0 Packet Indicator (Nd0)", NOT stream ID indicator!
        bool has_stream_id = detail::has_stream_id_field(decoded.type);

        // 5. Store header data
        structure_.header = decoded;
        structure_.has_stream_id = has_stream_id; // Derived from packet type (odd/even)

        // 6. Validate buffer size against declared packet size
        size_t required_bytes = structure_.header.size_words * vrt_word_size;
        if (buffer_size_ < required_bytes) {
            return ValidationError::buffer_too_small;
        }

        // 7. Calculate field offsets (in bytes)
        size_t offset_words = 1; // After header

        // Stream ID
        if (structure_.has_stream_id) {
            structure_.stream_id_offset = offset_words * vrt_word_size;
            offset_words++;
        }

        // Class ID
        if (structure_.header.has_class_id) {
            structure_.class_id_offset = offset_words * vrt_word_size;
            offset_words += 2; // 64-bit class ID
        }

        // Integer timestamp
        if (structure_.header.tsi != TsiType::none) {
            structure_.tsi_offset = offset_words * vrt_word_size;
            offset_words++;
        }

        // Fractional timestamp
        if (structure_.header.tsf != TsfType::none) {
            structure_.tsf_offset = offset_words * vrt_word_size;
            offset_words += 2; // 64-bit field
        }

        // Payload starts here
        structure_.payload_offset = offset_words * vrt_word_size;

        // 8. Calculate payload size
        size_t trailer_words = structure_.header.trailer_included ? 1 : 0;
        size_t payload_words = structure_.header.size_words - offset_words - trailer_words;
        structure_.payload_size_bytes = payload_words * vrt_word_size;

        // Trailer offset (if present)
        if (structure_.header.trailer_included) {
            structure_.trailer_offset = (structure_.header.size_words - 1) * vrt_word_size;
        }

        // 9. Sanity check: payload size should be non-negative
        if (structure_.header.size_words < offset_words + trailer_words) {
            return ValidationError::size_field_mismatch;
        }

        return ValidationError::none;
    }
};

// User-facing type alias for convenient usage
using SignalPacketView = RuntimeDataPacket;

} // namespace vrtio
