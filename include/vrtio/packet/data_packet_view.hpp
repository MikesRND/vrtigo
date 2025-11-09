#pragma once

#include "../core/types.hpp"
#include "../core/endian.hpp"
#include "../core/detail/header_decode.hpp"
#include <optional>
#include <span>
#include <cstring>

namespace vrtio {

/**
 * Runtime parser for data packets (signal and extension data)
 *
 * Provides safe, type-erased parsing of data packets with automatic
 * validation. Unlike DataPacket<...>, this class doesn't require compile-time
 * knowledge of the packet structure and automatically validates on construction.
 *
 * Safety:
 * - Validates automatically on construction (no manual validate() call needed)
 * - All accessors return std::optional (safe even if validation failed)
 * - Const-only view (cannot modify packet)
 * - Makes unsafe parsing patterns impossible
 *
 * Usage:
 *   DataPacketView view(rx_buffer, buffer_size);
 *   if (view.is_valid()) {
 *       if (auto id = view.stream_id()) {
 *           std::cout << "Stream ID: " << *id << "\n";
 *       }
 *       auto payload = view.payload();
 *       // Process payload...
 *   }
 */
class DataPacketView {
private:
    const uint8_t* buffer_;
    size_t buffer_size_;
    validation_error error_;

    struct ParsedStructure {
        // Header fields
        PacketType type = PacketType::SignalDataNoId;
        bool has_stream_id = false;
        bool has_trailer = false;
        tsi_type tsi = tsi_type::none;
        tsf_type tsf = tsf_type::none;
        uint16_t packet_size_words = 0;
        uint8_t packet_count = 0;

        // Field offsets (in bytes)
        size_t stream_id_offset = 0;
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
    explicit DataPacketView(const uint8_t* buffer, size_t buffer_size) noexcept
        : buffer_(buffer), buffer_size_(buffer_size), error_(validation_error::none) {
        error_ = validate_internal();
    }

    /**
     * Get validation error
     * @return validation_error::none if packet is valid, otherwise specific error
     */
    validation_error error() const noexcept {
        return error_;
    }

    /**
     * Check if packet is valid
     * @return true if validation passed
     */
    bool is_valid() const noexcept {
        return error_ == validation_error::none;
    }

    /**
     * Get packet type
     * @return PacketType if valid, otherwise SignalDataNoId
     */
    PacketType type() const noexcept {
        return structure_.type;
    }

    /**
     * Check if packet has stream ID
     * @return true if packet type is signal_data_with_stream
     */
    bool has_stream_id() const noexcept {
        return structure_.has_stream_id;
    }

    /**
     * Check if packet has trailer
     * @return true if trailer indicator is set
     */
    bool has_trailer() const noexcept {
        return structure_.has_trailer;
    }

    /**
     * Get TSI type
     * @return TSI type from header
     */
    tsi_type tsi() const noexcept {
        return structure_.tsi;
    }

    /**
     * Get TSF type
     * @return TSF type from header
     */
    tsf_type tsf() const noexcept {
        return structure_.tsf;
    }

    /**
     * Check if packet has integer timestamp
     * @return true if TSI != none
     */
    bool has_timestamp_integer() const noexcept {
        return structure_.tsi != tsi_type::none;
    }

    /**
     * Check if packet has fractional timestamp
     * @return true if TSF != none
     */
    bool has_timestamp_fractional() const noexcept {
        return structure_.tsf != tsf_type::none;
    }

    /**
     * Get packet count field
     * @return packet count (0-15) if valid, otherwise 0
     */
    uint8_t packet_count() const noexcept {
        return structure_.packet_count;
    }

    /**
     * Get stream ID
     * @return Stream ID if packet has stream_id and is valid, otherwise std::nullopt
     */
    std::optional<uint32_t> stream_id() const noexcept {
        if (!is_valid() || !structure_.has_stream_id) {
            return std::nullopt;
        }
        return read_u32(structure_.stream_id_offset);
    }

    /**
     * Get integer timestamp
     * @return Integer timestamp if packet has TSI and is valid, otherwise std::nullopt
     */
    std::optional<uint32_t> timestamp_integer() const noexcept {
        if (!is_valid() || structure_.tsi == tsi_type::none) {
            return std::nullopt;
        }
        return read_u32(structure_.tsi_offset);
    }

    /**
     * Get fractional timestamp
     * @return Fractional timestamp if packet has TSF and is valid, otherwise std::nullopt
     */
    std::optional<uint64_t> timestamp_fractional() const noexcept {
        if (!is_valid() || structure_.tsf == tsf_type::none) {
            return std::nullopt;
        }
        return read_u64(structure_.tsf_offset);
    }

    /**
     * Get trailer
     * @return Trailer word if packet has trailer and is valid, otherwise std::nullopt
     */
    std::optional<uint32_t> trailer() const noexcept {
        if (!is_valid() || !structure_.has_trailer) {
            return std::nullopt;
        }
        return read_u32(structure_.trailer_offset);
    }

    /**
     * Get payload data
     * @return Span of payload bytes if valid, otherwise empty span
     */
    std::span<const uint8_t> payload() const noexcept {
        if (!is_valid()) {
            return {};
        }
        return std::span<const uint8_t>(
            buffer_ + structure_.payload_offset,
            structure_.payload_size_bytes
        );
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
        return structure_.packet_size_words * vrt_word_size;
    }

    /**
     * Get packet size in words (from header size field)
     * @return Packet size in words
     */
    size_t packet_size_words() const noexcept {
        return structure_.packet_size_words;
    }

    /**
     * Get raw buffer pointer (for PacketBase concept compliance)
     * @return Pointer to packet buffer
     */
    const uint8_t* raw_bytes() const noexcept {
        return buffer_;
    }

    /**
     * Get payload size in bytes
     * @return Payload size in bytes
     */
    size_t payload_size_bytes() const noexcept {
        return structure_.payload_size_bytes;
    }

    /**
     * Get payload size in words
     * @return Payload size in words
     */
    size_t payload_size_words() const noexcept {
        return structure_.payload_size_bytes / vrt_word_size;
    }

    /**
     * Get raw buffer pointer
     * @return Pointer to packet buffer
     */
    const uint8_t* data() const noexcept {
        return buffer_;
    }

    /**
     * Get buffer size
     * @return Size of buffer in bytes
     */
    size_t buffer_size() const noexcept {
        return buffer_size_;
    }

private:
    validation_error validate_internal() noexcept {
        // 1. Check minimum buffer size for header
        if (!buffer_ || buffer_size_ < vrt_word_size) {
            return validation_error::buffer_too_small;
        }

        // 2. Read and decode header
        uint32_t header = read_u32(0);
        auto decoded = detail::decode_header(header);

        // 3. Validate packet type (must be signal or extension data)
        if (decoded.type != PacketType::SignalDataNoId &&
            decoded.type != PacketType::SignalData &&
            decoded.type != PacketType::ExtensionDataNoId &&
            decoded.type != PacketType::ExtensionData) {
            return validation_error::packet_type_mismatch;
        }

        // 4. Reject packets with class ID bit set
        // Signal packets don't support class IDs (data_packet.hpp:509 "not used in Phase 1")
        // If we allowed this, the two class ID words would shift all field offsets by 8 bytes,
        // causing stream_id/timestamp/payload/trailer to be misinterpreted
        if (decoded.has_class_id) {
            return validation_error::unsupported_field;
        }

        // 5. Determine stream ID presence from packet type (odd types have stream ID)
        // Per VITA 49.2 Table 5.1.1.1-1, stream ID presence is indicated by packet type:
        // - Type 0 (signal_data_no_stream) → no stream ID field
        // - Type 1 (signal_data_with_stream) → stream ID field present
        // Note: Bit 25 is the "Not a V49.0 Packet Indicator (Nd0)", NOT stream ID indicator!
        bool has_stream_id = detail::has_stream_id_field(decoded.type);

        // 6. Interpret packet-specific bits correctly
        // For Signal/Extension Data packets:
        // - Bit 26 = Trailer indicator
        // - Bit 25 = Nd0 (Not a V49.0 Packet Indicator)
        // - Bit 24 = Spectrum/Time indicator
        // Use the type-aware field from decoded header
        bool has_trailer = decoded.trailer_included;

        // 7. Store header fields
        structure_.type = decoded.type;
        structure_.has_stream_id = has_stream_id;  // Derived from packet type (odd/even)
        structure_.has_trailer = has_trailer;       // Bit 26 for Signal packets
        structure_.tsi = decoded.tsi;
        structure_.tsf = decoded.tsf;
        structure_.packet_size_words = decoded.size_words;
        structure_.packet_count = (header >> 16) & 0x0F;

        // 8. Validate buffer size against declared packet size
        size_t required_bytes = structure_.packet_size_words * vrt_word_size;
        if (buffer_size_ < required_bytes) {
            return validation_error::buffer_too_small;
        }

        // 9. Calculate field offsets (in bytes)
        size_t offset_words = 1;  // After header

        // Stream ID
        if (structure_.has_stream_id) {
            structure_.stream_id_offset = offset_words * vrt_word_size;
            offset_words++;
        }

        // Integer timestamp
        if (structure_.tsi != tsi_type::none) {
            structure_.tsi_offset = offset_words * vrt_word_size;
            offset_words++;
        }

        // Fractional timestamp
        if (structure_.tsf != tsf_type::none) {
            structure_.tsf_offset = offset_words * vrt_word_size;
            offset_words += 2;  // 64-bit field
        }

        // Payload starts here
        structure_.payload_offset = offset_words * vrt_word_size;

        // 10. Calculate payload size
        size_t trailer_words = structure_.has_trailer ? 1 : 0;
        size_t payload_words = structure_.packet_size_words - offset_words - trailer_words;
        structure_.payload_size_bytes = payload_words * vrt_word_size;

        // Trailer offset (if present)
        if (structure_.has_trailer) {
            structure_.trailer_offset = (structure_.packet_size_words - 1) * vrt_word_size;
        }

        // 11. Sanity check: payload size should be non-negative
        if (structure_.packet_size_words < offset_words + trailer_words) {
            return validation_error::size_field_mismatch;
        }

        return validation_error::none;
    }

    // Endianness-aware read helpers
    uint32_t read_u32(size_t byte_offset) const noexcept {
        uint32_t value;
        std::memcpy(&value, buffer_ + byte_offset, sizeof(value));
        return detail::network_to_host32(value);
    }

    uint64_t read_u64(size_t byte_offset) const noexcept {
        uint64_t value;
        std::memcpy(&value, buffer_ + byte_offset, sizeof(value));
        return detail::network_to_host64(value);
    }
};

// User-facing type alias for convenient usage
using SignalPacketView = DataPacketView;

}  // namespace vrtio
