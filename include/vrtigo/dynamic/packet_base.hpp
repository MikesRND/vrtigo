#pragma once

#include <optional>
#include <span>

#include <vrtigo/class_id.hpp>
#include <vrtigo/timestamp.hpp>
#include <vrtigo/types.hpp>

#include "../detail/buffer_io.hpp"
#include "../detail/header_decode.hpp"
#include "../detail/packet_header_accessor.hpp"

namespace vrtigo::dynamic::detail {

/**
 * Base class for runtime packet parsers
 *
 * Provides common functionality for parsing the generic VRT packet structure:
 * header, stream ID, class ID, and timestamp (the "prologue"). Derived classes
 * handle packet-type-specific fields (payload for data packets, CIF fields for
 * context packets).
 *
 * This is an implementation detail - users should use dynamic::DataPacket or
 * dynamic::ContextPacket directly.
 */
class PacketBase {
protected:
    const uint8_t* buffer_;
    size_t buffer_size_;
    ValidationError error_;

    struct PrologueData {
        vrtigo::detail::DecodedHeader header{};

        // Field offsets (in bytes)
        size_t stream_id_offset = 0;
        size_t class_id_offset = 0;
        size_t tsi_offset = 0;
        size_t tsf_offset = 0;
        size_t payload_offset = 0; // Where packet-specific content begins
    } prologue_;

    /**
     * Protected constructor - only derived classes can construct
     * @param buffer Raw packet bytes
     */
    explicit PacketBase(std::span<const uint8_t> buffer) noexcept
        : buffer_(buffer.data()),
          buffer_size_(buffer.size()),
          error_(ValidationError::none),
          prologue_{} {}

    /**
     * Parse the common packet prologue
     *
     * Parses the header, stream ID, class ID, and timestamp fields.
     * Calculates offsets for all prologue fields and sets payload_offset
     * to where packet-specific content begins.
     *
     * @return ValidationError::none on success, or specific error code
     */
    ValidationError parse_prologue() noexcept {
        // 1. Check minimum buffer size for header
        if (!buffer_ || buffer_size_ < vrt_word_size) {
            return ValidationError::buffer_too_small;
        }

        // 2. Read and decode header
        uint32_t header = vrtigo::detail::read_u32(buffer_, 0);
        prologue_.header = vrtigo::detail::decode_header(header);

        // 3. Validate buffer size against declared packet size
        size_t required_bytes = prologue_.header.size_words * vrt_word_size;
        if (buffer_size_ < required_bytes) {
            return ValidationError::buffer_too_small;
        }

        // 4. Calculate field offsets (in bytes)
        size_t offset_words = 1; // After header

        // Stream ID (presence determined by packet type)
        if (vrtigo::detail::HeaderView::has_stream_id(prologue_.header.type)) {
            prologue_.stream_id_offset = offset_words * vrt_word_size;
            offset_words++;
        }

        // Class ID (64-bit)
        if (prologue_.header.has_class_id) {
            prologue_.class_id_offset = offset_words * vrt_word_size;
            offset_words += 2;
        }

        // Integer timestamp
        if (prologue_.header.tsi != TsiType::none) {
            prologue_.tsi_offset = offset_words * vrt_word_size;
            offset_words++;
        }

        // Fractional timestamp (64-bit)
        if (prologue_.header.tsf != TsfType::none) {
            prologue_.tsf_offset = offset_words * vrt_word_size;
            offset_words += 2;
        }

        // Payload starts here (raw bytes for data, CIF fields for context)
        prologue_.payload_offset = offset_words * vrt_word_size;

        return ValidationError::none;
    }

    // Accessors for derived classes
    const uint8_t* buffer() const noexcept { return buffer_; }

public:
    /**
     * Get header accessor
     * @return Const accessor for header word fields
     */
    vrtigo::detail::HeaderView header() const noexcept {
        return vrtigo::detail::HeaderView{&prologue_.header};
    }

    /**
     * Get packet type
     * @return PacketType decoded from header
     */
    PacketType type() const noexcept { return prologue_.header.type; }

    /**
     * Check if packet has stream ID
     * @return true if packet type includes stream ID field
     */
    bool has_stream_id() const noexcept { return header().has_stream_id(); }

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
     * Check if packet has any timestamp component
     * @return true if TSI != none or TSF != none
     */
    bool has_timestamp() const noexcept { return header().has_timestamp(); }

    /**
     * Get packet count field
     * @return packet count (0-15)
     */
    uint8_t packet_count() const noexcept { return prologue_.header.packet_count; }

    /**
     * Get stream ID
     * @return Stream ID if packet type includes it, otherwise std::nullopt
     */
    std::optional<uint32_t> stream_id() const noexcept {
        if (!has_stream_id()) {
            return std::nullopt;
        }
        return vrtigo::detail::read_u32(buffer_, prologue_.stream_id_offset);
    }

    /**
     * Get class ID
     * @return ClassIdValue if class ID indicator is set, otherwise std::nullopt
     */
    std::optional<ClassIdValue> class_id() const noexcept {
        if (!prologue_.header.has_class_id) {
            return std::nullopt;
        }
        uint32_t word0 = vrtigo::detail::read_u32(buffer_, prologue_.class_id_offset);
        uint32_t word1 = vrtigo::detail::read_u32(buffer_, prologue_.class_id_offset + 4);
        return ClassIdValue::fromWords(word0, word1);
    }

    /**
     * Get timestamp as self-describing TimestampValue
     * @return TimestampValue if timestamp present, otherwise std::nullopt
     */
    std::optional<TimestampValue> timestamp() const noexcept {
        if (!has_timestamp()) {
            return std::nullopt;
        }
        uint32_t tsi_val = (prologue_.header.tsi != TsiType::none)
                               ? vrtigo::detail::read_u32(buffer_, prologue_.tsi_offset)
                               : 0;
        uint64_t tsf_val = (prologue_.header.tsf != TsfType::none)
                               ? vrtigo::detail::read_u64(buffer_, prologue_.tsf_offset)
                               : 0;
        return TimestampValue{tsi_val, tsf_val, prologue_.header.tsi, prologue_.header.tsf};
    }

    /**
     * Get byte offset where payload begins (after prologue)
     * @return Offset in bytes from start of packet
     */
    size_t payload_offset() const noexcept { return prologue_.payload_offset; }

    /**
     * Get packet size in bytes (from header size field)
     * @return Packet size in bytes
     */
    size_t size_bytes() const noexcept { return prologue_.header.size_words * vrt_word_size; }

    /**
     * Get packet size in words (from header size field)
     * @return Packet size in words
     */
    size_t size_words() const noexcept { return prologue_.header.size_words; }

    /**
     * Get buffer size
     * @return Size of buffer in bytes
     */
    size_t buffer_size() const noexcept { return buffer_size_; }

    /**
     * Get entire packet as bytes
     * @return Span of entire packet
     */
    std::span<const uint8_t> as_bytes() const noexcept {
        return std::span<const uint8_t>(buffer_, size_bytes());
    }
};

} // namespace vrtigo::dynamic::detail
