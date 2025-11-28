#pragma once

#include <optional>
#include <span>

#include <vrtigo/types.hpp>

#include "parse_result.hpp"
#include "runtime_packet_base.hpp"
#include "trailer_view.hpp"

namespace vrtigo {

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
 * Timestamp Access:
 * - timestamp() returns std::optional<TimestampValue> for unified access
 * - Use TimestampValue::as<TSI, TSF>() to narrow to typed Timestamp when needed
 *
 * Usage:
 *   auto result = RuntimeDataPacket::parse(rx_buffer);  // rx_buffer is std::span<const uint8_t>
 *   if (result.ok()) {
 *       const auto& view = result.value();
 *       if (auto ts = view.timestamp()) {
 *           std::cout << "TSI: " << ts->tsi() << "\n";
 *           if (auto typed = ts->as<TsiType::utc, TsfType::real_time>()) {
 *               auto chrono = typed->to_chrono();
 *           }
 *       }
 *   } else {
 *       std::cerr << "Parse error: " << result.error().message() << "\n";
 *   }
 */
class RuntimeDataPacket : public detail::RuntimePacketBase {
private:
    struct DataFields {
        size_t payload_size_bytes = 0;
        size_t trailer_offset = 0;
    } data_fields_;

    // Private constructor - use parse() to construct
    explicit RuntimeDataPacket(std::span<const uint8_t> buffer) noexcept
        : RuntimePacketBase(buffer),
          data_fields_{} {
        error_ = validate_data_packet();
    }

    ValidationError validate_data_packet() noexcept {
        // 1. Parse common prologue (header, stream_id, class_id, timestamp)
        ValidationError prologue_error = parse_prologue();
        if (prologue_error != ValidationError::none) {
            return prologue_error;
        }

        // 2. Validate packet type (must be signal or extension data)
        if (prologue_.header.type != PacketType::signal_data_no_id &&
            prologue_.header.type != PacketType::signal_data &&
            prologue_.header.type != PacketType::extension_data_no_id &&
            prologue_.header.type != PacketType::extension_data) {
            return ValidationError::packet_type_mismatch;
        }

        // 3. Calculate payload and trailer offsets
        size_t offset_words = prologue_.payload_offset / vrt_word_size;
        size_t trailer_words = prologue_.header.trailer_included ? 1 : 0;
        size_t payload_words = prologue_.header.size_words - offset_words - trailer_words;
        data_fields_.payload_size_bytes = payload_words * vrt_word_size;

        // Trailer offset (if present)
        if (prologue_.header.trailer_included) {
            data_fields_.trailer_offset = (prologue_.header.size_words - 1) * vrt_word_size;
        }

        // 4. Sanity check: payload size should be non-negative
        if (prologue_.header.size_words < offset_words + trailer_words) {
            return ValidationError::size_field_mismatch;
        }

        return ValidationError::none;
    }

public:
    /**
     * @brief Parse a data packet from raw bytes
     *
     * This is the only way to construct a RuntimeDataPacket. Returns
     * a ParseResult that either contains the valid packet or error information.
     *
     * @param buffer Raw packet bytes
     * @return ParseResult<RuntimeDataPacket> containing either the packet or error
     */
    [[nodiscard]] static ParseResult<RuntimeDataPacket>
    parse(std::span<const uint8_t> buffer) noexcept {
        RuntimeDataPacket packet(buffer);
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

    /**
     * Get trailer view
     * @return TrailerView if trailer indicator is set, otherwise std::nullopt
     */
    std::optional<TrailerView> trailer() const noexcept {
        if (!has_trailer()) {
            return std::nullopt;
        }
        return TrailerView(buffer_ + data_fields_.trailer_offset);
    }

    /**
     * Get payload data
     * @return Span of payload bytes
     */
    std::span<const uint8_t> payload() const noexcept {
        return std::span<const uint8_t>(buffer_ + prologue_.payload_offset,
                                        data_fields_.payload_size_bytes);
    }

    /**
     * Get payload size in bytes
     * @return Payload size in bytes
     */
    size_t payload_size_bytes() const noexcept { return data_fields_.payload_size_bytes; }

    /**
     * Get payload size in words
     * @return Payload size in words
     */
    size_t payload_size_words() const noexcept {
        return data_fields_.payload_size_bytes / vrt_word_size;
    }
};

} // namespace vrtigo
