#pragma once

#include <algorithm>
#include <span>

#include <cassert>
#include <cstring>
#include <vrtigo/class_id.hpp>
#include <vrtigo/timestamp.hpp>
#include <vrtigo/types.hpp>

#include "../detail/concepts.hpp"
#include "../detail/prologue.hpp"
#include "../detail/timestamp_traits.hpp"
#include "../detail/trailer.hpp"
#include "../detail/trailer_view.hpp"
#include "packet_base.hpp"

namespace vrtigo::typed {

template <PacketType Type, size_t PayloadWords, typename TimestampType = NoTimestamp,
          typename ClassIdType = NoClassId, typename TrailerType = NoTrailer>
    requires(Type == PacketType::signal_data_no_id || Type == PacketType::signal_data ||
             Type == PacketType::extension_data_no_id || Type == PacketType::extension_data) &&
            vrtigo::ValidPayloadWords<PayloadWords> && vrtigo::ValidTimestampType<TimestampType> &&
            vrtigo::ValidClassIdType<ClassIdType> && vrtigo::ValidTrailerType<TrailerType>
class DataPacketBuilder
    : public detail::PacketBase<
          DataPacketBuilder<Type, PayloadWords, TimestampType, ClassIdType, TrailerType>,
          vrtigo::Prologue<Type, ClassIdType, TimestampType, false>> {
private:
    using Base = detail::PacketBase<
        DataPacketBuilder<Type, PayloadWords, TimestampType, ClassIdType, TrailerType>,
        vrtigo::Prologue<Type, ClassIdType, TimestampType, false>>;

    friend Base;

public:
    // Inherit type aliases from base
    using typename Base::prologue_type;
    using typename Base::timestamp_type;

    // ========================================================================
    // Static constexpr property functions (packet-specific)
    // Inherited from base: has_stream_id(), has_class_id(), has_timestamp()
    // ========================================================================

    /// Check if packet has trailer
    static constexpr bool has_trailer() noexcept { return TrailerTraits<TrailerType>::has_trailer; }

    /// Get packet type
    static constexpr PacketType type() noexcept { return Type; }

    // ========================================================================
    // Static capacity methods (compile-time, for buffer sizing)
    // ========================================================================

    /// Maximum packet size in 32-bit words (for buffer allocation)
    static constexpr size_t max_size_words() noexcept {
        return prologue_type::size_words() + PayloadWords + (has_trailer() ? 1 : 0);
    }

    /// Maximum packet size in bytes (for buffer allocation)
    static constexpr size_t max_size_bytes() noexcept { return max_size_words() * vrt_word_size; }

    /// Minimum valid packet size in words (prologue + trailer, no payload)
    static constexpr size_t min_size_words() noexcept {
        return prologue_type::size_words() + (has_trailer() ? 1 : 0);
    }

    // ========================================================================
    // Runtime size methods (read from header - may be invalid on malformed packets)
    // Use validate_size() before trusting these values on untrusted buffers.
    // Span accessors (payload(), as_bytes(), trailer()) are always memory-safe.
    // ========================================================================

    /// Get current packet size in 32-bit words (from header)
    size_t size_words() const noexcept { return this->prologue_.packet_size(); }

    /// Get current packet size in bytes (from header)
    size_t size_bytes() const noexcept { return size_words() * vrt_word_size; }

    /// Get current payload size in 32-bit words (from header)
    size_t payload_words() const noexcept {
        size_t sz = size_words();
        size_t overhead = prologue_type::size_words() + (has_trailer() ? 1 : 0);
        // Prevent underflow, but don't clamp upper bound
        return (sz >= overhead) ? (sz - overhead) : 0;
    }

    /// Get current payload size in bytes (from header)
    size_t payload_size_bytes() const noexcept { return payload_words() * vrt_word_size; }

    // Compile-time check: ensure total packet size fits in 16-bit size field
    static_assert(prologue_type::size_words() + PayloadWords +
                          TrailerTraits<TrailerType>::size_words <=
                      max_packet_words,
                  "Packet size exceeds maximum (65535 words). "
                  "Reduce payload size or remove optional fields.");

    // Constructor: creates view over user-provided buffer.
    // If init=true, initializes a new packet at max size; otherwise just wraps existing data.
    //
    // SAFETY WARNING: When init=false (parsing untrusted data), you MUST call
    // validate_size() before trusting size-related values.
    //
    // Example of UNSAFE pattern (DO NOT DO THIS):
    //   DataPacket packet(untrusted_buffer, false);
    //   auto sz = packet.payload_size_bytes();  // May be garbage!
    //
    // Correct pattern:
    //   DataPacket packet(untrusted_buffer, false);
    //   if (packet.validate_size() == ValidationError::none) {
    //       auto sz = packet.payload_size_bytes();  // Safe after validation
    //   }
    //
    // Note: Span accessors (payload(), as_bytes(), trailer()) are always memory-safe
    // due to internal clamping, even on malformed headers.
    explicit DataPacketBuilder(std::span<uint8_t, max_size_bytes()> buffer,
                               bool init = true) noexcept
        : Base(buffer.data()) {
        if (init) {
            init_header();
            init_class_id();
        }
    }

    // ========================================================================
    // Inherited from typed::detail::PacketBase:
    //   - header(), header() const
    //   - packet_count(), set_packet_count()
    //   - stream_id(), set_stream_id() [requires has_stream_id()]
    //   - class_id(), set_class_id() [requires has_class_id()]
    //   - timestamp(), set_timestamp() [requires has_timestamp()]
    //   - as_bytes(), as_bytes() const
    // ========================================================================

    // ========================================================================
    // Validation
    // ========================================================================

    /// Validate header size against buffer capacity
    /// @param buffer_bytes Size of buffer in bytes
    /// @return ValidationError::none if valid
    ValidationError validate_size(size_t buffer_bytes) const noexcept {
        size_t header_size = size_words();
        size_t header_bytes = header_size * vrt_word_size;

        if (header_size < min_size_words()) {
            return ValidationError::size_field_mismatch;
        }
        if (header_size > max_size_words()) {
            return ValidationError::size_field_mismatch;
        }
        if (header_bytes > buffer_bytes) {
            return ValidationError::buffer_too_small;
        }
        return ValidationError::none;
    }

    /// Validate header size against max buffer capacity
    /// @return ValidationError::none if valid
    ValidationError validate_size() const noexcept { return validate_size(max_size_bytes()); }

    // ========================================================================
    // Payload size control
    // ========================================================================

    /// Set current payload size in 32-bit words
    /// @param words Payload size in words (0 to PayloadWords)
    /// @return true if size was set as requested, false if clamped to PayloadWords
    bool set_payload_size(size_t words) noexcept {
        bool valid = (words <= PayloadWords);

        // Debug diagnostic for catching bugs during development
        assert(valid && "Payload size exceeds maximum capacity");

        // Clamp to prevent buffer overrun
        if (!valid) {
            words = PayloadWords;
        }

        size_t total = prologue_type::size_words() + words;
        if constexpr (has_trailer()) {
            total += 1;
        }
        this->prologue_.set_packet_size(static_cast<uint16_t>(total));

        return valid;
    }

    // ========================================================================
    // Trailer view access
    // ========================================================================

    vrtigo::MutableTrailerView trailer() noexcept
        requires HasTrailerType<TrailerType>
    {
        return vrtigo::MutableTrailerView(this->buffer_ + trailer_offset_bytes());
    }

    vrtigo::TrailerView trailer() const noexcept
        requires HasTrailerType<TrailerType>
    {
        return vrtigo::TrailerView(this->buffer_ + trailer_offset_bytes());
    }

    // ========================================================================
    // Payload access
    // ========================================================================

    /// Get mutable payload span (size reflects current packet size, clamped for safety)
    std::span<uint8_t> payload() noexcept {
        return std::span<uint8_t>(this->buffer_ + payload_offset * vrt_word_size,
                                  safe_payload_words() * vrt_word_size);
    }

    /// Get const payload span (size reflects current packet size, clamped for safety)
    std::span<const uint8_t> payload() const noexcept {
        return std::span<const uint8_t>(this->buffer_ + payload_offset * vrt_word_size,
                                        safe_payload_words() * vrt_word_size);
    }

    /// Set payload data (auto-resizes packet, zeros padding bytes)
    /// @param data Payload data to copy
    /// @return true if all data fit, false if truncated to max payload capacity
    bool set_payload(std::span<const uint8_t> data) noexcept {
        // Calculate payload words (round up to word boundary)
        constexpr size_t max_payload_bytes = PayloadWords * vrt_word_size;
        bool truncated = (data.size() > max_payload_bytes);
        size_t payload_bytes = truncated ? max_payload_bytes : data.size();
        size_t words = (payload_bytes + vrt_word_size - 1) / vrt_word_size;
        size_t padded_bytes = words * vrt_word_size;

        // Auto-resize packet to fit data
        set_payload_size(words);

        // Copy data
        uint8_t* dest = this->buffer_ + payload_offset * vrt_word_size;
        std::memcpy(dest, data.data(), payload_bytes);

        // Zero-fill padding bytes to prevent stale data leakage
        if (padded_bytes > payload_bytes) {
            std::memset(dest + payload_bytes, 0, padded_bytes - payload_bytes);
        }

        return !truncated;
    }

    /// Set payload data (legacy overload)
    void set_payload(const uint8_t* data, size_t size) noexcept {
        set_payload(std::span<const uint8_t>(data, size));
    }

private:
    // Internal offsets (payload_offset is static, trailer_offset is dynamic)
    static constexpr size_t payload_offset = prologue_type::payload_offset;

    // Initialize header with packet metadata (uses max_size_words, not runtime size_words)
    void init_header() noexcept { this->prologue_.init_header(max_size_words(), 0, has_trailer()); }

    // Initialize class ID field (zero-initialize, values set via setClassId())
    void init_class_id() noexcept {
        if constexpr (Base::has_class_id()) {
            this->prologue_.init_class_id();
        }
    }

    // ========================================================================
    // Internal safety helpers (clamped sizes for safe span construction)
    // These ensure span accessors never exceed buffer bounds, even on malformed headers.
    // ========================================================================

    /// Clamped payload words - never exceeds PayloadWords
    size_t safe_payload_words() const noexcept {
        size_t raw = payload_words();
        return (raw <= PayloadWords) ? raw : PayloadWords;
    }

    /// Clamped size words - always in [min_size_words, max_size_words]
    size_t safe_size_words() const noexcept {
        size_t raw = size_words();
        if (raw < min_size_words())
            return min_size_words();
        if (raw > max_size_words())
            return max_size_words();
        return raw;
    }

    /// Dynamic trailer offset based on current (clamped) payload size
    size_t trailer_offset_bytes() const noexcept {
        return (payload_offset + safe_payload_words()) * vrt_word_size;
    }
};

// User-facing type aliases for convenient usage
// Template parameter order: PayloadWords (required), TimestampType, ClassIdType, TrailerType

template <size_t PayloadWords, typename TimestampType = NoTimestamp,
          typename ClassIdType = NoClassId, typename TrailerType = NoTrailer>
using SignalDataPacketBuilder = DataPacketBuilder<PacketType::signal_data, PayloadWords,
                                                  TimestampType, ClassIdType, TrailerType>;

template <size_t PayloadWords, typename TimestampType = NoTimestamp,
          typename ClassIdType = NoClassId, typename TrailerType = NoTrailer>
using SignalDataPacketBuilderNoId = DataPacketBuilder<PacketType::signal_data_no_id, PayloadWords,
                                                      TimestampType, ClassIdType, TrailerType>;

template <size_t PayloadWords, typename TimestampType = NoTimestamp,
          typename ClassIdType = NoClassId, typename TrailerType = NoTrailer>
using ExtensionDataPacketBuilder = DataPacketBuilder<PacketType::extension_data, PayloadWords,
                                                     TimestampType, ClassIdType, TrailerType>;

template <size_t PayloadWords, typename TimestampType = NoTimestamp,
          typename ClassIdType = NoClassId, typename TrailerType = NoTrailer>
using ExtensionDataPacketBuilderNoId =
    DataPacketBuilder<PacketType::extension_data_no_id, PayloadWords, TimestampType, ClassIdType,
                      TrailerType>;

} // namespace vrtigo::typed
