#pragma once

#include <span>
#include <stdexcept>

#include <cstdio>
#include <cstring>
#include <vrtio/class_id.hpp>
#include <vrtio/timestamp.hpp>
#include <vrtio/types.hpp>

#include "buffer_io.hpp"
#include "concepts.hpp"
#include "endian.hpp"
#include "header.hpp"
#include "header_decode.hpp"
#include "header_init.hpp"
#include "prologue.hpp"
#include "timestamp_traits.hpp"
#include "trailer.hpp"
#include "trailer_view.hpp"

namespace vrtio {

template <PacketType Type, typename ClassIdType = NoClassId, typename TimeStampType = NoTimeStamp,
          Trailer HasTrailer = Trailer::None, size_t PayloadWords = 0>
    requires(Type == PacketType::SignalDataNoId || Type == PacketType::SignalData ||
             Type == PacketType::ExtensionDataNoId || Type == PacketType::ExtensionData) &&
            ValidPayloadWords<PayloadWords> && ValidTimestampType<TimeStampType> &&
            ValidClassIdType<ClassIdType>
class DataPacket {
private:
    // Use Prologue for common header fields
    using prologue_type = Prologue<Type, ClassIdType, TimeStampType, false>;

public:
    // Type aliases
    using timestamp_type = TimeStampType;

    // Essential packet properties
    static constexpr bool has_stream_id = prologue_type::has_stream_id;
    static constexpr bool has_class_id = prologue_type::has_class_id;
    static constexpr bool has_timestamp = prologue_type::has_timestamp;
    static constexpr bool has_trailer = (HasTrailer == Trailer::Included);

    // Payload configuration
    static constexpr size_t payload_words = PayloadWords;
    static constexpr size_t payload_size_bytes = PayloadWords * vrt_word_size;

    // Total packet size
    static constexpr size_t total_words =
        prologue_type::total_words + PayloadWords + (has_trailer ? 1 : 0);
    static constexpr size_t size_bytes = total_words * vrt_word_size;

    // Compile-time check: ensure total packet size fits in 16-bit size field
    static_assert(total_words <= max_packet_words,
                  "Packet size exceeds maximum (65535 words). "
                  "Reduce payload size or remove optional fields.");

    // Constructor: creates view over user-provided buffer.
    // If init=true, initializes a new packet; otherwise just wraps existing data.
    //
    // SAFETY WARNING: When init=false (parsing untrusted data), you MUST call
    // validate() before accessing packet fields.
    //
    // Example of UNSAFE pattern (DO NOT DO THIS):
    //   DataPacket packet(untrusted_buffer, false);
    //   auto id = packet.stream_id();  // DANGEROUS! No validation!
    //
    // Correct pattern:
    //   DataPacket packet(untrusted_buffer, false);
    //   if (packet.validate(buffer_size) == ValidationError::none) {
    //       auto id = packet.stream_id();  // Safe after validation
    //   }
    explicit DataPacket(uint8_t* buffer, bool init = true) noexcept
        : buffer_(buffer),
          prologue_(buffer) {
        if (init) {
            init_header();
            init_class_id();
        }
    }

    // Prevent copying (packet is a view)
    DataPacket(const DataPacket&) = delete;
    DataPacket& operator=(const DataPacket&) = delete;

    // Allow moving
    DataPacket(DataPacket&&) noexcept = default;
    DataPacket& operator=(DataPacket&&) noexcept = default;

    // Header field accessors (always available)

    // Get packet count (4-bit field: valid range 0-15)
    uint8_t packet_count() const noexcept { return prologue_.packet_count(); }

    // Set packet count (4-bit field: valid range 0-15)
    // NOTE: Values > 15 will be silently truncated to 4 bits.
    // In debug builds, this will trigger an assertion for values > 15.
    void set_packet_count(uint8_t count) noexcept {
#ifndef NDEBUG
        if (count > 15) {
            // Debug build: warn about truncation
            // This is a logic error - packet count should be in range 0-15
            std::fprintf(stderr,
                         "WARNING: packet_count value %u exceeds 4-bit limit (15). "
                         "Value will be truncated to %u.\n",
                         static_cast<unsigned>(count), static_cast<unsigned>(count & 0x0F));
        }
#endif
        prologue_.set_packet_count(count & 0x0F);
    }

    uint16_t packet_size_words() const noexcept { return prologue_.packet_size(); }

    // Stream ID accessors (only if has_stream_id)

    uint32_t stream_id() const noexcept
        requires(has_stream_id)
    {
        return prologue_.stream_id();
    }

    void set_stream_id(uint32_t id) noexcept
        requires(has_stream_id)
    {
        prologue_.set_stream_id(id);
    }

    // Timestamp accessors

    /**
     * Get timestamp as the packet's TimeStampType.
     * Only available when packet has a timestamp type (not NoTimeStamp).
     * @return TimeStampType object containing both integer and fractional parts
     */
    TimeStampType timestamp() const noexcept
        requires(HasTimestamp<TimeStampType>)
    {
        return prologue_.timestamp();
    }

    /**
     * Set timestamp from the packet's TimeStampType.
     * Only available when packet has a timestamp type (not NoTimeStamp).
     * @param ts TimeStampType object to set
     */
    void set_timestamp(TimeStampType ts) noexcept
        requires(HasTimestamp<TimeStampType>)
    {
        prologue_.set_timestamp(ts);
    }

    // Class ID accessors

    /**
     * Get class ID as ClassIdValue.
     * Only available when packet has a class ID type (not NoClassId).
     * @return ClassIdValue object containing OUI, ICC, and PCC
     */
    ClassIdValue class_id() const noexcept
        requires(has_class_id)
    {
        return prologue_.class_id();
    }

    /**
     * Set class ID from ClassIdValue object.
     * Only available when packet has a class ID type (not NoClassId).
     * @param cid ClassIdValue object containing OUI, ICC, PCC, and PBC fields
     */
    void set_class_id(ClassIdValue cid) noexcept
        requires(has_class_id)
    {
        prologue_.set_class_id(cid);
    }

    // Trailer view access

    TrailerView trailer() noexcept
        requires(HasTrailer == Trailer::Included)
    {
        return TrailerView(buffer_ + trailer_offset * vrt_word_size);
    }

    ConstTrailerView trailer() const noexcept
        requires(HasTrailer == Trailer::Included)
    {
        return ConstTrailerView(buffer_ + trailer_offset * vrt_word_size);
    }

    // Payload access

    std::span<uint8_t, payload_size_bytes> payload() noexcept {
        return std::span<uint8_t, payload_size_bytes>(buffer_ + payload_offset * vrt_word_size,
                                                      payload_size_bytes);
    }

    std::span<const uint8_t, payload_size_bytes> payload() const noexcept {
        return std::span<const uint8_t, payload_size_bytes>(
            buffer_ + payload_offset * vrt_word_size, payload_size_bytes);
    }

    void set_payload(const uint8_t* data, size_t size) noexcept {
        auto dest = payload();
        size_t copy_size = std::min(size, dest.size());
        std::memcpy(dest.data(), data, copy_size);
    }

    // Full packet buffer access

    std::span<uint8_t, size_bytes> as_bytes() noexcept {
        return std::span<uint8_t, size_bytes>(buffer_, size_bytes);
    }

    std::span<const uint8_t, size_bytes> as_bytes() const noexcept {
        return std::span<const uint8_t, size_bytes>(buffer_, size_bytes);
    }

    // Validation: verify packet header matches template configuration
    //
    // CRITICAL: You MUST call this method when parsing untrusted data before
    // accessing any packet fields.
    //
    // Returns: ValidationError::none on success, or specific error code
    ValidationError validate(size_t buffer_size) const noexcept {
        // Check 1: Buffer must be at least as large as our packet
        if (buffer_size < size_bytes) {
            return ValidationError::buffer_too_small;
        }

        // Read and decode header using shared utility
        uint32_t header = prologue_.header();
        auto decoded = detail::decode_header(header);

        // Check 2: Packet type field (bits 31-28)
        if (decoded.type != Type) {
            return ValidationError::packet_type_mismatch;
        }

        // Check 3: Class ID bit (bit 27)
        if (decoded.has_class_id != has_class_id) {
            return ValidationError::class_id_bit_mismatch;
        }

        // Check 4: Trailer bit (bit 26) - use type-aware field
        if (decoded.trailer_included != (HasTrailer == Trailer::Included)) {
            return ValidationError::trailer_bit_mismatch;
        }

        // Check 5: TSI field (bits 23-22)
        if (decoded.tsi != prologue_type::tsi) {
            return ValidationError::tsi_mismatch;
        }

        // Check 6: TSF field (bits 21-20)
        if (decoded.tsf != prologue_type::tsf) {
            return ValidationError::tsf_mismatch;
        }

        // Check 7: Size field (bits 15-0)
        if (decoded.size_words != total_words) {
            return ValidationError::size_field_mismatch;
        }

        return ValidationError::none;
    }

private:
    uint8_t* buffer_;                // View over user-provided memory
    mutable prologue_type prologue_; // Mutable to allow const accessors

    // Internal offsets
    static constexpr size_t payload_offset = prologue_type::payload_offset;
    static constexpr size_t trailer_offset = payload_offset + PayloadWords;

    // Initialize header with packet metadata
    void init_header() noexcept {
        prologue_.init_header(total_words, 0, HasTrailer == Trailer::Included);
    }

    // Initialize class ID field (zero-initialize, values set via setClassId())
    void init_class_id() noexcept {
        if constexpr (has_class_id) {
            prologue_.init_class_id();
        }
    }
};

// User-facing type aliases for convenient usage

// Specific aliases that line up with PacketType enum names
template <typename ClassIdType = NoClassId, typename TimeStampType = NoTimeStamp,
          Trailer HasTrailer = Trailer::None, size_t PayloadWords = 0>
using SignalDataPacket =
    DataPacket<PacketType::SignalData, ClassIdType, TimeStampType, HasTrailer, PayloadWords>;

template <typename ClassIdType = NoClassId, typename TimeStampType = NoTimeStamp,
          Trailer HasTrailer = Trailer::None, size_t PayloadWords = 0>
using SignalDataPacketNoId =
    DataPacket<PacketType::SignalDataNoId, ClassIdType, TimeStampType, HasTrailer, PayloadWords>;

template <typename ClassIdType = NoClassId, typename TimeStampType = NoTimeStamp,
          Trailer HasTrailer = Trailer::None, size_t PayloadWords = 0>
using ExtensionDataPacket =
    DataPacket<PacketType::ExtensionData, ClassIdType, TimeStampType, HasTrailer, PayloadWords>;

template <typename ClassIdType = NoClassId, typename TimeStampType = NoTimeStamp,
          Trailer HasTrailer = Trailer::None, size_t PayloadWords = 0>
using ExtensionDataPacketNoId =
    DataPacket<PacketType::ExtensionDataNoId, ClassIdType, TimeStampType, HasTrailer, PayloadWords>;

// Deprecated legacy alias retained for source compatibility. Will be removed in a future release.
} // namespace vrtio
