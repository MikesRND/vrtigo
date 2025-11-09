#pragma once

#include "../core/types.hpp"
#include "../core/endian.hpp"
#include "../core/header.hpp"
#include "../core/concepts.hpp"
#include "../core/trailer.hpp"
#include "../core/trailer_view.hpp"
#include "../core/timestamp.hpp"
#include "../core/timestamp_traits.hpp"
#include "../core/detail/header_decode.hpp"
#include "../core/detail/buffer_io.hpp"
#include "../core/detail/header_init.hpp"
#include <cstring>
#include <cstdio>
#include <span>
#include <stdexcept>

namespace vrtio {

template<
    PacketType Type,
    typename TimeStampType = NoTimeStamp,
    Trailer HasTrailer = Trailer::None,
    size_t PayloadWords = 0
>
    requires (Type == PacketType::SignalDataNoId ||
              Type == PacketType::SignalData ||
              Type == PacketType::ExtensionDataNoId ||
              Type == PacketType::ExtensionData) &&
             ValidPayloadWords<PayloadWords> &&
             ValidTimestampType<TimeStampType>
class DataPacket {
private:
    // Extract TSI and TSF from TimeStampType
    static constexpr tsi_type TSI = TimestampTraits<TimeStampType>::tsi;
    static constexpr tsf_type TSF = TimestampTraits<TimeStampType>::tsf;

public:
    // Timestamp type alias
    using timestamp_type = TimeStampType;

    // Compile-time packet configuration
    static constexpr PacketType packet_type_v = Type;
    static constexpr tsi_type tsi_type_v = TSI;
    static constexpr tsf_type tsf_type_v = TSF;
    static constexpr bool has_trailer = (HasTrailer == Trailer::Included);
    static constexpr size_t payload_words = PayloadWords;

    // Derived constants
    static constexpr bool has_stream_id = (Type == PacketType::SignalData ||
                                           Type == PacketType::ExtensionData);
    static constexpr bool has_timestamp = TimestampTraits<TimeStampType>::has_timestamp;

    // Size calculation (in 32-bit words)
    static constexpr size_t header_words = 1;
    static constexpr size_t stream_id_words = has_stream_id ? 1 : 0;
    static constexpr size_t tsi_words = (TSI != tsi_type::none) ? 1 : 0;
    static constexpr size_t tsf_words = (TSF != tsf_type::none) ? 2 : 0;
    static constexpr size_t trailer_words = (HasTrailer == Trailer::Included) ? 1 : 0;

    static constexpr size_t total_words =
        header_words + stream_id_words + tsi_words +
        tsf_words + PayloadWords + trailer_words;

    // Compile-time check: ensure total packet size fits in 16-bit size field
    static_assert(total_words <= max_packet_words,
                  "Packet size exceeds maximum (65535 words). "
                  "Reduce payload size or remove optional fields.");

    static constexpr size_t size_bytes = total_words * vrt_word_size;
    static constexpr size_t payload_size_bytes = PayloadWords * vrt_word_size;

    // Field offsets (in 32-bit words)
    static constexpr size_t header_offset = 0;
    static constexpr size_t stream_id_offset = header_offset + header_words;
    static constexpr size_t tsi_offset = stream_id_offset + stream_id_words;
    static constexpr size_t tsf_offset = tsi_offset + tsi_words;
    static constexpr size_t payload_offset = tsf_offset + tsf_words;
    static constexpr size_t trailer_offset = payload_offset + PayloadWords;

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
    //   if (packet.validate(buffer_size) == validation_error::none) {
    //       auto id = packet.stream_id();  // Safe after validation
    //   }
    explicit DataPacket(uint8_t* buffer, bool init = true) noexcept
        : buffer_(buffer) {
        if (init) {
            init_header();
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
    uint8_t packet_count() const noexcept {
        return (detail::read_u32(buffer_, header_offset * vrt_word_size) >> 16) & 0x0F;
    }

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
                static_cast<unsigned>(count),
                static_cast<unsigned>(count & 0x0F));
        }
        #endif

        uint32_t header = detail::read_u32(buffer_, header_offset * vrt_word_size);
        header = (header & 0xFFF0FFFF) | ((static_cast<uint32_t>(count) & 0x0F) << 16);
        detail::write_u32(buffer_, header_offset * vrt_word_size, header);
    }

    uint16_t packet_size_words() const noexcept {
        return detail::read_u32(buffer_, header_offset * vrt_word_size) & 0xFFFF;
    }

    // Stream ID accessors (only if has_stream_id)

    uint32_t stream_id() const noexcept requires(has_stream_id) {
        return detail::read_u32(buffer_, stream_id_offset * vrt_word_size);
    }

    void set_stream_id(uint32_t id) noexcept requires(has_stream_id) {
        detail::write_u32(buffer_, stream_id_offset * vrt_word_size, id);
    }

    // Timestamp accessors

    /**
     * Get timestamp as the packet's TimeStampType.
     * Only available when packet has a timestamp type (not NoTimeStamp).
     * @return TimeStampType object containing both integer and fractional parts
     */
    TimeStampType getTimeStamp() const noexcept
        requires(HasTimestamp<TimeStampType>) {
        uint32_t tsi_val = (TSI != tsi_type::none)
            ? detail::read_u32(buffer_, tsi_offset * vrt_word_size) : 0;
        uint64_t tsf_val = (TSF != tsf_type::none)
            ? detail::read_u64(buffer_, tsf_offset * vrt_word_size) : 0;
        return TimeStampType::fromComponents(tsi_val, tsf_val);
    }

    /**
     * Set timestamp from the packet's TimeStampType.
     * Only available when packet has a timestamp type (not NoTimeStamp).
     * @param ts TimeStampType object to set
     */
    void setTimeStamp(const TimeStampType& ts) noexcept
        requires(HasTimestamp<TimeStampType>) {
        if constexpr (TSI != tsi_type::none) {
            detail::write_u32(buffer_, tsi_offset * vrt_word_size, ts.seconds());
        }
        if constexpr (TSF != tsf_type::none) {
            detail::write_u64(buffer_, tsf_offset * vrt_word_size, ts.fractional());
        }
    }

    // Trailer view access

    TrailerView trailer() noexcept requires(HasTrailer == Trailer::Included) {
        return TrailerView(buffer_ + trailer_offset * vrt_word_size);
    }

    ConstTrailerView trailer() const noexcept requires(HasTrailer == Trailer::Included) {
        return ConstTrailerView(buffer_ + trailer_offset * vrt_word_size);
    }

    // Payload access

    std::span<uint8_t, payload_size_bytes> payload() noexcept {
        return std::span<uint8_t, payload_size_bytes>(
            buffer_ + payload_offset * vrt_word_size,
            payload_size_bytes
        );
    }

    std::span<const uint8_t, payload_size_bytes> payload() const noexcept {
        return std::span<const uint8_t, payload_size_bytes>(
            buffer_ + payload_offset * vrt_word_size,
            payload_size_bytes
        );
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
    // Returns: validation_error::none on success, or specific error code
    validation_error validate(size_t buffer_size) const noexcept {
        // Check 1: Buffer must be at least as large as our packet
        if (buffer_size < size_bytes) {
            return validation_error::buffer_too_small;
        }

        // Read and decode header using shared utility
        uint32_t header = detail::read_u32(buffer_, header_offset * vrt_word_size);
        auto decoded = detail::decode_header(header);

        // Check 2: Packet type field (bits 31-28)
        if (decoded.type != Type) {
            return validation_error::packet_type_mismatch;
        }

        // Check 3: Trailer bit (bit 26) - use type-aware field
        if (decoded.trailer_included != (HasTrailer == Trailer::Included)) {
            return validation_error::trailer_bit_mismatch;
        }

        // Check 4: TSI field (bits 23-22)
        if (decoded.tsi != TSI) {
            return validation_error::tsi_mismatch;
        }

        // Check 5: TSF field (bits 21-20)
        if (decoded.tsf != TSF) {
            return validation_error::tsf_mismatch;
        }

        // Check 6: Size field (bits 15-0)
        if (decoded.size_words != total_words) {
            return validation_error::size_field_mismatch;
        }

        return validation_error::none;
    }

private:
    uint8_t* buffer_;  // View over user-provided memory

    // Initialize header with packet metadata
    void init_header() noexcept {
        // Build header using shared helper
        uint32_t header = detail::build_header(
            static_cast<uint8_t>(Type),              // Packet type
            false,                                    // Class ID indicator (not supported for data packets)
            HasTrailer == Trailer::Included,         // Bit 26: Trailer indicator
            false,                                    // Bit 25: Nd0 indicator (V49.0 compatible mode)
            false,                                    // Bit 24: Spectrum/Time (not used for signal/extension data)
            TSI,                                      // TSI field
            TSF,                                      // TSF field
            0,                                        // Packet count (initialized to 0)
            total_words                               // Packet size in words
        );

        detail::write_u32(buffer_, header_offset * vrt_word_size, header);
    }
};

// User-facing type aliases for convenient usage

// Specific aliases that line up with PacketType enum names
template<
    typename TimeStampType = NoTimeStamp,
    Trailer HasTrailer = Trailer::None,
    size_t PayloadWords = 0
>
using SignalDataPacket = DataPacket<PacketType::SignalData, TimeStampType, HasTrailer, PayloadWords>;

template<
    typename TimeStampType = NoTimeStamp,
    Trailer HasTrailer = Trailer::None,
    size_t PayloadWords = 0
>
using SignalDataPacketNoId = DataPacket<PacketType::SignalDataNoId, TimeStampType, HasTrailer, PayloadWords>;

template<
    typename TimeStampType = NoTimeStamp,
    Trailer HasTrailer = Trailer::None,
    size_t PayloadWords = 0
>
using ExtensionDataPacket = DataPacket<PacketType::ExtensionData, TimeStampType, HasTrailer, PayloadWords>;

template<
    typename TimeStampType = NoTimeStamp,
    Trailer HasTrailer = Trailer::None,
    size_t PayloadWords = 0
>
using ExtensionDataPacketNoId = DataPacket<PacketType::ExtensionDataNoId, TimeStampType, HasTrailer, PayloadWords>;

// Deprecated legacy alias retained for source compatibility. Will be removed in a future release.
}  // namespace vrtio
