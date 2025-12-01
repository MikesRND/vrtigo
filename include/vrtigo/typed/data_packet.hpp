#pragma once

#include <algorithm>
#include <span>

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

template <PacketType Type, typename ClassIdType = NoClassId, typename TimestampType = NoTimestamp,
          Trailer HasTrailer = Trailer::none, size_t PayloadWords = 0>
    requires(Type == PacketType::signal_data_no_id || Type == PacketType::signal_data ||
             Type == PacketType::extension_data_no_id || Type == PacketType::extension_data) &&
            vrtigo::ValidPayloadWords<PayloadWords> && vrtigo::ValidTimestampType<TimestampType> &&
            vrtigo::ValidClassIdType<ClassIdType>
class DataPacketBuilder
    : public detail::PacketBase<
          DataPacketBuilder<Type, ClassIdType, TimestampType, HasTrailer, PayloadWords>,
          vrtigo::Prologue<Type, ClassIdType, TimestampType, false>> {
private:
    using Base = detail::PacketBase<
        DataPacketBuilder<Type, ClassIdType, TimestampType, HasTrailer, PayloadWords>,
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
    static constexpr bool has_trailer() noexcept { return HasTrailer == Trailer::included; }

    /// Get packet type
    static constexpr PacketType type() noexcept { return Type; }

    /// Get payload size in 32-bit words
    static constexpr size_t payload_words() noexcept { return PayloadWords; }

    /// Get payload size in bytes
    static constexpr size_t payload_size_bytes() noexcept { return PayloadWords * vrt_word_size; }

    /// Get packet size in 32-bit words
    static constexpr size_t size_words() noexcept {
        return prologue_type::size_words() + PayloadWords + (has_trailer() ? 1 : 0);
    }

    /// Get packet size in bytes
    static constexpr size_t size_bytes() noexcept { return size_words() * vrt_word_size; }

    // Compile-time check: ensure total packet size fits in 16-bit size field
    static_assert(prologue_type::size_words() + PayloadWords +
                          ((HasTrailer == Trailer::included) ? 1 : 0) <=
                      max_packet_words,
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
    explicit DataPacketBuilder(std::span<uint8_t, size_bytes()> buffer, bool init = true) noexcept
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

    // Trailer view access

    vrtigo::MutableTrailerView trailer() noexcept
        requires(HasTrailer == Trailer::included)
    {
        return vrtigo::MutableTrailerView(this->buffer_ + trailer_offset * vrt_word_size);
    }

    vrtigo::TrailerView trailer() const noexcept
        requires(HasTrailer == Trailer::included)
    {
        return vrtigo::TrailerView(this->buffer_ + trailer_offset * vrt_word_size);
    }

    // Payload access

    std::span<uint8_t, payload_size_bytes()> payload() noexcept {
        return std::span<uint8_t, payload_size_bytes()>(
            this->buffer_ + payload_offset * vrt_word_size, payload_size_bytes());
    }

    std::span<const uint8_t, payload_size_bytes()> payload() const noexcept {
        return std::span<const uint8_t, payload_size_bytes()>(
            this->buffer_ + payload_offset * vrt_word_size, payload_size_bytes());
    }

    void set_payload(const uint8_t* data, size_t size) noexcept {
        auto dest = payload();
        size_t copy_size = std::min(size, dest.size());
        std::memcpy(dest.data(), data, copy_size);
    }

private:
    // Internal offsets
    static constexpr size_t payload_offset = prologue_type::payload_offset;
    static constexpr size_t trailer_offset = payload_offset + PayloadWords;

    // Initialize header with packet metadata
    void init_header() noexcept { this->prologue_.init_header(size_words(), 0, has_trailer()); }

    // Initialize class ID field (zero-initialize, values set via setClassId())
    void init_class_id() noexcept {
        if constexpr (Base::has_class_id()) {
            this->prologue_.init_class_id();
        }
    }
};

// User-facing type aliases for convenient usage

// Specific aliases that line up with PacketType enum names

template <typename ClassIdType = NoClassId, typename TimestampType = NoTimestamp,
          Trailer HasTrailer = Trailer::none, size_t PayloadWords = 0>
using SignalDataPacketBuilder = DataPacketBuilder<PacketType::signal_data, ClassIdType,
                                                  TimestampType, HasTrailer, PayloadWords>;

template <typename ClassIdType = NoClassId, typename TimestampType = NoTimestamp,
          Trailer HasTrailer = Trailer::none, size_t PayloadWords = 0>
using SignalDataPacketBuilderNoId = DataPacketBuilder<PacketType::signal_data_no_id, ClassIdType,
                                                      TimestampType, HasTrailer, PayloadWords>;

template <typename ClassIdType = NoClassId, typename TimestampType = NoTimestamp,
          Trailer HasTrailer = Trailer::none, size_t PayloadWords = 0>
using ExtensionDataPacketBuilder = DataPacketBuilder<PacketType::extension_data, ClassIdType,
                                                     TimestampType, HasTrailer, PayloadWords>;

template <typename ClassIdType = NoClassId, typename TimestampType = NoTimestamp,
          Trailer HasTrailer = Trailer::none, size_t PayloadWords = 0>
using ExtensionDataPacketBuilderNoId =
    DataPacketBuilder<PacketType::extension_data_no_id, ClassIdType, TimestampType, HasTrailer,
                      PayloadWords>;

} // namespace vrtigo::typed
