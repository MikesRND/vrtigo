#pragma once

#include "data_packet.hpp"
#include "../core/packet_concepts.hpp"
#include "../core/trailer_view.hpp"
#include <utility>

namespace vrtio {

// Forward declaration
template<typename PacketType>
    requires FixedPacketLike<PacketType>
class PacketBuilder;

/**
 * Builder for fluent packet construction on user-provided buffer
 *
 * Supports fixed-structure compile-time packet types (DataPacket and its aliases).
 * Methods are enabled/disabled based on packet capabilities using C++20 concepts.
 *
 * Usage:
 *   auto packet = PacketBuilder<MyPacketType>(buffer)
 *       .stream_id(0x1234)
 *       .timestamp_integer(ts)
 *       .payload(data, size)
 *       .build();
 *
 * Note: This builder is for fixed-structure packets only. For context packets
 * (ContextPacket), use the field access API: get/set(packet, field::name, value).
 */
template<typename PacketType>
    requires FixedPacketLike<PacketType>
class PacketBuilder {
public:
    // Constructor takes user buffer and initializes it
    explicit PacketBuilder(uint8_t* buffer) noexcept
        : buffer_(buffer) {
        // Initialize the packet header in the user's buffer
        PacketType packet(buffer_);
    }

    // Stream ID (only available if packet has stream ID)
    auto& stream_id(uint32_t id) noexcept
        requires HasStreamId<PacketType> {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.set_stream_id(id);
        return *this;
    }

    // Integer timestamp (only available if packet has TSI)
    auto& timestamp_integer(uint32_t ts) noexcept
        requires HasTimestampInteger<PacketType> {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.set_timestamp_integer(ts);
        return *this;
    }

    // Fractional timestamp (only available if packet has TSF)
    auto& timestamp_fractional(uint64_t ts) noexcept
        requires HasTimestampFractional<PacketType> {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.set_timestamp_fractional(ts);
        return *this;
    }

    // Unified timestamp setter (works with packet's TimeStampType)
    auto& timestamp(const typename PacketType::timestamp_type& ts) noexcept
        requires requires(PacketType& p, typename PacketType::timestamp_type t) {
            p.setTimeStamp(t);
        } {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.setTimeStamp(ts);
        return *this;
    }

    // Trailer (only available if packet has trailer)
    auto& trailer(uint32_t t) noexcept
        requires HasTrailer<PacketType> {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.trailer().set_raw(t);
        return *this;
    }

    auto& trailer(const TrailerBuilder& builder) noexcept
        requires HasTrailer<PacketType> {
        return trailer(builder.value());
    }

    // Individual trailer field setters (only available if packet has trailer)

    /**
     * Set trailer valid data indicator
     * @param valid true if data is valid
     */
    auto& trailer_valid_data(bool valid) noexcept
        requires HasTrailer<PacketType> {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.trailer().set_valid_data(valid);
        return *this;
    }

    /**
     * Set trailer calibrated time indicator
     * @param calibrated true if time is calibrated
     */
    auto& trailer_calibrated_time(bool calibrated) noexcept
        requires HasTrailer<PacketType> {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.trailer().set_calibrated_time(calibrated);
        return *this;
    }

    /**
     * Set trailer over-range indicator
     * @param over_range true if over-range occurred
     */
    auto& trailer_over_range(bool over_range) noexcept
        requires HasTrailer<PacketType> {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.trailer().set_over_range(over_range);
        return *this;
    }

    /**
     * Set trailer sample loss indicator
     * @param loss true if sample loss occurred
     */
    auto& trailer_sample_loss(bool loss) noexcept
        requires HasTrailer<PacketType> {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.trailer().set_sample_loss(loss);
        return *this;
    }

    /**
     * Set trailer reference lock indicator
     * @param locked true if reference is locked
     */
    auto& trailer_reference_lock(bool locked) noexcept
        requires HasTrailer<PacketType> {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.trailer().set_reference_lock(locked);
        return *this;
    }

    /**
     * Set trailer context packets count
     * @param count Number of context packets (0-127)
     */
    auto& trailer_context_packets(uint8_t count) noexcept
        requires HasTrailer<PacketType> {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.trailer().set_context_packets(count);
        return *this;
    }

    /**
     * Set trailer AGC/MGC indicator
     * @param active true if AGC/MGC is active
     */
    auto& trailer_agc_mgc(bool active) noexcept
        requires HasTrailer<PacketType> {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.trailer().set_agc_mgc(active);
        return *this;
    }

    /**
     * Set trailer detected signal indicator
     * @param detected true if signal is detected
     */
    auto& trailer_detected_signal(bool detected) noexcept
        requires HasTrailer<PacketType> {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.trailer().set_detected_signal(detected);
        return *this;
    }

    /**
     * Set trailer spectral inversion indicator
     * @param inverted true if spectral inversion is present
     */
    auto& trailer_spectral_inversion(bool inverted) noexcept
        requires HasTrailer<PacketType> {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.trailer().set_spectral_inversion(inverted);
        return *this;
    }

    /**
     * Set trailer reference point indicator
     * @param ref_point true if reference point is indicated
     */
    auto& trailer_reference_point(bool ref_point) noexcept
        requires HasTrailer<PacketType> {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.trailer().set_reference_point(ref_point);
        return *this;
    }

    /**
     * Set trailer signal detected indicator
     * @param detected true if signal is detected
     */
    auto& trailer_signal_detected(bool detected) noexcept
        requires HasTrailer<PacketType> {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.trailer().set_signal_detected(detected);
        return *this;
    }

    // Packet count (available for all packet types)
    auto& packet_count(uint8_t count) noexcept
        requires HasPacketCount<PacketType> {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.set_packet_count(count);
        return *this;
    }

    // Payload (from raw pointer and size)
    // Only available for packet types that have a payload field
    auto& payload(const uint8_t* data, size_t size) noexcept
        requires HasPayload<PacketType> {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.set_payload(data, size);
        return *this;
    }

    // Payload (from span)
    auto& payload(std::span<const uint8_t> data) noexcept
        requires HasPayload<PacketType> {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.set_payload(data.data(), data.size());
        return *this;
    }

    // Payload (from container)
    template<typename Container>
        requires ConstBuffer<Container> && HasPayload<PacketType>
    auto& payload(const Container& data) noexcept {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.set_payload(
            reinterpret_cast<const uint8_t*>(data.data()),
            data.size()
        );
        return *this;
    }

    // Build: returns a new packet view over the buffer
    PacketType build() noexcept {
        return PacketType(buffer_, false);  // Return view, don't reinitialize
    }

    // Get a packet view (alternative to build())
    PacketType packet() noexcept {
        return PacketType(buffer_, false);  // Return view, don't reinitialize
    }

    // Access to buffer
    std::span<uint8_t, PacketType::size_bytes> as_bytes() noexcept {
        return std::span<uint8_t, PacketType::size_bytes>(buffer_, PacketType::size_bytes);
    }

    std::span<const uint8_t, PacketType::size_bytes> as_bytes() const noexcept {
        return std::span<const uint8_t, PacketType::size_bytes>(buffer_, PacketType::size_bytes);
    }

private:
    uint8_t* buffer_;  // Just store the buffer pointer, no packet copy!
};

// Deduction guide
template<typename PacketType>
PacketBuilder(PacketType) -> PacketBuilder<PacketType>;

// Helper function to create builder
template<typename PacketType>
auto make_builder(uint8_t* buffer) noexcept {
    return PacketBuilder<PacketType>(buffer);
}

}  // namespace vrtio
