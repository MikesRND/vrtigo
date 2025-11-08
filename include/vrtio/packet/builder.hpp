#pragma once

#include "signal_packet.hpp"
#include "../core/trailer.hpp"
#include <utility>

namespace vrtio {

// Forward declaration
template<typename PacketType>
class PacketBuilder;

// Builder for fluent packet construction on user-provided buffer
// CRITICAL FIX: Now operates directly on user buffer, no internal copy
template<typename PacketType>
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
        requires requires(PacketType& p) { p.set_stream_id(id); } {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.set_stream_id(id);
        return *this;
    }

    // Integer timestamp (only available if packet has TSI)
    auto& timestamp_integer(uint32_t ts) noexcept
        requires requires(PacketType& p) { p.set_timestamp_integer(ts); } {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.set_timestamp_integer(ts);
        return *this;
    }

    // Fractional timestamp (only available if packet has TSF)
    auto& timestamp_fractional(uint64_t ts) noexcept
        requires requires(PacketType& p) { p.set_timestamp_fractional(ts); } {
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
        requires requires(PacketType& p) { p.set_trailer(t); } {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.set_trailer(t);
        return *this;
    }

    // Individual trailer field setters (only available if packet has trailer)

    /**
     * Set trailer valid data indicator
     * @param valid true if data is valid
     */
    auto& trailer_valid_data(bool valid) noexcept
        requires requires(PacketType& p) { p.set_trailer_valid_data(valid); } {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.set_trailer_valid_data(valid);
        return *this;
    }

    /**
     * Set trailer calibrated time indicator
     * @param calibrated true if time is calibrated
     */
    auto& trailer_calibrated_time(bool calibrated) noexcept
        requires requires(PacketType& p) { p.set_trailer_calibrated_time(calibrated); } {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.set_trailer_calibrated_time(calibrated);
        return *this;
    }

    /**
     * Set trailer over-range indicator
     * @param over_range true if over-range occurred
     */
    auto& trailer_over_range(bool over_range) noexcept
        requires requires(PacketType& p) { p.set_trailer_over_range(over_range); } {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.set_trailer_over_range(over_range);
        return *this;
    }

    /**
     * Set trailer sample loss indicator
     * @param loss true if sample loss occurred
     */
    auto& trailer_sample_loss(bool loss) noexcept
        requires requires(PacketType& p) { p.set_trailer_sample_loss(loss); } {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.set_trailer_sample_loss(loss);
        return *this;
    }

    /**
     * Set trailer reference lock indicator
     * @param locked true if reference is locked
     */
    auto& trailer_reference_lock(bool locked) noexcept
        requires requires(PacketType& p) { p.set_trailer_reference_lock(locked); } {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.set_trailer_reference_lock(locked);
        return *this;
    }

    /**
     * Set trailer context packets count
     * @param count Number of context packets (0-127)
     */
    auto& trailer_context_packets(uint8_t count) noexcept
        requires requires(PacketType& p) { p.set_trailer_context_packets(count); } {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.set_trailer_context_packets(count);
        return *this;
    }

    /**
     * Set trailer to indicate good status (valid data and calibrated time)
     */
    auto& trailer_good_status() noexcept
        requires requires(PacketType& p) { p.set_trailer_good_status(); } {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.set_trailer_good_status();
        return *this;
    }

    /**
     * Set trailer AGC/MGC indicator
     * @param active true if AGC/MGC is active
     */
    auto& trailer_agc_mgc(bool active) noexcept
        requires requires(PacketType& p) { p.set_trailer_agc_mgc(active); } {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.set_trailer_agc_mgc(active);
        return *this;
    }

    /**
     * Set trailer detected signal indicator
     * @param detected true if signal is detected
     */
    auto& trailer_detected_signal(bool detected) noexcept
        requires requires(PacketType& p) { p.set_trailer_detected_signal(detected); } {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.set_trailer_detected_signal(detected);
        return *this;
    }

    /**
     * Set trailer spectral inversion indicator
     * @param inverted true if spectral inversion is present
     */
    auto& trailer_spectral_inversion(bool inverted) noexcept
        requires requires(PacketType& p) { p.set_trailer_spectral_inversion(inverted); } {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.set_trailer_spectral_inversion(inverted);
        return *this;
    }

    /**
     * Set trailer reference point indicator
     * @param ref_point true if reference point is indicated
     */
    auto& trailer_reference_point(bool ref_point) noexcept
        requires requires(PacketType& p) { p.set_trailer_reference_point(ref_point); } {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.set_trailer_reference_point(ref_point);
        return *this;
    }

    /**
     * Set trailer signal detected indicator
     * @param detected true if signal is detected
     */
    auto& trailer_signal_detected(bool detected) noexcept
        requires requires(PacketType& p) { p.set_trailer_signal_detected(detected); } {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.set_trailer_signal_detected(detected);
        return *this;
    }

    /**
     * Set multiple trailer status flags at once
     * @param valid_data true if data is valid
     * @param calibrated_time true if time is calibrated
     * @param over_range true if over-range occurred
     * @param sample_loss true if sample loss occurred
     */
    auto& trailer_status(bool valid_data, bool calibrated_time,
                        bool over_range = false, bool sample_loss = false) noexcept
        requires requires(PacketType& p) { p.set_trailer(0); } {
        PacketType packet(buffer_, false);  // Don't reinitialize
        uint32_t t = 0;
        if (valid_data) t |= trailer::valid_data_mask;
        if (calibrated_time) t |= trailer::calibrated_time_mask;
        if (over_range) t |= trailer::over_range_mask;
        if (sample_loss) t |= trailer::sample_loss_mask;
        packet.set_trailer(t);
        return *this;
    }

    // Packet count
    auto& packet_count(uint8_t count) noexcept {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.set_packet_count(count);
        return *this;
    }

    // Payload (from raw pointer and size)
    auto& payload(const uint8_t* data, size_t size) noexcept {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.set_payload(data, size);
        return *this;
    }

    // Payload (from span)
    auto& payload(std::span<const uint8_t> data) noexcept {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.set_payload(data.data(), data.size());
        return *this;
    }

    // Payload (from container)
    template<typename Container>
        requires ConstBuffer<Container>
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