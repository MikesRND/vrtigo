#pragma once

#include "signal_packet.hpp"
#include <utility>

namespace vrtio {

// Forward declaration
template<typename PacketType>
class packet_builder;

// Builder for fluent packet construction on user-provided buffer
// CRITICAL FIX: Now operates directly on user buffer, no internal copy
template<typename PacketType>
class packet_builder {
public:
    // Constructor takes user buffer and initializes it
    explicit packet_builder(uint8_t* buffer) noexcept
        : buffer_(buffer) {
        // Initialize the packet header in the user's buffer
        PacketType packet(buffer_);
    }

    // Stream ID (only available if packet has stream ID)
    auto& stream_id(uint32_t id) noexcept
        requires requires(PacketType p) { p.set_stream_id(id); } {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.set_stream_id(id);
        return *this;
    }

    // Integer timestamp (only available if packet has TSI)
    auto& timestamp_integer(uint32_t ts) noexcept
        requires requires(PacketType p) { p.set_timestamp_integer(ts); } {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.set_timestamp_integer(ts);
        return *this;
    }

    // Fractional timestamp (only available if packet has TSF)
    auto& timestamp_fractional(uint64_t ts) noexcept
        requires requires(PacketType p) { p.set_timestamp_fractional(ts); } {
        PacketType packet(buffer_, false);  // Don't reinitialize
        packet.set_timestamp_fractional(ts);
        return *this;
    }

    // Trailer (only available if packet has trailer)
    auto& trailer(uint32_t t) noexcept
        requires requires(PacketType p) { p.set_trailer(t); } {
        PacketType packet(buffer_, false);  // Don't reinitialize
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
packet_builder(PacketType) -> packet_builder<PacketType>;

// Helper function to create builder
template<typename PacketType>
auto make_builder(uint8_t* buffer) noexcept {
    return packet_builder<PacketType>(buffer);
}

}  // namespace vrtio