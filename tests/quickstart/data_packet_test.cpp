// [TITLE]
// Data Packet (Signal Samples)
// [/TITLE]
//
// This test demonstrates a simple VRT data packet creation
// with an incrementing sequence payload and current timestamp.

#include <array>

#include <gtest/gtest.h>
#include <vrtigo.hpp>

// [EXAMPLE]
// Creating a Signal Data Packet
// [/EXAMPLE]

// [DESCRIPTION]
// This example demonstrates creating a simple VRT signal data packet with:
// - UTC timestamp (current time)
// - 8-byte payload
// - Stream identifier
//
// The builder API allows direct packet construction in user-provided buffers.
// [/DESCRIPTION]

TEST(QuickstartSnippet, CreateDataPacket) {
    // [SNIPPET]
    // Create a VRT Signal Data Packet with timestamp and payload

    // Define packet builder type with UTC timestamp
    using PacketType =
        vrtigo::typed::SignalDataPacketBuilder<vrtigo::NoClassId,        // No class ID
                                               vrtigo::UtcRealTimestamp, // Include UTC timestamp
                                               vrtigo::Trailer::none,    // No trailer
                                               2 // Max payload words (8 bytes)
                                               >;

    // Allocate aligned buffer for the packet
    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    // Create simple payload
    std::array<uint8_t, 8> payload{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};

    // Build packet with current timestamp and payload
    PacketType packet(buffer);
    packet.set_stream_id(0x12345678);                      // Set stream identifier
    packet.set_timestamp(vrtigo::UtcRealTimestamp::now()); // Set current time
    packet.set_packet_count(1);                            // First packet in stream
    packet.set_payload(payload.data(), payload.size());    // Attach payload

    // The packet is now ready to transmit
    // You can access fields: packet.stream_id(), packet.timestamp(), etc.
    // [/SNIPPET]

    // Basic verification that the packet was created correctly
    ASSERT_EQ(packet.stream_id(), 0x12345678);
    ASSERT_EQ(packet.packet_count(), 1);
    ASSERT_EQ(packet.payload().size(), 8);

    // Verify payload contents
    auto payload_view = packet.payload();
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(payload_view[i], i + 1);
    }
}