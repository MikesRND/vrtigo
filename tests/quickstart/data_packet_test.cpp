// Quickstart Test: Create Data Packet
//
// This test demonstrates a simple VRT data packet creation
// with an incrementing sequence payload and current timestamp.
//
// The code between snippet markers can be extracted for documentation.

#include <array>

#include <gtest/gtest.h>
#include <vrtigo.hpp>

TEST(QuickstartSnippet, CreateDataPacket) {
    // [QUICKSTART-DESC]
    // This example demonstrates creating a simple VRT signal data packet with:
    // - UTC timestamp (current time)
    // - 8-byte payload
    // - Stream identifier
    //
    // The builder pattern provides a fluent API for packet creation.
    // [/QUICKSTART-DESC]

    // [QUICKSTART]
    // Create a VRT Signal Data Packet with timestamp and payload

    // Define packet type with UTC timestamp
    using PacketType = vrtigo::SignalDataPacket<vrtigo::NoClassId,     // No class ID
                                               vrtigo::TimeStampUTC,  // Include UTC timestamp
                                               vrtigo::Trailer::none, // No trailer
                                               2                     // Max payload words (8 bytes)
                                               >;

    // Allocate aligned buffer for the packet
    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};

    // Create simple payload
    std::array<uint8_t, 8> payload{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};

    // Build packet with current timestamp and payload
    auto packet = vrtigo::PacketBuilder<PacketType>(buffer.data())
                      .stream_id(0x12345678)                 // Set stream identifier
                      .timestamp(vrtigo::TimeStampUTC::now()) // Set current time
                      .packet_count(1)                       // First packet in stream
                      .payload(payload)                      // Attach payload
                      .build();

    // The packet is now ready to transmit
    // You can access fields: packet.stream_id(), packet.timestamp(), etc.
    // [/QUICKSTART]

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