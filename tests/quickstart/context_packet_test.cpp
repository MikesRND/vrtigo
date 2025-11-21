// Quickstart Test: Create Context Packet
//
// This test demonstrates creating a VRT context packet
// with signal parameters like sample rate and bandwidth.
//
// The code between snippet markers can be extracted for documentation.

#include <array>

#include <gtest/gtest.h>
#include <vrtigo.hpp>

TEST(QuickstartSnippet, CreateContextPacket) {
    using namespace vrtigo::field; // For convenient field access

    // [QUICKSTART-DESC]
    // This example demonstrates creating a VRT context packet to describe
    // signal characteristics.
    //
    // [/QUICKSTART-DESC]

    // [QUICKSTART]
    // Create a VRT Context Packet with signal parameters
    using namespace vrtigo::field; // Enable short field syntax

    // Define context packet type with sample rate and bandwidth fields
    using PacketType = vrtigo::ContextPacket<vrtigo::NoTimeStamp, // No timestamp for this example
                                            vrtigo::NoClassId,   // No class ID
                                            sample_rate,        // Include sample rate field
                                            bandwidth           // Include bandwidth field
                                            >;

    // Allocate aligned buffer for the packet
    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};

    // Create context packet
    PacketType packet(buffer.data());

    packet.set_stream_id(0x12345678);
    packet[sample_rate].set_value(100'000'000.0); // 100 MHz sample rate
    packet[bandwidth].set_value(40'000'000.0);    // 40 MHz bandwidth

    // The packet is ready to transmit
    // You can read back values:
    auto fs = packet[sample_rate].value(); // Returns 100'000'000.0 Hz
    auto bw = packet[bandwidth].value();   // Returns 40'000'000.0 Hz
    // [/QUICKSTART]

    // Basic verification that the packet was created correctly
    ASSERT_EQ(packet.stream_id(), 0x12345678);
    ASSERT_DOUBLE_EQ(packet[sample_rate].value(), 100'000'000.0);
    ASSERT_DOUBLE_EQ(packet[bandwidth].value(), 40'000'000.0);

    // Verify we can read the values back
    ASSERT_DOUBLE_EQ(fs, 100'000'000.0);
    ASSERT_DOUBLE_EQ(bw, 40'000'000.0);
}