// [TITLE]
// Data Packet (Signal Samples)
// [/TITLE]
//
// This test demonstrates a simple VRT data packet creation
// with typed sample access and current timestamp.

#include <array>

#include <gtest/gtest.h>
#include <vrtigo.hpp>
#include <vrtigo/sample_span.hpp>

// [EXAMPLE]
// Creating a Signal Data Packet
// [/EXAMPLE]

// [DESCRIPTION]
// This example demonstrates creating a simple VRT signal data packet with:
// - UTC timestamp (current time)
// - 8 real int16_t samples using typed sample access
// - Stream identifier
//
// The builder API allows direct packet construction in user-provided buffers.
// [/DESCRIPTION]

TEST(QuickstartSnippet, CreateDataPacket) {
    // [SNIPPET]
    // Create a VRT Signal Data Packet with 8 real int16_t samples

    // Define packet builder type: 4 words = 16 bytes = 8 int16_t samples
    using PacketType =
        vrtigo::typed::SignalDataPacketBuilder<4,                       // Payload words (16 bytes)
                                               vrtigo::UtcRealTimestamp // Include UTC timestamp
                                               >;

    // Allocate aligned buffer for the packet
    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    // Build packet structure
    PacketType packet(buffer);
    packet.set_stream_id(0x12345678);                      // Set stream identifier
    packet.set_timestamp(vrtigo::UtcRealTimestamp::now()); // Set current time
    packet.set_packet_count(1);                            // First packet in stream

    // Write samples using typed access - handles endian conversion automatically
    auto samples = vrtigo::SampleSpan<int16_t>(packet.payload());
    for (size_t i = 0; i < samples.count(); ++i) {
        samples.set(i, static_cast<int16_t>(i * 100)); // 0, 100, 200, ...
    }

    // The packet is now ready to transmit
    // Read back: samples[0] returns 0, samples[1] returns 100, etc.
    // [/SNIPPET]

    // Basic verification that the packet was created correctly
    ASSERT_EQ(packet.stream_id(), 0x12345678);
    ASSERT_EQ(packet.packet_count(), 1);
    ASSERT_EQ(packet.payload().size(), 16);

    // Verify samples via typed access
    auto view = vrtigo::SampleSpanView<int16_t>(packet.payload());
    ASSERT_EQ(view.count(), 8);
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(view[i], static_cast<int16_t>(i * 100));
    }
}