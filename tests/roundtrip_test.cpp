#include <array>

#include <cstring>
#include <gtest/gtest.h>
#include <vrtigo.hpp>

// Test fixture for round-trip tests
class RoundTripTest : public ::testing::Test {
protected:
    // Helper to verify payload data
    template <size_t N>
    void fill_test_payload(std::span<uint8_t, N> payload) {
        for (size_t i = 0; i < payload.size(); ++i) {
            payload[i] = static_cast<uint8_t>((i * 7 + 13) & 0xFF);
        }
    }

    template <size_t N>
    void verify_test_payload(std::span<const uint8_t, N> payload) {
        for (size_t i = 0; i < payload.size(); ++i) {
            EXPECT_EQ(payload[i], static_cast<uint8_t>((i * 7 + 13) & 0xFF))
                << "Payload mismatch at byte " << i;
        }
    }

    template <size_t N>
    void verify_test_payload(std::span<uint8_t, N> payload) {
        verify_test_payload(std::span<const uint8_t, N>(payload));
    }
};

// Test 1: Minimal packet (no stream, no timestamps, no trailer)
TEST_F(RoundTripTest, MinimalPacket) {
    using PacketType =
        vrtigo::SignalDataPacketNoId<vrtigo::NoClassId, vrtigo::NoTimeStamp, // No timestamps
                                    vrtigo::Trailer::none,                 // No trailer
                                    128                                   // 512 bytes payload
                                    >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};

    // Create and populate packet
    PacketType packet(buffer.data());
    packet.set_packet_count(5);
    fill_test_payload(packet.payload());

    // Verify packet header matches compile-time configuration
    EXPECT_EQ(packet.validate(PacketType::size_bytes), vrtigo::ValidationError::none);

    // Parse packet from same buffer (simulates network round-trip)
    PacketType received(buffer.data(), false);

    ASSERT_EQ(received.validate(PacketType::size_bytes), vrtigo::ValidationError::none);

    // Verify fields
    EXPECT_EQ(received.packet_count(), 5);

    // Verify payload
    verify_test_payload(received.payload());
}

// Test 2: Packet with stream ID (type 1)
TEST_F(RoundTripTest, PacketWithStreamId) {
    using PacketType =
        vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::NoTimeStamp, vrtigo::Trailer::none, 256>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};

    // Create packet
    PacketType packet(buffer.data());
    packet.set_stream_id(0x12345678);
    packet.set_packet_count(10);
    fill_test_payload(packet.payload());

    // Parse packet
    PacketType received(buffer.data(), false);

    // Verify fields
    EXPECT_EQ(received.stream_id(), 0x12345678);
    EXPECT_EQ(received.packet_count(), 10);
    verify_test_payload(received.payload());
}

// Test 3: Packet with integer timestamp (TSI)
TEST_F(RoundTripTest, PacketWithIntegerTimestamp) {
    using PacketType =
        vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::TimeStampUTC, // Using UTC timestamps
                                vrtigo::Trailer::none, 128>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};

    // Create packet
    PacketType packet(buffer.data());
    packet.set_stream_id(0xABCDEF00);
    auto ts = vrtigo::TimeStampUTC::from_components(1699000000, 0);
    packet.set_timestamp(ts);
    packet.set_packet_count(7);
    fill_test_payload(packet.payload());

    // Parse packet
    PacketType received(buffer.data(), false);

    // Verify fields
    EXPECT_EQ(received.stream_id(), 0xABCDEF00);
    auto read_ts = received.timestamp();
    EXPECT_EQ(read_ts.seconds(), 1699000000);
    EXPECT_EQ(received.packet_count(), 7);
    verify_test_payload(received.payload());
}

// Test 4: Packet with fractional timestamp (TSF)
TEST_F(RoundTripTest, PacketWithFractionalTimestamp) {
    using PacketType =
        vrtigo::SignalDataPacket<vrtigo::NoClassId,
                                vrtigo::TimeStampUTC, // Using UTC timestamps (with picoseconds)
                                vrtigo::Trailer::none, 256>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};

    // Create packet
    PacketType packet(buffer.data());
    packet.set_stream_id(0xCAFEBABE);
    auto ts = vrtigo::TimeStampUTC::from_components(1234567890,
                                                   999999999999ULL); // Max picoseconds in a second
    packet.set_timestamp(ts);
    packet.set_packet_count(15);
    fill_test_payload(packet.payload());

    // Parse packet
    PacketType received(buffer.data(), false);

    // Verify fields
    EXPECT_EQ(received.stream_id(), 0xCAFEBABE);
    auto read_ts = received.timestamp();
    EXPECT_EQ(read_ts.seconds(), 1234567890);
    EXPECT_EQ(read_ts.fractional(), 999999999999ULL);
    EXPECT_EQ(received.packet_count(), 15);
    verify_test_payload(received.payload());
}

// Test 5: Packet with trailer
TEST_F(RoundTripTest, PacketWithTrailer) {
    using PacketType =
        vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::TimeStampUTC, // Using UTC timestamps
                                vrtigo::Trailer::included,              // Has trailer
                                128>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};

    // Create packet
    PacketType packet(buffer.data());
    packet.set_stream_id(0xDEADBEEF);
    auto ts = vrtigo::TimeStampUTC::from_components(1500000000, 0);
    packet.set_timestamp(ts);
    vrtigo::TrailerBuilder{0x80000001}.apply(packet.trailer()); // Some status bits
    packet.set_packet_count(3);
    fill_test_payload(packet.payload());

    // Parse packet
    PacketType received(buffer.data(), false);

    // Verify fields
    EXPECT_EQ(received.stream_id(), 0xDEADBEEF);
    auto read_ts = received.timestamp();
    EXPECT_EQ(read_ts.seconds(), 1500000000);
    EXPECT_EQ(received.trailer().raw(), 0x80000001);
    EXPECT_EQ(received.packet_count(), 3);
    verify_test_payload(received.payload());
}

// Test 6: Full-featured packet (all optional fields)
TEST_F(RoundTripTest, FullFeaturedPacket) {
    using PacketType =
        vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::TimeStampUTC, // UTC with picoseconds
                                vrtigo::Trailer::included,              // Has trailer
                                512                                    // 2048 bytes payload
                                >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};

    // Create packet with all fields
    PacketType packet(buffer.data());
    packet.set_stream_id(0x01234567);
    auto ts = vrtigo::TimeStampUTC::from_components(1699123456, 123456789012ULL);
    packet.set_timestamp(ts);
    vrtigo::TrailerBuilder{0xF0F0F0F0}.apply(packet.trailer());
    packet.set_packet_count(13);
    fill_test_payload(packet.payload());

    // Parse packet
    PacketType received(buffer.data(), false);

    ASSERT_EQ(received.validate(PacketType::size_bytes), vrtigo::ValidationError::none);

    // Verify all fields
    EXPECT_EQ(received.stream_id(), 0x01234567);
    auto read_ts = received.timestamp();
    EXPECT_EQ(read_ts.seconds(), 1699123456);
    EXPECT_EQ(read_ts.fractional(), 123456789012ULL);
    EXPECT_EQ(received.trailer().raw(), 0xF0F0F0F0);
    EXPECT_EQ(received.packet_count(), 13);
    verify_test_payload(received.payload());
}

// Test 7: Builder round-trip
TEST_F(RoundTripTest, BuilderRoundTrip) {
    using PacketType =
        vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::TimeStampUTC, // UTC with picoseconds
                                vrtigo::Trailer::included,              // Trailer included
                                256>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> tx_buffer;
    alignas(4) std::array<uint8_t, 1024> payload_data;

    // Fill payload
    for (size_t i = 0; i < payload_data.size(); ++i) {
        payload_data[i] = static_cast<uint8_t>(i & 0xFF);
    }

    // Build packet using fluent API
    auto trailer_cfg = vrtigo::TrailerBuilder{}.clear().context_packet_count(1);

    auto ts = vrtigo::TimeStampUTC::from_components(1700000000, 500000000000ULL);
    vrtigo::PacketBuilder<PacketType>(tx_buffer.data())
        .stream_id(0xFEEDFACE)
        .timestamp(ts)
        .trailer(trailer_cfg)
        .packet_count(9)
        .payload(payload_data.data(), payload_data.size())
        .build();

    // Simulate network transmission by copying buffer
    alignas(4) std::array<uint8_t, PacketType::size_bytes> rx_buffer;
    std::memcpy(rx_buffer.data(), tx_buffer.data(), PacketType::size_bytes);

    // Parse received packet
    PacketType received(rx_buffer.data(), false);

    // Verify all fields match
    EXPECT_EQ(received.stream_id(), 0xFEEDFACE);
    auto read_ts = received.timestamp();
    EXPECT_EQ(read_ts.seconds(), 1700000000);
    EXPECT_EQ(read_ts.fractional(), 500000000000ULL);
    // context_packet_count(1) sets count=1 (bit 0) and E bit=1 (bit 7) = 0x81
    EXPECT_EQ(received.trailer().raw(), 0x00000081);
    EXPECT_EQ(received.packet_count(), 9);

    // Verify payload
    auto rx_payload = received.payload();
    for (size_t i = 0; i < rx_payload.size(); ++i) {
        EXPECT_EQ(rx_payload[i], static_cast<uint8_t>(i & 0xFF))
            << "Payload mismatch at byte " << i;
    }
}

// Test 8: Multiple sequential packets
TEST_F(RoundTripTest, MultiplePackets) {
    using PacketType =
        vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::TimeStampUTC, vrtigo::Trailer::none, 128>;

    constexpr size_t NUM_PACKETS = 10;
    alignas(4) std::array<uint8_t, PacketType::size_bytes * NUM_PACKETS> buffer;

    // Create multiple packets
    for (size_t i = 0; i < NUM_PACKETS; ++i) {
        uint8_t* packet_buf = buffer.data() + (i * PacketType::size_bytes);
        PacketType packet(packet_buf);

        packet.set_stream_id(0x1000 + i);
        auto ts = vrtigo::TimeStampUTC::from_components(1600000000 + i * 1000, 0);
        packet.set_timestamp(ts);
        packet.set_packet_count(i);

        // Unique payload per packet
        auto payload = packet.payload();
        for (size_t j = 0; j < payload.size(); ++j) {
            payload[j] = static_cast<uint8_t>((i + j) & 0xFF);
        }
    }

    // Parse and verify all packets
    for (size_t i = 0; i < NUM_PACKETS; ++i) {
        uint8_t* packet_buf = buffer.data() + (i * PacketType::size_bytes);
        PacketType received(packet_buf, false);

        EXPECT_EQ(received.stream_id(), 0x1000 + i);
        auto read_ts = received.timestamp();
        EXPECT_EQ(read_ts.seconds(), 1600000000 + i * 1000);
        EXPECT_EQ(received.packet_count(), i);

        // Verify payload
        auto payload = received.payload();
        for (size_t j = 0; j < payload.size(); ++j) {
            EXPECT_EQ(payload[j], static_cast<uint8_t>((i + j) & 0xFF))
                << "Packet " << i << " payload mismatch at byte " << j;
        }
    }
}

// Test 9: Verify header bits are set correctly
TEST_F(RoundTripTest, HeaderBitsCorrect) {
    using PacketType =
        vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::TimeStampUTC, // UTC with picoseconds
                                vrtigo::Trailer::included,              // Has trailer
                                256>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};
    PacketType packet(buffer.data());

    // Read raw header (network byte order)
    uint32_t raw_header;
    std::memcpy(&raw_header, buffer.data(), 4);
    raw_header = vrtigo::detail::network_to_host32(raw_header);

    // Verify packet type field (bits 31-28) = 1
    EXPECT_EQ((raw_header >> 28) & 0x0F, 1);

    // Verify trailer bit (bit 26) = 1
    EXPECT_EQ((raw_header >> 26) & 0x01, 1);

    // Verify TSI field (bits 23-22) = 1 (UTC)
    EXPECT_EQ((raw_header >> 22) & 0x03, 1);

    // Verify TSF field (bits 21-20) = 2 (real-time)
    EXPECT_EQ((raw_header >> 20) & 0x03, 2);

    // Verify packet size (bits 15-0)
    EXPECT_EQ(raw_header & 0xFFFF, PacketType::size_words);
}

// Test 10: Type 0 packet (no stream ID)
TEST_F(RoundTripTest, Type0PacketNoStreamId) {
    using PacketType =
        vrtigo::SignalDataPacketNoId<vrtigo::NoClassId, vrtigo::TimeStampUTC, // Using UTC timestamps
                                    vrtigo::Trailer::none, 256>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};

    // Create packet
    PacketType packet(buffer.data());
    auto ts = vrtigo::TimeStampUTC::from_components(1234567890, 0);
    packet.set_timestamp(ts);
    packet.set_packet_count(7);
    fill_test_payload(packet.payload());

    // Verify has_stream_id is false
    EXPECT_FALSE(PacketType::has_stream_id);

    // Parse packet
    PacketType received(buffer.data(), false);

    // Verify fields (note: no stream_id() method available)
    auto read_ts = received.timestamp();
    EXPECT_EQ(read_ts.seconds(), 1234567890);
    EXPECT_EQ(received.packet_count(), 7);
    verify_test_payload(received.payload());

    // Verify header packet type is 0
    uint32_t raw_header;
    std::memcpy(&raw_header, buffer.data(), 4);
    raw_header = vrtigo::detail::network_to_host32(raw_header);
    EXPECT_EQ((raw_header >> 28) & 0x0F, 0);
}
