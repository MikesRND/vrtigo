// Copyright (c) 2025 Michael Smith
// SPDX-License-Identifier: MIT

#include <array>
#include <chrono>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <vrtigo/vrtigo_utils.hpp>

using namespace vrtigo::utils::netio;

// Import specific types from vrtigo namespace to avoid ambiguity
using vrtigo::NoClassId;
using vrtigo::NoTimestamp;
using vrtigo::NoTrailer;
using vrtigo::PacketVariant;
using vrtigo::UtcRealTimestamp;
using vrtigo::typed::SignalDataPacketBuilder;

// Test fixture for UDP writer tests
class UDPWriterTest : public ::testing::Test {
protected:
    static constexpr uint16_t test_port = 15000;
    static constexpr uint16_t test_port_2 = 15001;
};

// =============================================================================
// Basic Functionality Tests
// =============================================================================

TEST_F(UDPWriterTest, CreateBoundWriter) {
    // Create writer in bound mode
    EXPECT_NO_THROW({
        UDPVRTWriter writer("127.0.0.1", test_port);
        EXPECT_EQ(writer.packets_sent(), 0);
        EXPECT_EQ(writer.bytes_sent(), 0);
    });
}

TEST_F(UDPWriterTest, CreateUnboundWriter) {
    // Create writer in unbound mode
    EXPECT_NO_THROW({
        UDPVRTWriter writer(0); // bind to any port
        EXPECT_EQ(writer.packets_sent(), 0);
        EXPECT_EQ(writer.bytes_sent(), 0);
    });
}

TEST_F(UDPWriterTest, WriteCompileTimePacket) {
    // Create reader to receive packet
    vrtigo::utils::netio::UDPVRTReader<> reader(test_port);
    reader.try_set_timeout(std::chrono::milliseconds(100));

    // Create writer
    UDPVRTWriter writer("127.0.0.1", test_port);

    // Create and send packet
    using PacketType = vrtigo::typed::SignalDataPacketBuilder<64>;
    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    PacketType packet(buffer);
    packet.set_stream_id(0x12345678);
    packet.set_packet_count(1);

    EXPECT_TRUE(writer.write_packet(packet));
    EXPECT_EQ(writer.packets_sent(), 1);
    EXPECT_GT(writer.bytes_sent(), 0);

    // Verify reader can receive it
    auto received = reader.read_next_packet();
    ASSERT_TRUE(received.has_value());
    ASSERT_TRUE(received->ok()) << received->error().message();
    EXPECT_TRUE(vrtigo::is_data_packet(received->value()));

    auto data_pkt = std::get<vrtigo::dynamic::DataPacketView>(received->value());
    EXPECT_EQ(data_pkt.stream_id(), 0x12345678);
}

TEST_F(UDPWriterTest, WriteMultiplePackets) {
    vrtigo::utils::netio::UDPVRTReader<> reader(test_port);
    reader.try_set_timeout(std::chrono::milliseconds(100));

    UDPVRTWriter writer("127.0.0.1", test_port);

    using PacketType = vrtigo::typed::SignalDataPacketBuilder<64>;
    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    // Send 10 packets
    for (uint32_t i = 0; i < 10; i++) {
        PacketType packet(buffer);
        packet.set_stream_id(i);
        packet.set_packet_count(static_cast<uint8_t>(i));
        EXPECT_TRUE(writer.write_packet(packet));
    }

    EXPECT_EQ(writer.packets_sent(), 10);

    // Receive and verify
    for (uint32_t i = 0; i < 10; i++) {
        auto received = reader.read_next_packet();
        ASSERT_TRUE(received.has_value());
        ASSERT_TRUE(received->ok()) << received->error().message();

        auto data_pkt = std::get<vrtigo::dynamic::DataPacketView>(received->value());
        EXPECT_EQ(data_pkt.stream_id(), i);
    }
}

// =============================================================================
// Round-Trip Tests (Writer -> Reader)
// =============================================================================

TEST_F(UDPWriterTest, RoundTripDataPacket) {
    vrtigo::utils::netio::UDPVRTReader<> reader(test_port);
    reader.try_set_timeout(std::chrono::milliseconds(100));

    UDPVRTWriter writer("127.0.0.1", test_port);

    // Create packet with stream ID and timestamp
    using PacketType = vrtigo::typed::SignalDataPacketBuilder<64, UtcRealTimestamp>;
    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    const uint32_t test_stream_id = 0xABCDEF01;
    auto test_timestamp = UtcRealTimestamp::now();

    PacketType packet(buffer);
    packet.set_stream_id(test_stream_id);
    packet.set_timestamp(test_timestamp);
    packet.set_packet_count(1);

    EXPECT_TRUE(writer.write_packet(packet));

    // Read back
    auto received = reader.read_next_packet();
    ASSERT_TRUE(received.has_value());
    ASSERT_TRUE(received->ok()) << received->error().message();

    auto data_pkt = std::get<vrtigo::dynamic::DataPacketView>(received->value());
    EXPECT_EQ(data_pkt.stream_id(), test_stream_id);

    auto ts = data_pkt.timestamp();
    ASSERT_TRUE(ts.has_value());
    EXPECT_EQ(ts->tsi(), test_timestamp.tsi());
    EXPECT_EQ(ts->tsf(), test_timestamp.tsf());
}

TEST_F(UDPWriterTest, RoundTripContextPacket) {
    vrtigo::utils::netio::UDPVRTReader<> reader(test_port);
    reader.try_set_timeout(std::chrono::milliseconds(100));

    UDPVRTWriter writer("127.0.0.1", test_port);

    // Create context packet
    using PacketType = vrtigo::typed::ContextPacketBuilder<NoTimestamp, NoClassId,
                                                           vrtigo::field::reference_point_id>;
    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    const uint32_t test_stream_id = 0x87654321;
    const uint32_t test_ref_point = 0x12345678;

    PacketType packet(buffer);
    packet.set_stream_id(test_stream_id);
    packet[vrtigo::field::reference_point_id].set_encoded(test_ref_point);

    EXPECT_TRUE(writer.write_packet(packet));

    // Read back
    auto received = reader.read_next_packet();
    ASSERT_TRUE(received.has_value());
    ASSERT_TRUE(received->ok()) << received->error().message();
    EXPECT_TRUE(vrtigo::is_context_packet(received->value()));

    auto ctx_pkt = std::get<vrtigo::dynamic::ContextPacketView>(received->value());
    EXPECT_EQ(ctx_pkt.stream_id(), test_stream_id);
}

// Note: InvalidPacket handling test removed - InvalidPacket is no longer part of PacketVariant.
// Parse errors are now represented as ParseResult<PacketVariant> with error() method.

// =============================================================================
// MTU Enforcement Tests
// =============================================================================

TEST_F(UDPWriterTest, EnforceMTU) {
    UDPVRTWriter writer("127.0.0.1", test_port);

    // Set small MTU
    writer.set_mtu(100);

    // Create packet larger than MTU
    using PacketType = vrtigo::typed::SignalDataPacketBuilder<256>;
    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    std::array<uint8_t, 1024> large_payload{};
    PacketType packet(buffer);
    packet.set_stream_id(0x99999999);
    packet.set_packet_count(1);
    packet.set_payload(large_payload.data(), large_payload.size());

    // Should reject due to MTU
    EXPECT_FALSE(writer.write_packet(packet));
    EXPECT_EQ(writer.transport_status().errno_value, EMSGSIZE);
}

TEST_F(UDPWriterTest, MTUAllowsValidPacket) {
    vrtigo::utils::netio::UDPVRTReader<> reader(test_port);
    reader.try_set_timeout(std::chrono::milliseconds(100));

    UDPVRTWriter writer("127.0.0.1", test_port);

    // Set reasonable MTU
    writer.set_mtu(1500);

    // Create packet within MTU
    using PacketType = vrtigo::typed::SignalDataPacketBuilder<64>;
    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    std::array<uint8_t, 256> payload{};
    PacketType packet(buffer);
    packet.set_stream_id(0x11111111);
    packet.set_packet_count(1);
    packet.set_payload(payload.data(), payload.size());

    // Should succeed
    EXPECT_TRUE(writer.write_packet(packet));

    // Verify received
    auto received = reader.read_next_packet();
    ASSERT_TRUE(received.has_value());
    ASSERT_TRUE(received->ok()) << received->error().message();
}

// =============================================================================
// Unbound Mode Tests
// =============================================================================

TEST_F(UDPWriterTest, UnboundModeMultipleDestinations) {
    // Create two readers on different ports
    vrtigo::utils::netio::UDPVRTReader<> reader1(test_port);
    vrtigo::utils::netio::UDPVRTReader<> reader2(test_port_2);
    reader1.try_set_timeout(std::chrono::milliseconds(100));
    reader2.try_set_timeout(std::chrono::milliseconds(100));

    // Create unbound writer
    UDPVRTWriter writer(0);

    // Create packet
    using PacketType = vrtigo::typed::SignalDataPacketBuilder<64>;
    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    PacketType packet(buffer);
    packet.set_stream_id(0x55555555);
    packet.set_packet_count(1);

    // Convert to PacketVariant for the per-destination API
    auto packet_bytes = packet.as_bytes();
    auto parse_result = vrtigo::dynamic::DataPacketView::parse(packet_bytes);
    ASSERT_TRUE(parse_result.ok()) << parse_result.error().message();
    PacketVariant variant = parse_result.value();

    // Send to first destination
    struct sockaddr_in dest1 {};
    dest1.sin_family = AF_INET;
    dest1.sin_port = htons(test_port);
    inet_pton(AF_INET, "127.0.0.1", &dest1.sin_addr);

    EXPECT_TRUE(writer.write_packet(variant, dest1));

    // Send to second destination
    struct sockaddr_in dest2 {};
    dest2.sin_family = AF_INET;
    dest2.sin_port = htons(test_port_2);
    inet_pton(AF_INET, "127.0.0.1", &dest2.sin_addr);

    EXPECT_TRUE(writer.write_packet(variant, dest2));

    EXPECT_EQ(writer.packets_sent(), 2);

    // Verify both readers received
    auto recv1 = reader1.read_next_packet();
    ASSERT_TRUE(recv1.has_value());
    ASSERT_TRUE(recv1->ok()) << recv1->error().message();

    auto recv2 = reader2.read_next_packet();
    ASSERT_TRUE(recv2.has_value());
    ASSERT_TRUE(recv2->ok()) << recv2->error().message();
}

// =============================================================================
// Flush Tests
// =============================================================================

TEST_F(UDPWriterTest, FlushIsNoOp) {
    UDPVRTWriter writer("127.0.0.1", test_port);

    // flush() should always succeed for UDP (no buffering)
    EXPECT_TRUE(writer.flush());

    // Create and send packet
    using PacketType = vrtigo::typed::SignalDataPacketBuilder<64>;
    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    PacketType packet(buffer);
    packet.set_stream_id(0x12345678);
    packet.set_packet_count(1);

    EXPECT_TRUE(writer.write_packet(packet));

    // flush() should still succeed
    EXPECT_TRUE(writer.flush());
}

// =============================================================================
// Move Semantics Tests
// =============================================================================

TEST_F(UDPWriterTest, MoveConstructor) {
    vrtigo::utils::netio::UDPVRTReader<> reader(test_port);
    reader.try_set_timeout(std::chrono::milliseconds(100));

    UDPVRTWriter writer1("127.0.0.1", test_port);

    // Send packet with writer1
    using PacketType = vrtigo::typed::SignalDataPacketBuilder<64>;
    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    PacketType packet(buffer);
    packet.set_stream_id(0x11111111);
    packet.set_packet_count(1);

    EXPECT_TRUE(writer1.write_packet(packet));
    EXPECT_EQ(writer1.packets_sent(), 1);

    // Move to writer2
    UDPVRTWriter writer2(std::move(writer1));
    EXPECT_EQ(writer2.packets_sent(), 1);

    // writer2 should still be usable
    EXPECT_TRUE(writer2.write_packet(packet));
    EXPECT_EQ(writer2.packets_sent(), 2);
}

TEST_F(UDPWriterTest, MoveAssignment) {
    UDPVRTWriter writer1("127.0.0.1", test_port);
    UDPVRTWriter writer2("127.0.0.1", test_port_2);

    using PacketType = vrtigo::typed::SignalDataPacketBuilder<64>;
    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    PacketType packet(buffer);
    packet.set_stream_id(0x22222222);
    packet.set_packet_count(1);

    writer1.write_packet(packet);
    EXPECT_EQ(writer1.packets_sent(), 1);

    // Move assign
    writer2 = std::move(writer1);
    EXPECT_EQ(writer2.packets_sent(), 1);
}
