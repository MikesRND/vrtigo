// Copyright (c) 2025 Michael Smith
// SPDX-License-Identifier: MIT

#include <array>
#include <chrono>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <vrtigo/utils/detail/reader_error.hpp>
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
    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};

    PacketType packet(buffer);
    packet.set_stream_id(0x12345678);
    packet.set_packet_count(1);

    EXPECT_TRUE(writer.write_packet(packet.as_bytes()));
    EXPECT_EQ(writer.packets_sent(), 1);
    EXPECT_GT(writer.bytes_sent(), 0);

    // Verify reader can receive it
    auto received = reader.read_next_packet();
    ASSERT_TRUE(received.has_value()) << vrtigo::utils::error_message(received.error());
    EXPECT_TRUE(vrtigo::is_data_packet(*received));

    auto data_pkt = std::get<vrtigo::dynamic::DataPacketView>(*received);
    EXPECT_EQ(data_pkt.stream_id(), 0x12345678);
}

TEST_F(UDPWriterTest, WriteMultiplePackets) {
    vrtigo::utils::netio::UDPVRTReader<> reader(test_port);
    reader.try_set_timeout(std::chrono::milliseconds(100));

    UDPVRTWriter writer("127.0.0.1", test_port);

    using PacketType = vrtigo::typed::SignalDataPacketBuilder<64>;
    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};

    // Send 10 packets
    for (uint32_t i = 0; i < 10; i++) {
        PacketType packet(buffer);
        packet.set_stream_id(i);
        packet.set_packet_count(static_cast<uint8_t>(i));
        EXPECT_TRUE(writer.write_packet(packet.as_bytes()));
    }

    EXPECT_EQ(writer.packets_sent(), 10);

    // Receive and verify
    for (uint32_t i = 0; i < 10; i++) {
        auto received = reader.read_next_packet();
        ASSERT_TRUE(received.has_value()) << vrtigo::utils::error_message(received.error());

        auto data_pkt = std::get<vrtigo::dynamic::DataPacketView>(*received);
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
    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};

    const uint32_t test_stream_id = 0xABCDEF01;
    auto test_timestamp = UtcRealTimestamp::now();

    PacketType packet(buffer);
    packet.set_stream_id(test_stream_id);
    packet.set_timestamp(test_timestamp);
    packet.set_packet_count(1);

    EXPECT_TRUE(writer.write_packet(packet.as_bytes()));

    // Read back
    auto received = reader.read_next_packet();
    ASSERT_TRUE(received.has_value()) << vrtigo::utils::error_message(received.error());

    auto data_pkt = std::get<vrtigo::dynamic::DataPacketView>(*received);
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

    EXPECT_TRUE(writer.write_packet(packet.as_bytes()));

    // Read back
    auto received = reader.read_next_packet();
    ASSERT_TRUE(received.has_value()) << vrtigo::utils::error_message(received.error());
    EXPECT_TRUE(vrtigo::is_context_packet(*received));

    auto ctx_pkt = std::get<vrtigo::dynamic::ContextPacketView>(*received);
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
    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};

    std::array<uint8_t, 1024> large_payload{};
    PacketType packet(buffer);
    packet.set_stream_id(0x99999999);
    packet.set_packet_count(1);
    packet.set_payload(large_payload.data(), large_payload.size());

    // Should reject due to MTU
    EXPECT_FALSE(writer.write_packet(packet.as_bytes()));
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
    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};

    std::array<uint8_t, 256> payload{};
    PacketType packet(buffer);
    packet.set_stream_id(0x11111111);
    packet.set_packet_count(1);
    packet.set_payload(payload.data(), payload.size());

    // Should succeed
    EXPECT_TRUE(writer.write_packet(packet.as_bytes()));

    // Verify received
    auto received = reader.read_next_packet();
    ASSERT_TRUE(received.has_value()) << vrtigo::utils::error_message(received.error());
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
    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};

    PacketType packet(buffer);
    packet.set_stream_id(0x55555555);
    packet.set_packet_count(1);

    // Convert to PacketVariant for the per-destination API
    auto packet_bytes = packet.as_bytes();
    auto parse_result = vrtigo::dynamic::DataPacketView::parse(packet_bytes);
    ASSERT_TRUE(parse_result.has_value()) << parse_result.error().message();
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
    ASSERT_TRUE(recv1.has_value()) << vrtigo::utils::error_message(recv1.error());

    auto recv2 = reader2.read_next_packet();
    ASSERT_TRUE(recv2.has_value()) << vrtigo::utils::error_message(recv2.error());
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
    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};

    PacketType packet(buffer);
    packet.set_stream_id(0x12345678);
    packet.set_packet_count(1);

    EXPECT_TRUE(writer.write_packet(packet.as_bytes()));

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
    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};

    PacketType packet(buffer);
    packet.set_stream_id(0x11111111);
    packet.set_packet_count(1);

    EXPECT_TRUE(writer1.write_packet(packet.as_bytes()));
    EXPECT_EQ(writer1.packets_sent(), 1);

    // Move to writer2
    UDPVRTWriter writer2(std::move(writer1));
    EXPECT_EQ(writer2.packets_sent(), 1);

    // writer2 should still be usable
    EXPECT_TRUE(writer2.write_packet(packet.as_bytes()));
    EXPECT_EQ(writer2.packets_sent(), 2);
}

TEST_F(UDPWriterTest, MoveAssignment) {
    UDPVRTWriter writer1("127.0.0.1", test_port);
    UDPVRTWriter writer2("127.0.0.1", test_port_2);

    using PacketType = vrtigo::typed::SignalDataPacketBuilder<64>;
    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};

    PacketType packet(buffer);
    packet.set_stream_id(0x22222222);
    packet.set_packet_count(1);

    writer1.write_packet(packet.as_bytes());
    EXPECT_EQ(writer1.packets_sent(), 1);

    // Move assign
    writer2 = std::move(writer1);
    EXPECT_EQ(writer2.packets_sent(), 1);
}

// =============================================================================
// UDPVRTReader Raw API Tests (read_next_raw, packets_read, transport_status)
// =============================================================================

TEST_F(UDPWriterTest, UDPVRTReaderReadNextRaw) {
    vrtigo::utils::netio::UDPVRTReader<> reader(test_port);
    reader.try_set_timeout(std::chrono::milliseconds(100));

    UDPVRTWriter writer("127.0.0.1", test_port);

    // Create and send test packets
    using PacketType = SignalDataPacketBuilder<64>;
    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};

    // Send 3 packets with different stream IDs
    for (uint32_t i = 0; i < 3; i++) {
        PacketType packet(buffer);
        packet.set_stream_id(0x10000000 + i);
        packet.set_packet_count(static_cast<uint8_t>(i));
        EXPECT_TRUE(writer.write_packet(packet.as_bytes()));
    }

    // Read using read_next_raw() and verify raw bytes
    for (uint32_t i = 0; i < 3; i++) {
        auto raw_bytes = reader.read_next_raw();

        // Verify we got valid data
        ASSERT_FALSE(raw_bytes.empty()) << "Failed to read packet " << i;

        // Verify minimum VRT packet size (at least header)
        EXPECT_GE(raw_bytes.size(), 4u) << "Packet " << i << " too small";

        // Verify word-aligned size
        EXPECT_EQ(raw_bytes.size() % 4, 0u) << "Packet " << i << " not word-aligned";

        // Verify we can extract a valid header
        if (raw_bytes.size() >= 4) {
            uint32_t network_header;
            std::memcpy(&network_header, raw_bytes.data(), 4);
            uint32_t header = vrtigo::detail::network_to_host32(network_header);

            // Verify packet size field matches actual size
            uint16_t size_words = static_cast<uint16_t>(header & 0xFFFF);
            EXPECT_EQ(size_words * 4, raw_bytes.size()) << "Packet " << i << " size mismatch";
        }
    }
}

TEST_F(UDPWriterTest, UDPVRTReaderPacketsReadCounter) {
    vrtigo::utils::netio::UDPVRTReader<> reader(test_port);
    reader.try_set_timeout(std::chrono::milliseconds(100));

    UDPVRTWriter writer("127.0.0.1", test_port);

    using PacketType = SignalDataPacketBuilder<64>;
    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};

    // Initial counter should be 0
    EXPECT_EQ(reader.packets_read(), 0u);

    // Send and read 5 packets
    for (uint32_t i = 0; i < 5; i++) {
        PacketType packet(buffer);
        packet.set_stream_id(0x20000000 + i);
        packet.set_packet_count(static_cast<uint8_t>(i));
        EXPECT_TRUE(writer.write_packet(packet.as_bytes()));

        auto raw_bytes = reader.read_next_raw();
        ASSERT_FALSE(raw_bytes.empty()) << "Failed to read packet " << i;

        // Counter should increment after each successful read
        EXPECT_EQ(reader.packets_read(), i + 1) << "Counter mismatch after packet " << i;
    }

    // Final count should be 5
    EXPECT_EQ(reader.packets_read(), 5u);
}

TEST_F(UDPWriterTest, UDPVRTReaderTransportStatusOnSuccess) {
    vrtigo::utils::netio::UDPVRTReader<> reader(test_port);
    reader.try_set_timeout(std::chrono::milliseconds(100));

    UDPVRTWriter writer("127.0.0.1", test_port);

    using PacketType = SignalDataPacketBuilder<64>;
    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};

    PacketType packet(buffer);
    packet.set_stream_id(0x30303030);
    packet.set_packet_count(42);
    EXPECT_TRUE(writer.write_packet(packet.as_bytes()));

    // Read the packet
    auto raw_bytes = reader.read_next_raw();
    ASSERT_FALSE(raw_bytes.empty());

    // Verify transport status indicates success
    const auto& status = reader.transport_status();
    EXPECT_EQ(status.state, UDPTransportStatus::State::packet_ready);
    EXPECT_EQ(status.bytes_received, raw_bytes.size());
    EXPECT_FALSE(status.is_terminal());
    EXPECT_FALSE(status.is_truncated());
    EXPECT_EQ(status.errno_value, 0);

    // Verify header was decoded correctly
    EXPECT_NE(status.header, 0u);
    EXPECT_EQ(status.packet_type, PacketType::type());
}

TEST_F(UDPWriterTest, UDPVRTReaderRawThenParsed) {
    vrtigo::utils::netio::UDPVRTReader<> reader(test_port);
    reader.try_set_timeout(std::chrono::milliseconds(100));

    UDPVRTWriter writer("127.0.0.1", test_port);

    using PacketType = SignalDataPacketBuilder<64>;
    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};

    // Send first packet - read with read_next_raw()
    PacketType packet1(buffer);
    packet1.set_stream_id(0x40000001);
    packet1.set_packet_count(1);
    EXPECT_TRUE(writer.write_packet(packet1.as_bytes()));

    auto raw_bytes = reader.read_next_raw();
    ASSERT_FALSE(raw_bytes.empty());
    EXPECT_EQ(reader.packets_read(), 1u);

    // Send second packet - read with read_next_packet()
    PacketType packet2(buffer);
    packet2.set_stream_id(0x40000002);
    packet2.set_packet_count(2);
    EXPECT_TRUE(writer.write_packet(packet2.as_bytes()));

    auto received = reader.read_next_packet();
    ASSERT_TRUE(received.has_value()) << vrtigo::utils::error_message(received.error());
    EXPECT_EQ(reader.packets_read(), 2u);

    auto data_pkt = std::get<vrtigo::dynamic::DataPacketView>(*received);
    EXPECT_EQ(data_pkt.stream_id(), 0x40000002);
    EXPECT_EQ(data_pkt.packet_count(), 2);

    // Send third packet - read with read_next_raw() again
    PacketType packet3(buffer);
    packet3.set_stream_id(0x40000003);
    packet3.set_packet_count(3);
    EXPECT_TRUE(writer.write_packet(packet3.as_bytes()));

    auto raw_bytes3 = reader.read_next_raw();
    ASSERT_FALSE(raw_bytes3.empty());
    EXPECT_EQ(reader.packets_read(), 3u);

    // Verify we can still parse the raw bytes manually if needed
    auto parse_result = vrtigo::dynamic::DataPacketView::parse(raw_bytes3);
    ASSERT_TRUE(parse_result.has_value());
    EXPECT_EQ(parse_result->stream_id(), 0x40000003);
    EXPECT_EQ(parse_result->packet_count(), 3);
}
