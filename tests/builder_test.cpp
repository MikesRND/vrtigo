#include <array>

#include <gtest/gtest.h>
#include <vrtigo.hpp>

// Test fixture for builder tests
class BuilderTest : public ::testing::Test {
protected:
    template <size_t N>
    void fill_payload(std::array<uint8_t, N>& data) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<uint8_t>(i & 0xFF);
        }
    }
};

// Test 1: Basic builder usage
TEST_F(BuilderTest, BasicBuilder) {
    using PacketType = vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::UtcRealTimestamp,
                                                vrtigo::Trailer::none, 256>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};
    std::array<uint8_t, 1024> payload;
    fill_payload(payload);

    auto ts = vrtigo::UtcRealTimestamp(1699000000, 0);
    auto packet = vrtigo::PacketBuilder<PacketType>(buffer)
                      .stream_id(0x12345678)
                      .timestamp(ts)
                      .packet_count(5)
                      .payload(payload.data(), payload.size())
                      .build();

    EXPECT_EQ(packet.stream_id(), 0x12345678);
    EXPECT_EQ(packet.timestamp().tsi(), 1699000000);
    EXPECT_EQ(packet.packet_count(), 5);
}

// Test 2: Fluent chaining
TEST_F(BuilderTest, FluentChaining) {
    using PacketType = vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::UtcRealTimestamp,
                                                vrtigo::Trailer::included, 128>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    // Single expression with all fields
    auto trailer_cfg = vrtigo::TrailerBuilder{}.clear().context_packet_count(1);

    auto ts = vrtigo::UtcRealTimestamp(1234567890, 500000000000ULL);
    auto packet = vrtigo::PacketBuilder<PacketType>(buffer)
                      .stream_id(0xABCDEF00)
                      .timestamp(ts)
                      .trailer(trailer_cfg)
                      .packet_count(10)
                      .build();

    auto read_ts = packet.timestamp();
    EXPECT_EQ(packet.stream_id(), 0xABCDEF00);
    EXPECT_EQ(read_ts.tsi(), 1234567890);
    EXPECT_EQ(read_ts.tsf(), 500000000000ULL);
    // context_packet_count(1) sets count=1 (bit 0) and E bit=1 (bit 7) = 0x81
    EXPECT_EQ(packet.trailer().raw(), 0x00000081);
    EXPECT_EQ(packet.packet_count(), 10);
}

TEST_F(BuilderTest, TrailerBuilderValueObject) {
    using PacketType = vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::UtcRealTimestamp,
                                                vrtigo::Trailer::included, 64>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    auto trailer_cfg = vrtigo::TrailerBuilder{}
                           .valid_data(true)
                           .calibrated_time(true)
                           .context_packet_count(7)
                           .reference_lock(true);

    auto packet = vrtigo::PacketBuilder<PacketType>(buffer)
                      .stream_id(0x0BADBEEF)
                      .trailer(trailer_cfg)
                      .packet_count(2)
                      .build();

    EXPECT_TRUE(packet.trailer().valid_data());
    EXPECT_TRUE(packet.trailer().calibrated_time());
    EXPECT_EQ(packet.trailer().context_packet_count(), 7);
    EXPECT_TRUE(packet.trailer().reference_lock());
}

TEST_F(BuilderTest, TrailerBuilderChaining) {
    using PacketType = vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::UtcRealTimestamp,
                                                vrtigo::Trailer::included, 64>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    auto trailer_cfg = vrtigo::TrailerBuilder{}
                           .clear()
                           .valid_data(true)
                           .calibrated_time(true)
                           .context_packet_count(3)
                           .over_range(true);

    auto packet = vrtigo::PacketBuilder<PacketType>(buffer)
                      .stream_id(0x10203040)
                      .trailer(trailer_cfg)
                      .packet_count(4)
                      .build();

    EXPECT_TRUE(packet.trailer().valid_data());
    EXPECT_TRUE(packet.trailer().calibrated_time());
    EXPECT_EQ(packet.trailer().context_packet_count(), 3);
    EXPECT_TRUE(packet.trailer().over_range());
    EXPECT_EQ(packet.packet_count(), 4);
}

// Explicitly verify the legacy raw-literal setter remains available
TEST_F(BuilderTest, RawTrailerLiteralStillSupported) {
    using PacketType = vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::UtcRealTimestamp,
                                                vrtigo::Trailer::included, 32>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    auto packet =
        vrtigo::PacketBuilder<PacketType>(buffer).stream_id(0x1234ABCD).trailer(0xA5A5A5A5).build();

    EXPECT_EQ(packet.trailer().raw(), 0xA5A5A5A5);
}

// Test 3: Builder with span payload
TEST_F(BuilderTest, BuilderWithSpan) {
    using PacketType = vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::NoTimestamp,
                                                vrtigo::Trailer::none, 256>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};
    std::array<uint8_t, 1024> payload_data;
    fill_payload(payload_data);

    std::span<const uint8_t> payload_span(payload_data);

    auto packet = vrtigo::PacketBuilder<PacketType>(buffer)
                      .stream_id(0xCAFEBABE)
                      .payload(payload_span)
                      .build();

    EXPECT_EQ(packet.stream_id(), 0xCAFEBABE);

    // Verify payload
    auto rx_payload = packet.payload();
    for (size_t i = 0; i < rx_payload.size(); ++i) {
        EXPECT_EQ(rx_payload[i], static_cast<uint8_t>(i & 0xFF));
    }
}

// Test 4: Builder with container payload
TEST_F(BuilderTest, BuilderWithContainer) {
    using PacketType = vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::UtcRealTimestamp,
                                                vrtigo::Trailer::none, 128>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};
    std::vector<uint8_t> payload_vector(512);
    for (size_t i = 0; i < payload_vector.size(); ++i) {
        payload_vector[i] = static_cast<uint8_t>((i * 3) & 0xFF);
    }

    auto ts = vrtigo::UtcRealTimestamp(1500000000, 0);
    auto packet = vrtigo::PacketBuilder<PacketType>(buffer)
                      .stream_id(0xDEADBEEF)
                      .timestamp(ts)
                      .payload(payload_vector)
                      .build();

    EXPECT_EQ(packet.stream_id(), 0xDEADBEEF);

    // Verify payload
    auto rx_payload = packet.payload();
    for (size_t i = 0; i < rx_payload.size(); ++i) {
        EXPECT_EQ(rx_payload[i], static_cast<uint8_t>((i * 3) & 0xFF));
    }
}

// Test 5: Partial builder (not all optional fields)
TEST_F(BuilderTest, PartialBuilder) {
    using PacketType = vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::UtcRealTimestamp,
                                                vrtigo::Trailer::included, 256>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    // Only set some fields
    auto ts = vrtigo::UtcRealTimestamp(1600000000, 0);
    auto packet = vrtigo::PacketBuilder<PacketType>(buffer)
                      .stream_id(0x11111111)
                      .timestamp(ts)
                      // Skip trailer
                      .packet_count(7)
                      .build();

    EXPECT_EQ(packet.stream_id(), 0x11111111);
    EXPECT_EQ(packet.timestamp().tsi(), 1600000000);
    EXPECT_EQ(packet.packet_count(), 7);
    // Trailer will be zero (default initialized)
}

// Test 6: Builder returns reference (no copy)
TEST_F(BuilderTest, BuilderReturnsReference) {
    using PacketType = vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::NoTimestamp,
                                                vrtigo::Trailer::none, 128>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    auto packet1 = vrtigo::PacketBuilder<PacketType>(buffer).stream_id(0x12345678).build();

    // Modify through reference
    packet1.set_stream_id(0x87654321);

    // Create another view of same buffer
    PacketType packet2(buffer);

    // Should see the modification
    EXPECT_EQ(packet2.stream_id(), 0x87654321);
}

// Test 7: as_bytes() method
TEST_F(BuilderTest, AsBytesMethod) {
    using PacketType = vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::UtcRealTimestamp,
                                                vrtigo::Trailer::none, 128>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    auto ts = vrtigo::UtcRealTimestamp(1700000000, 0);
    vrtigo::PacketBuilder<PacketType> builder(buffer);
    builder.stream_id(0xFEEDFACE).timestamp(ts);

    // Get bytes from builder
    auto bytes = builder.as_bytes();

    EXPECT_EQ(bytes.size(), PacketType::size_bytes());
    EXPECT_EQ(bytes.data(), buffer.data());
}

// Test 8: make_builder helper
TEST_F(BuilderTest, MakeBuilderHelper) {
    using PacketType = vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::UtcRealTimestamp,
                                                vrtigo::Trailer::none, 256>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    auto ts = vrtigo::UtcRealTimestamp(1800000000, 0);
    auto builder = vrtigo::make_builder<PacketType>(buffer);
    auto packet = builder.stream_id(0x99999999).timestamp(ts).build();

    EXPECT_EQ(packet.stream_id(), 0x99999999);
    EXPECT_EQ(packet.timestamp().tsi(), 1800000000);
}

// Test 9: Builder without optional fields (Type 0)
TEST_F(BuilderTest, BuilderType0NoStream) {
    using PacketType = vrtigo::SignalDataPacketNoId<vrtigo::NoClassId, vrtigo::UtcRealTimestamp,
                                                    vrtigo::Trailer::none, 128>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    // Builder for type 0 - no stream_id() method
    auto ts = vrtigo::UtcRealTimestamp(1234567890, 0);
    auto packet = vrtigo::PacketBuilder<PacketType>(buffer).timestamp(ts).packet_count(3).build();

    EXPECT_EQ(packet.timestamp().tsi(), 1234567890);
    EXPECT_EQ(packet.packet_count(), 3);
    EXPECT_FALSE(PacketType::has_stream_id());
}

// Test 10: Multiple builders on different buffers
TEST_F(BuilderTest, MultipleBuilders) {
    using PacketType = vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::UtcRealTimestamp,
                                                vrtigo::Trailer::none, 128>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer1;
    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer2;

    auto ts1 = vrtigo::UtcRealTimestamp(1000000000, 0);
    auto packet1 =
        vrtigo::PacketBuilder<PacketType>(buffer1).stream_id(0x11111111).timestamp(ts1).build();

    auto ts2 = vrtigo::UtcRealTimestamp(2000000000, 0);
    auto packet2 =
        vrtigo::PacketBuilder<PacketType>(buffer2).stream_id(0x22222222).timestamp(ts2).build();

    EXPECT_EQ(packet1.stream_id(), 0x11111111);
    EXPECT_EQ(packet1.timestamp().tsi(), 1000000000);

    EXPECT_EQ(packet2.stream_id(), 0x22222222);
    EXPECT_EQ(packet2.timestamp().tsi(), 2000000000);

    // Packets should be independent
    packet1.set_stream_id(0x33333333);
    EXPECT_EQ(packet1.stream_id(), 0x33333333);
    EXPECT_EQ(packet2.stream_id(), 0x22222222); // Unchanged
}

// Test 11: Builder build() method returns reference
TEST_F(BuilderTest, BuilderBuildMethod) {
    using PacketType = vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::NoTimestamp,
                                                vrtigo::Trailer::none, 128>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    vrtigo::PacketBuilder<PacketType> builder(buffer);
    builder.stream_id(0xAAAAAAAA);

    // Build and get packet reference
    auto packet_ref = builder.build();
    EXPECT_EQ(packet_ref.stream_id(), 0xAAAAAAAA);

    // Continue modifying builder
    builder.packet_count(5);

    auto packet = builder.build();
    EXPECT_EQ(packet.packet_count(), 5);
}
