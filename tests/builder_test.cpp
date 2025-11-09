#include <vrtio.hpp>
#include <gtest/gtest.h>
#include <array>
#include <vrtio/core/trailer_view.hpp>

// Test fixture for builder tests
class BuilderTest : public ::testing::Test {
protected:
    template<size_t N>
    void fill_payload(std::array<uint8_t, N>& data) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<uint8_t>(i & 0xFF);
        }
    }
};

// Test 1: Basic builder usage
TEST_F(BuilderTest, BasicBuilder) {
    using PacketType = vrtio::SignalDataPacket<
        vrtio::TimeStampUTC,
        vrtio::Trailer::None,
        256
    >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};
    std::array<uint8_t, 1024> payload;
    fill_payload(payload);

    auto ts = vrtio::TimeStampUTC::fromComponents(1699000000, 0);
    auto packet = vrtio::PacketBuilder<PacketType>(buffer.data())
        .stream_id(0x12345678)
        .timestamp(ts)
        .packet_count(5)
        .payload(payload.data(), payload.size())
        .build();

    EXPECT_EQ(packet.stream_id(), 0x12345678);
    EXPECT_EQ(packet.getTimeStamp().seconds(), 1699000000);
    EXPECT_EQ(packet.packet_count(), 5);
}

// Test 2: Fluent chaining
TEST_F(BuilderTest, FluentChaining) {
    using PacketType = vrtio::SignalDataPacket<
        vrtio::TimeStampUTC,
        vrtio::Trailer::Included,
        128
    >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};

    // Single expression with all fields
    auto trailer_cfg = vrtio::TrailerBuilder{}
        .clear()
        .context_packets(1);

    auto ts = vrtio::TimeStampUTC::fromComponents(1234567890, 500000000000ULL);
    auto packet = vrtio::PacketBuilder<PacketType>(buffer.data())
        .stream_id(0xABCDEF00)
        .timestamp(ts)
        .trailer(trailer_cfg)
        .packet_count(10)
        .build();

    auto read_ts = packet.getTimeStamp();
    EXPECT_EQ(packet.stream_id(), 0xABCDEF00);
    EXPECT_EQ(read_ts.seconds(), 1234567890);
    EXPECT_EQ(read_ts.fractional(), 500000000000ULL);
    EXPECT_EQ(packet.trailer().raw(), 0x00000001);
    EXPECT_EQ(packet.packet_count(), 10);
}

TEST_F(BuilderTest, TrailerBuilderValueObject) {
    using PacketType = vrtio::SignalDataPacket<
        vrtio::TimeStampUTC,
        vrtio::Trailer::Included,
        64
    >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};

    auto trailer_cfg = vrtio::TrailerBuilder{}
        .valid_data(true)
        .calibrated_time(true)
        .context_packets(7)
        .reference_lock(true);

    auto packet = vrtio::PacketBuilder<PacketType>(buffer.data())
        .stream_id(0x0BADBEEF)
        .trailer(trailer_cfg)
        .packet_count(2)
        .build();

    EXPECT_TRUE(packet.trailer().valid_data());
    EXPECT_TRUE(packet.trailer().calibrated_time());
    EXPECT_EQ(packet.trailer().context_packets(), 7);
    EXPECT_TRUE(packet.trailer().reference_lock());
}

TEST_F(BuilderTest, TrailerBuilderChaining) {
    using PacketType = vrtio::SignalDataPacket<
        vrtio::TimeStampUTC,
        vrtio::Trailer::Included,
        64
    >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};

    auto trailer_cfg = vrtio::TrailerBuilder{}
        .clear()
        .valid_data(true)
        .calibrated_time(true)
        .context_packets(3)
        .over_range(true);

    auto packet = vrtio::PacketBuilder<PacketType>(buffer.data())
        .stream_id(0x10203040)
        .trailer(trailer_cfg)
        .packet_count(4)
        .build();

    EXPECT_TRUE(packet.trailer().valid_data());
    EXPECT_TRUE(packet.trailer().calibrated_time());
    EXPECT_EQ(packet.trailer().context_packets(), 3);
    EXPECT_TRUE(packet.trailer().over_range());
    EXPECT_EQ(packet.packet_count(), 4);
}

// Explicitly verify the legacy raw-literal setter remains available
TEST_F(BuilderTest, RawTrailerLiteralStillSupported) {
    using PacketType = vrtio::SignalDataPacket<
        vrtio::TimeStampUTC,
        vrtio::Trailer::Included,
        32
    >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};

    auto packet = vrtio::PacketBuilder<PacketType>(buffer.data())
        .stream_id(0x1234ABCD)
        .trailer(0xA5A5A5A5)
        .build();

    EXPECT_EQ(packet.trailer().raw(), 0xA5A5A5A5);
}

// Test 3: Builder with span payload
TEST_F(BuilderTest, BuilderWithSpan) {
    using PacketType = vrtio::SignalDataPacket<
        vrtio::NoTimeStamp,
        vrtio::Trailer::None,
        256
    >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};
    std::array<uint8_t, 1024> payload_data;
    fill_payload(payload_data);

    std::span<const uint8_t> payload_span(payload_data);

    auto packet = vrtio::PacketBuilder<PacketType>(buffer.data())
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
    using PacketType = vrtio::SignalDataPacket<
        vrtio::TimeStampUTC,
        vrtio::Trailer::None,
        128
    >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};
    std::vector<uint8_t> payload_vector(512);
    for (size_t i = 0; i < payload_vector.size(); ++i) {
        payload_vector[i] = static_cast<uint8_t>((i * 3) & 0xFF);
    }

    auto ts = vrtio::TimeStampUTC::fromComponents(1500000000, 0);
    auto packet = vrtio::PacketBuilder<PacketType>(buffer.data())
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
    using PacketType = vrtio::SignalDataPacket<
        vrtio::TimeStampUTC,
        vrtio::Trailer::Included,
        256
    >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};

    // Only set some fields
    auto ts = vrtio::TimeStampUTC::fromComponents(1600000000, 0);
    auto packet = vrtio::PacketBuilder<PacketType>(buffer.data())
        .stream_id(0x11111111)
        .timestamp(ts)
        // Skip trailer
        .packet_count(7)
        .build();

    EXPECT_EQ(packet.stream_id(), 0x11111111);
    EXPECT_EQ(packet.getTimeStamp().seconds(), 1600000000);
    EXPECT_EQ(packet.packet_count(), 7);
    // Trailer will be zero (default initialized)
}

// Test 6: Builder returns reference (no copy)
TEST_F(BuilderTest, BuilderReturnsReference) {
    using PacketType = vrtio::SignalDataPacket<
        vrtio::NoTimeStamp,
        vrtio::Trailer::None,
        128
    >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};

    auto packet1 = vrtio::PacketBuilder<PacketType>(buffer.data())
        .stream_id(0x12345678)
        .build();

    // Modify through reference
    packet1.set_stream_id(0x87654321);

    // Create another view of same buffer
    PacketType packet2(buffer.data());

    // Should see the modification
    EXPECT_EQ(packet2.stream_id(), 0x87654321);
}

// Test 7: as_bytes() method
TEST_F(BuilderTest, AsBytesMethod) {
    using PacketType = vrtio::SignalDataPacket<
        vrtio::TimeStampUTC,
        vrtio::Trailer::None,
        128
    >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};

    auto ts = vrtio::TimeStampUTC::fromComponents(1700000000, 0);
    vrtio::PacketBuilder<PacketType> builder(buffer.data());
    builder.stream_id(0xFEEDFACE)
           .timestamp(ts);

    // Get bytes from builder
    auto bytes = builder.as_bytes();

    EXPECT_EQ(bytes.size(), PacketType::size_bytes);
    EXPECT_EQ(bytes.data(), buffer.data());
}

// Test 8: make_builder helper
TEST_F(BuilderTest, MakeBuilderHelper) {
    using PacketType = vrtio::SignalDataPacket<
        vrtio::TimeStampUTC,
        vrtio::Trailer::None,
        256
    >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};

    auto ts = vrtio::TimeStampUTC::fromComponents(1800000000, 0);
    auto builder = vrtio::make_builder<PacketType>(buffer.data());
    auto packet = builder
        .stream_id(0x99999999)
        .timestamp(ts)
        .build();

    EXPECT_EQ(packet.stream_id(), 0x99999999);
    EXPECT_EQ(packet.getTimeStamp().seconds(), 1800000000);
}

// Test 9: Builder without optional fields (Type 0)
TEST_F(BuilderTest, BuilderType0NoStream) {
    using PacketType = vrtio::SignalDataPacketNoId<
        vrtio::TimeStampUTC,
        vrtio::Trailer::None,
        128
    >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};

    // Builder for type 0 - no stream_id() method
    auto ts = vrtio::TimeStampUTC::fromComponents(1234567890, 0);
    auto packet = vrtio::PacketBuilder<PacketType>(buffer.data())
        .timestamp(ts)
        .packet_count(3)
        .build();

    EXPECT_EQ(packet.getTimeStamp().seconds(), 1234567890);
    EXPECT_EQ(packet.packet_count(), 3);
    EXPECT_FALSE(PacketType::has_stream_id);
}

// Test 10: Multiple builders on different buffers
TEST_F(BuilderTest, MultipleBuilders) {
    using PacketType = vrtio::SignalDataPacket<
        vrtio::TimeStampUTC,
        vrtio::Trailer::None,
        128
    >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer1;
    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer2;

    auto ts1 = vrtio::TimeStampUTC::fromComponents(1000000000, 0);
    auto packet1 = vrtio::PacketBuilder<PacketType>(buffer1.data())
        .stream_id(0x11111111)
        .timestamp(ts1)
        .build();

    auto ts2 = vrtio::TimeStampUTC::fromComponents(2000000000, 0);
    auto packet2 = vrtio::PacketBuilder<PacketType>(buffer2.data())
        .stream_id(0x22222222)
        .timestamp(ts2)
        .build();

    EXPECT_EQ(packet1.stream_id(), 0x11111111);
    EXPECT_EQ(packet1.getTimeStamp().seconds(), 1000000000);

    EXPECT_EQ(packet2.stream_id(), 0x22222222);
    EXPECT_EQ(packet2.getTimeStamp().seconds(), 2000000000);

    // Packets should be independent
    packet1.set_stream_id(0x33333333);
    EXPECT_EQ(packet1.stream_id(), 0x33333333);
    EXPECT_EQ(packet2.stream_id(), 0x22222222);  // Unchanged
}

// Test 11: Builder build() method returns reference
TEST_F(BuilderTest, BuilderBuildMethod) {
    using PacketType = vrtio::SignalDataPacket<
        vrtio::NoTimeStamp,
        vrtio::Trailer::None,
        128
    >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};

    vrtio::PacketBuilder<PacketType> builder(buffer.data());
    builder.stream_id(0xAAAAAAAA);

    // Build and get packet reference
    auto packet_ref = builder.build();
    EXPECT_EQ(packet_ref.stream_id(), 0xAAAAAAAA);

    // Continue modifying builder
    builder.packet_count(5);

    auto packet = builder.build();
    EXPECT_EQ(packet.packet_count(), 5);
}
