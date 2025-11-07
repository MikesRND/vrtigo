#include <vrtio.hpp>
#include <gtest/gtest.h>
#include <array>

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
    using PacketType = vrtio::signal_packet<
        vrtio::packet_type::signal_data_with_stream,
        vrtio::tsi_type::utc,
        vrtio::tsf_type::none,
        false,
        256
    >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};
    std::array<uint8_t, 1024> payload;
    fill_payload(payload);

    auto packet = vrtio::packet_builder<PacketType>(buffer.data())
        .stream_id(0x12345678)
        .timestamp_integer(1699000000)
        .packet_count(5)
        .payload(payload.data(), payload.size())
        .build();

    EXPECT_EQ(packet.stream_id(), 0x12345678);
    EXPECT_EQ(packet.timestamp_integer(), 1699000000);
    EXPECT_EQ(packet.packet_count(), 5);
}

// Test 2: Fluent chaining
TEST_F(BuilderTest, FluentChaining) {
    using PacketType = vrtio::signal_packet<
        vrtio::packet_type::signal_data_with_stream,
        vrtio::tsi_type::gps,
        vrtio::tsf_type::real_time,
        true,
        128
    >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};

    // Single expression with all fields
    auto packet = vrtio::packet_builder<PacketType>(buffer.data())
        .stream_id(0xABCDEF00)
        .timestamp_integer(1234567890)
        .timestamp_fractional(500000000000ULL)
        .trailer(0x00000001)
        .packet_count(10)
        .build();

    EXPECT_EQ(packet.stream_id(), 0xABCDEF00);
    EXPECT_EQ(packet.timestamp_integer(), 1234567890);
    EXPECT_EQ(packet.timestamp_fractional(), 500000000000ULL);
    EXPECT_EQ(packet.trailer(), 0x00000001);
    EXPECT_EQ(packet.packet_count(), 10);
}

// Test 3: Builder with span payload
TEST_F(BuilderTest, BuilderWithSpan) {
    using PacketType = vrtio::signal_packet<
        vrtio::packet_type::signal_data_with_stream,
        vrtio::tsi_type::none,
        vrtio::tsf_type::none,
        false,
        256
    >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};
    std::array<uint8_t, 1024> payload_data;
    fill_payload(payload_data);

    std::span<const uint8_t> payload_span(payload_data);

    auto packet = vrtio::packet_builder<PacketType>(buffer.data())
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
    using PacketType = vrtio::signal_packet<
        vrtio::packet_type::signal_data_with_stream,
        vrtio::tsi_type::utc,
        vrtio::tsf_type::none,
        false,
        128
    >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};
    std::vector<uint8_t> payload_vector(512);
    for (size_t i = 0; i < payload_vector.size(); ++i) {
        payload_vector[i] = static_cast<uint8_t>((i * 3) & 0xFF);
    }

    auto packet = vrtio::packet_builder<PacketType>(buffer.data())
        .stream_id(0xDEADBEEF)
        .timestamp_integer(1500000000)
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
    using PacketType = vrtio::signal_packet<
        vrtio::packet_type::signal_data_with_stream,
        vrtio::tsi_type::utc,
        vrtio::tsf_type::real_time,
        true,
        256
    >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};

    // Only set some fields
    auto packet = vrtio::packet_builder<PacketType>(buffer.data())
        .stream_id(0x11111111)
        .timestamp_integer(1600000000)
        // Skip fractional timestamp
        // Skip trailer
        .packet_count(7)
        .build();

    EXPECT_EQ(packet.stream_id(), 0x11111111);
    EXPECT_EQ(packet.timestamp_integer(), 1600000000);
    EXPECT_EQ(packet.packet_count(), 7);
    // Fractional and trailer will be zero (default initialized)
}

// Test 6: Builder returns reference (no copy)
TEST_F(BuilderTest, BuilderReturnsReference) {
    using PacketType = vrtio::signal_packet<
        vrtio::packet_type::signal_data_with_stream,
        vrtio::tsi_type::none,
        vrtio::tsf_type::none,
        false,
        128
    >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};

    auto packet1 = vrtio::packet_builder<PacketType>(buffer.data())
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
    using PacketType = vrtio::signal_packet<
        vrtio::packet_type::signal_data_with_stream,
        vrtio::tsi_type::utc,
        vrtio::tsf_type::none,
        false,
        128
    >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};

    vrtio::packet_builder<PacketType> builder(buffer.data());
    builder.stream_id(0xFEEDFACE)
           .timestamp_integer(1700000000);

    // Get bytes from builder
    auto bytes = builder.as_bytes();

    EXPECT_EQ(bytes.size(), PacketType::size_bytes);
    EXPECT_EQ(bytes.data(), buffer.data());
}

// Test 8: make_builder helper
TEST_F(BuilderTest, MakeBuilderHelper) {
    using PacketType = vrtio::signal_packet<
        vrtio::packet_type::signal_data_with_stream,
        vrtio::tsi_type::gps,
        vrtio::tsf_type::none,
        false,
        256
    >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};

    auto builder = vrtio::make_builder<PacketType>(buffer.data());
    auto packet = builder
        .stream_id(0x99999999)
        .timestamp_integer(1800000000)
        .build();

    EXPECT_EQ(packet.stream_id(), 0x99999999);
    EXPECT_EQ(packet.timestamp_integer(), 1800000000);
}

// Test 9: Builder without optional fields (Type 0)
TEST_F(BuilderTest, BuilderType0NoStream) {
    using PacketType = vrtio::signal_packet<
        vrtio::packet_type::signal_data_no_stream,  // Type 0
        vrtio::tsi_type::utc,
        vrtio::tsf_type::none,
        false,
        128
    >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};

    // Builder for type 0 - no stream_id() method
    auto packet = vrtio::packet_builder<PacketType>(buffer.data())
        .timestamp_integer(1234567890)
        .packet_count(3)
        .build();

    EXPECT_EQ(packet.timestamp_integer(), 1234567890);
    EXPECT_EQ(packet.packet_count(), 3);
    EXPECT_FALSE(PacketType::has_stream_id);
}

// Test 10: Multiple builders on different buffers
TEST_F(BuilderTest, MultipleBuilders) {
    using PacketType = vrtio::signal_packet<
        vrtio::packet_type::signal_data_with_stream,
        vrtio::tsi_type::utc,
        vrtio::tsf_type::none,
        false,
        128
    >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer1;
    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer2;

    auto packet1 = vrtio::packet_builder<PacketType>(buffer1.data())
        .stream_id(0x11111111)
        .timestamp_integer(1000000000)
        .build();

    auto packet2 = vrtio::packet_builder<PacketType>(buffer2.data())
        .stream_id(0x22222222)
        .timestamp_integer(2000000000)
        .build();

    EXPECT_EQ(packet1.stream_id(), 0x11111111);
    EXPECT_EQ(packet1.timestamp_integer(), 1000000000);

    EXPECT_EQ(packet2.stream_id(), 0x22222222);
    EXPECT_EQ(packet2.timestamp_integer(), 2000000000);

    // Packets should be independent
    packet1.set_stream_id(0x33333333);
    EXPECT_EQ(packet1.stream_id(), 0x33333333);
    EXPECT_EQ(packet2.stream_id(), 0x22222222);  // Unchanged
}

// Test 11: Builder packet() method
TEST_F(BuilderTest, BuilderPacketMethod) {
    using PacketType = vrtio::signal_packet<
        vrtio::packet_type::signal_data_with_stream,
        vrtio::tsi_type::none,
        vrtio::tsf_type::none,
        false,
        128
    >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};

    vrtio::packet_builder<PacketType> builder(buffer.data());
    builder.stream_id(0xAAAAAAAA);

    // Access packet before calling build()
    auto packet_ref = builder.packet();
    EXPECT_EQ(packet_ref.stream_id(), 0xAAAAAAAA);

    // Continue building
    builder.packet_count(5);

    auto packet = builder.build();
    EXPECT_EQ(packet.packet_count(), 5);
}
