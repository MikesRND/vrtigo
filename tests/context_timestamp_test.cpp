#include "context_test_fixture.hpp"

TEST_F(ContextPacketTest, TimestampInitialization) {
    using TestContext = ContextPacket<
        true,                   // Has stream ID
        TimeStampUTC,          // UTC timestamp
        NoClassId,
        0, 0, 0,               // No context fields
        false
    >;

    TestContext packet(buffer.data());

    // Timestamps should be zero-initialized
    EXPECT_EQ(packet.timestamp_integer(), 0);
    EXPECT_EQ(packet.timestamp_fractional(), 0);
}

TEST_F(ContextPacketTest, TimestampIntegerAccess) {
    using TestContext = ContextPacket<
        true,
        TimeStampUTC,
        NoClassId,
        0, 0, 0,
        false
    >;

    TestContext packet(buffer.data());

    // Set integer timestamp
    packet.set_timestamp_integer(1699000000);
    EXPECT_EQ(packet.timestamp_integer(), 1699000000);
}

TEST_F(ContextPacketTest, TimestampFractionalAccess) {
    using TestContext = ContextPacket<
        true,
        TimeStampUTC,
        NoClassId,
        0, 0, 0,
        false
    >;

    TestContext packet(buffer.data());

    // Set fractional timestamp
    uint64_t frac = 500000000000ULL;
    packet.set_timestamp_fractional(frac);
    EXPECT_EQ(packet.timestamp_fractional(), frac);
}

TEST_F(ContextPacketTest, UnifiedTimestampAccess) {
    using TestContext = ContextPacket<
        true,
        TimeStampUTC,
        NoClassId,
        0, 0, 0,
        false
    >;

    TestContext packet(buffer.data());

    // Create a timestamp
    TimeStampUTC ts(1699000000, 250000000000ULL);

    // Set using unified API
    packet.setTimeStamp(ts);

    // Verify individual components
    EXPECT_EQ(packet.timestamp_integer(), 1699000000);
    EXPECT_EQ(packet.timestamp_fractional(), 250000000000ULL);

    // Get using unified API
    auto retrieved = packet.getTimeStamp();
    EXPECT_EQ(retrieved.seconds(), 1699000000);
    EXPECT_EQ(retrieved.fractional(), 250000000000ULL);
}

TEST_F(ContextPacketTest, TimestampWithClassId) {
    using TestClassId = ClassId<0x123456, 0xABCDEF00>;
    using TestContext = ContextPacket<
        true,
        TimeStampUTC,
        TestClassId,           // Has class ID (8 bytes)
        0, 0, 0,
        false
    >;

    TestContext packet(buffer.data());

    // Timestamps should be zero-initialized even with class ID present
    EXPECT_EQ(packet.timestamp_integer(), 0);
    EXPECT_EQ(packet.timestamp_fractional(), 0);

    // Set and verify
    packet.set_timestamp_integer(1234567890);
    packet.set_timestamp_fractional(999999999999ULL);

    EXPECT_EQ(packet.timestamp_integer(), 1234567890);
    EXPECT_EQ(packet.timestamp_fractional(), 999999999999ULL);
}

TEST_F(ContextPacketTest, TimestampWithContextFields) {
    constexpr uint32_t cif0_mask = cif0::BANDWIDTH | cif0::SAMPLE_RATE;
    using TestContext = ContextPacket<
        true,
        TimeStampUTC,
        NoClassId,
        cif0_mask, 0, 0,
        false
    >;

    TestContext packet(buffer.data());

    // Set all fields
    packet.set_stream_id(0x12345678);
    packet.set_timestamp_integer(1600000000);
    packet.set_timestamp_fractional(123456789012ULL);
    set(packet, field::bandwidth, 20'000'000ULL);
    set(packet, field::sample_rate, 10'000'000ULL);

    // Verify all fields
    EXPECT_EQ(packet.stream_id(), 0x12345678);
    EXPECT_EQ(packet.timestamp_integer(), 1600000000);
    EXPECT_EQ(packet.timestamp_fractional(), 123456789012ULL);
    EXPECT_EQ(get(packet, field::bandwidth).value(), 20'000'000);
    EXPECT_EQ(get(packet, field::sample_rate).value(), 10'000'000);
}

TEST_F(ContextPacketTest, TimestampNoStreamId) {
    using TestContext = ContextPacket<
        false,                  // No stream ID
        TimeStampUTC,
        NoClassId,
        0, 0, 0,
        false
    >;

    TestContext packet(buffer.data());

    // Should still work correctly
    packet.set_timestamp_integer(987654321);
    packet.set_timestamp_fractional(111111111111ULL);

    EXPECT_EQ(packet.timestamp_integer(), 987654321);
    EXPECT_EQ(packet.timestamp_fractional(), 111111111111ULL);
}

