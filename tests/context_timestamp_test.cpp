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
    auto ts = packet.getTimeStamp();
    EXPECT_EQ(ts.seconds(), 0);
    EXPECT_EQ(ts.fractional(), 0);
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

    // Set timestamp with integer only
    TimeStampUTC ts(1699000000, 0);
    packet.setTimeStamp(ts);
    auto read_ts = packet.getTimeStamp();
    EXPECT_EQ(read_ts.seconds(), 1699000000);
    EXPECT_EQ(read_ts.fractional(), 0);
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

    // Set timestamp with fractional part
    uint64_t frac = 500000000000ULL;
    TimeStampUTC ts(1000, frac);
    packet.setTimeStamp(ts);
    auto read_ts = packet.getTimeStamp();
    EXPECT_EQ(read_ts.seconds(), 1000);
    EXPECT_EQ(read_ts.fractional(), frac);
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
    auto ts_init = packet.getTimeStamp();
    EXPECT_EQ(ts_init.seconds(), 0);
    EXPECT_EQ(ts_init.fractional(), 0);

    // Set and verify
    TimeStampUTC ts(1234567890, 999999999999ULL);
    packet.setTimeStamp(ts);

    auto read_ts = packet.getTimeStamp();
    EXPECT_EQ(read_ts.seconds(), 1234567890);
    EXPECT_EQ(read_ts.fractional(), 999999999999ULL);
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
    TimeStampUTC ts(1600000000, 123456789012ULL);
    packet.setTimeStamp(ts);
    get(packet, field::bandwidth).set_value(20'000'000.0);      // 20 MHz
    get(packet, field::sample_rate).set_value(10'000'000.0);    // 10 MSPS

    // Verify all fields
    EXPECT_EQ(packet.stream_id(), 0x12345678);
    auto read_ts = packet.getTimeStamp();
    EXPECT_EQ(read_ts.seconds(), 1600000000);
    EXPECT_EQ(read_ts.fractional(), 123456789012ULL);
    EXPECT_DOUBLE_EQ(get(packet, field::bandwidth).value(), 20'000'000.0);
    EXPECT_DOUBLE_EQ(get(packet, field::sample_rate).value(), 10'000'000.0);
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
    TimeStampUTC ts(987654321, 111111111111ULL);
    packet.setTimeStamp(ts);

    auto read_ts = packet.getTimeStamp();
    EXPECT_EQ(read_ts.seconds(), 987654321);
    EXPECT_EQ(read_ts.fractional(), 111111111111ULL);
}

