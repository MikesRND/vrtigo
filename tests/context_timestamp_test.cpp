#include "context_test_fixture.hpp"

using namespace vrtio::field;

TEST_F(ContextPacketTest, TimestampInitialization) {
    // Note: Context packets always have Stream ID per VITA 49.2 spec
    using TestContext = ContextPacket<TimeStampUTC, NoClassId>; // No context fields

    TestContext packet(buffer.data());

    // Timestamps should be zero-initialized
    auto ts = packet.getTimeStamp();
    EXPECT_EQ(ts.seconds(), 0);
    EXPECT_EQ(ts.fractional(), 0);
}

TEST_F(ContextPacketTest, TimestampIntegerAccess) {
    using TestContext = ContextPacket<TimeStampUTC, NoClassId>;

    TestContext packet(buffer.data());

    // Set timestamp with integer only
    TimeStampUTC ts(1699000000, 0);
    packet.setTimeStamp(ts);
    auto read_ts = packet.getTimeStamp();
    EXPECT_EQ(read_ts.seconds(), 1699000000);
    EXPECT_EQ(read_ts.fractional(), 0);
}

TEST_F(ContextPacketTest, TimestampFractionalAccess) {
    using TestContext = ContextPacket<TimeStampUTC, NoClassId>;

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
    using TestContext = ContextPacket<TimeStampUTC, NoClassId>;

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
    using TestContext = ContextPacket<TimeStampUTC, TestClassId>; // Has class ID (8 bytes)

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
    using TestContext = ContextPacket<TimeStampUTC, NoClassId, bandwidth, sample_rate>;

    TestContext packet(buffer.data());

    // Set all fields
    packet.set_stream_id(0x12345678);
    TimeStampUTC ts(1600000000, 123456789012ULL);
    packet.setTimeStamp(ts);
    get(packet, bandwidth).set_value(20'000'000.0);   // 20 MHz
    get(packet, sample_rate).set_value(10'000'000.0); // 10 MSPS

    // Verify all fields
    EXPECT_EQ(packet.stream_id(), 0x12345678);
    auto read_ts = packet.getTimeStamp();
    EXPECT_EQ(read_ts.seconds(), 1600000000);
    EXPECT_EQ(read_ts.fractional(), 123456789012ULL);
    EXPECT_DOUBLE_EQ(get(packet, bandwidth).value(), 20'000'000.0);
    EXPECT_DOUBLE_EQ(get(packet, sample_rate).value(), 10'000'000.0);
}

// Test removed: Per VITA 49.2 spec, Context packets ALWAYS have Stream ID
// The TimestampNoStreamId test is no longer valid
