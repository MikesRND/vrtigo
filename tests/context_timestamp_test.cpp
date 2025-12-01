#include "context_test_fixture.hpp"

using namespace vrtigo::field;

TEST_F(ContextPacketTest, TimestampInitialization) {
    // Note: Context packets always have Stream ID per VITA 49.2 spec
    using TestContext =
        typed::ContextPacketBuilder<UtcRealTimestamp, NoClassId>; // No context fields

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> pkt_buffer{};
    TestContext packet(pkt_buffer);

    // Timestamps should be zero-initialized
    auto ts = packet.timestamp();
    EXPECT_EQ(ts.tsi(), 0);
    EXPECT_EQ(ts.tsf(), 0);
}

TEST_F(ContextPacketTest, TimestampIntegerAccess) {
    using TestContext = typed::ContextPacketBuilder<UtcRealTimestamp, NoClassId>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> pkt_buffer{};
    TestContext packet(pkt_buffer);

    // Set timestamp with integer only
    UtcRealTimestamp ts(1699000000, 0);
    packet.set_timestamp(ts);
    auto read_ts = packet.timestamp();
    EXPECT_EQ(read_ts.tsi(), 1699000000);
    EXPECT_EQ(read_ts.tsf(), 0);
}

TEST_F(ContextPacketTest, TimestampFractionalAccess) {
    using TestContext = typed::ContextPacketBuilder<UtcRealTimestamp, NoClassId>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> pkt_buffer{};
    TestContext packet(pkt_buffer);

    // Set timestamp with fractional part
    uint64_t frac = 500000000000ULL;
    UtcRealTimestamp ts(1000, frac);
    packet.set_timestamp(ts);
    auto read_ts = packet.timestamp();
    EXPECT_EQ(read_ts.tsi(), 1000);
    EXPECT_EQ(read_ts.tsf(), frac);
}

TEST_F(ContextPacketTest, UnifiedTimestampAccess) {
    using TestContext = typed::ContextPacketBuilder<UtcRealTimestamp, NoClassId>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> pkt_buffer{};
    TestContext packet(pkt_buffer);

    // Create a timestamp
    UtcRealTimestamp ts(1699000000, 250000000000ULL);

    // Set using unified API
    packet.set_timestamp(ts);

    // Get using unified API
    auto retrieved = packet.timestamp();
    EXPECT_EQ(retrieved.tsi(), 1699000000);
    EXPECT_EQ(retrieved.tsf(), 250000000000ULL);
}

TEST_F(ContextPacketTest, TimestampWithClassId) {
    using TestContext =
        typed::ContextPacketBuilder<UtcRealTimestamp, ClassId>; // Has class ID (8 bytes)

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> pkt_buffer{};
    TestContext packet(pkt_buffer);

    // Timestamps should be zero-initialized even with class ID present
    auto ts_init = packet.timestamp();
    EXPECT_EQ(ts_init.tsi(), 0);
    EXPECT_EQ(ts_init.tsf(), 0);

    // Set class ID and timestamp
    ClassIdValue cid(0x123456, 0x5678, 0xABCD);
    packet.set_class_id(cid);

    UtcRealTimestamp ts(1234567890, 999999999999ULL);
    packet.set_timestamp(ts);

    // Verify timestamp
    auto read_ts = packet.timestamp();
    EXPECT_EQ(read_ts.tsi(), 1234567890);
    EXPECT_EQ(read_ts.tsf(), 999999999999ULL);

    // Verify class ID
    auto read_cid = packet.class_id();
    EXPECT_EQ(read_cid.oui(), 0x123456U);
    EXPECT_EQ(read_cid.icc(), 0x5678U);
    EXPECT_EQ(read_cid.pcc(), 0xABCDU);
}

TEST_F(ContextPacketTest, TimestampWithContextFields) {
    using TestContext =
        typed::ContextPacketBuilder<UtcRealTimestamp, NoClassId, bandwidth, sample_rate>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> pkt_buffer{};
    TestContext packet(pkt_buffer);

    // Set all fields
    packet.set_stream_id(0x12345678);
    UtcRealTimestamp ts(1600000000, 123456789012ULL);
    packet.set_timestamp(ts);
    packet[bandwidth].set_value(20'000'000.0);   // 20 MHz
    packet[sample_rate].set_value(10'000'000.0); // 10 MSPS

    // Verify all fields
    EXPECT_EQ(packet.stream_id(), 0x12345678);
    auto read_ts = packet.timestamp();
    EXPECT_EQ(read_ts.tsi(), 1600000000);
    EXPECT_EQ(read_ts.tsf(), 123456789012ULL);
    EXPECT_DOUBLE_EQ(packet[bandwidth].value(), 20'000'000.0);
    EXPECT_DOUBLE_EQ(packet[sample_rate].value(), 10'000'000.0);
}

// Test removed: Per VITA 49.2 spec, Context packets ALWAYS have Stream ID
// The TimestampNoStreamId test is no longer valid
