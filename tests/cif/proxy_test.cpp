#include <vrtio.hpp>
#include <gtest/gtest.h>
#include <array>

using namespace vrtio;

// =============================================================================
// FieldProxy Interface Tests
// Tests for FieldProxy methods: raw_bytes(), set_raw_bytes(), raw_value(),
// set_raw_value(), offset(), size(), has_value(), operator bool, operator*
// =============================================================================

class FieldProxyTest : public ::testing::Test {
protected:
    alignas(4) std::array<uint8_t, 256> buffer{};
};

// =============================================================================
// Basic Access Tests
// =============================================================================

TEST_F(FieldProxyTest, BasicSetAndGet) {
    using TestContext = ContextPacket<
        true, NoTimeStamp, NoClassId,
        cif0::BANDWIDTH,
        0, 0, 0, false
    >;

    TestContext packet(buffer.data());

    // Set bandwidth raw value (Q52.12 format)
    get(packet, field::bandwidth).set_raw_value(20'000'000ULL);

    // Get and verify raw value
    auto bw = get(packet, field::bandwidth);
    ASSERT_TRUE(bw.has_value());
    EXPECT_EQ(bw.raw_value(), 20'000'000ULL);
}

TEST_F(FieldProxyTest, FieldPresenceCheck) {
    // Create packet WITH bandwidth
    using WithBandwidth = ContextPacket<
        true, NoTimeStamp, NoClassId,
        cif0::BANDWIDTH, 0, 0, 0, false
    >;

    WithBandwidth pkt1(buffer.data());

    EXPECT_TRUE(has(pkt1, field::bandwidth));
    EXPECT_TRUE(get(pkt1, field::bandwidth).has_value());

    // Create packet WITHOUT bandwidth
    using WithoutBandwidth = ContextPacket<
        true, NoTimeStamp, NoClassId,
        cif0::SAMPLE_RATE,  // Has sample_rate, NOT bandwidth
        0, 0, 0, false
    >;

    alignas(4) std::array<uint8_t, WithoutBandwidth::size_bytes> buf2{};
    WithoutBandwidth pkt2(buf2.data());

    EXPECT_FALSE(has(pkt2, field::bandwidth));
    EXPECT_FALSE(get(pkt2, field::bandwidth).has_value());
}

TEST_F(FieldProxyTest, UncheckedAccess) {
    using TestContext = ContextPacket<
        true, NoTimeStamp, NoClassId,
        cif0::BANDWIDTH, 0, 0, 0, false
    >;

    TestContext packet(buffer.data());

    // Set bandwidth
    get(packet, field::bandwidth).set_raw_value(1'000'000ULL);

    // get_unchecked() for zero-overhead access (no presence check)
    uint64_t bw_direct = get_unchecked(packet, field::bandwidth);
    EXPECT_EQ(bw_direct, 1'000'000ULL);
}

// =============================================================================
// Raw Byte Access Tests
// =============================================================================

TEST_F(FieldProxyTest, RawBytesAccess) {
    using TestContext = ContextPacket<
        true, NoTimeStamp, NoClassId,
        cif0::BANDWIDTH, 0, 0, 0, false
    >;

    TestContext packet(buffer.data());

    get(packet, field::bandwidth).set_raw_value(1'000'000ULL);

    // Get field proxy
    auto bw_proxy = get(packet, field::bandwidth);
    ASSERT_TRUE(bw_proxy.has_value());

    // Get raw bytes
    auto bw_raw = bw_proxy.raw_bytes();

    // Bandwidth is 64-bit (2 words = 8 bytes)
    EXPECT_EQ(bw_raw.size(), 8);
}

TEST_F(FieldProxyTest, RawBytesManipulation) {
    using TestContext = ContextPacket<
        true, NoTimeStamp, NoClassId,
        cif0::BANDWIDTH, 0, 0, 0, false
    >;

    TestContext packet(buffer.data());

    // Write raw bytes directly (network byte order)
    uint8_t test_bytes[8] = {
        0x00, 0x00, 0x00, 0x00,  // Upper 32 bits
        0x00, 0x0F, 0x42, 0x40   // Lower 32 bits = 1000000
    };

    auto bw_proxy = get(packet, field::bandwidth);
    bw_proxy.set_raw_bytes(std::span<const uint8_t>(test_bytes, 8));

    // Verify value was written
    EXPECT_EQ(bw_proxy.raw_value(), 1'000'000ULL);
}

TEST_F(FieldProxyTest, OffsetAndSize) {
    using TestContext = ContextPacket<
        true, NoTimeStamp, NoClassId,
        cif0::BANDWIDTH, 0, 0, 0, false
    >;

    TestContext packet(buffer.data());

    auto bw_proxy = get(packet, field::bandwidth);

    // Bandwidth is 8 bytes (2 words)
    EXPECT_EQ(bw_proxy.size(), 8);

    // Offset should be after header
    EXPECT_GT(bw_proxy.offset(), 0);
}

TEST_F(FieldProxyTest, MissingFieldHandling) {
    using TestContext = ContextPacket<
        true, NoTimeStamp, NoClassId,
        cif0::SAMPLE_RATE,  // Has sample_rate, NOT bandwidth
        0, 0, 0, false
    >;

    TestContext packet(buffer.data());

    auto missing_proxy = get(packet, field::bandwidth);

    EXPECT_FALSE(missing_proxy.has_value());
    EXPECT_FALSE(missing_proxy);  // operator bool

    auto missing_data = missing_proxy.raw_bytes();
    EXPECT_TRUE(missing_data.empty());
}

TEST_F(FieldProxyTest, DifferentFieldSizes) {
    using TestContext = ContextPacket<
        true, NoTimeStamp, NoClassId,
        cif0::BANDWIDTH | cif0::SAMPLE_RATE | cif0::GAIN,
        0, 0, 0, false
    >;

    TestContext packet(buffer.data());

    // Gain is 32-bit (4 bytes)
    auto gain_proxy = get(packet, field::gain);
    EXPECT_EQ(gain_proxy.size(), 4);

    // Sample rate is 64-bit (8 bytes)
    auto sr_proxy = get(packet, field::sample_rate);
    EXPECT_EQ(sr_proxy.size(), 8);

    // Bandwidth is 64-bit (8 bytes)
    auto bw_proxy = get(packet, field::bandwidth);
    EXPECT_EQ(bw_proxy.size(), 8);
}

TEST_F(FieldProxyTest, ConditionalPatternCompatibility) {
    using TestContext = ContextPacket<
        true, NoTimeStamp, NoClassId,
        cif0::GAIN, 0, 0, 0, false
    >;

    TestContext packet(buffer.data());

    get(packet, field::gain).set_raw_value(42U);

    // if (auto field = get(...)) pattern works
    if (auto field = get(packet, field::gain)) {
        EXPECT_EQ(field.raw_value(), 42U);
    } else {
        FAIL() << "Field should be present";
    }
}

TEST_F(FieldProxyTest, MultipleProxiesToSameField) {
    using TestContext = ContextPacket<
        true, NoTimeStamp, NoClassId,
        cif0::BANDWIDTH, 0, 0, 0, false
    >;

    TestContext packet(buffer.data());

    auto bw_proxy1 = get(packet, field::bandwidth);
    auto bw_proxy2 = get(packet, field::bandwidth);

    EXPECT_EQ(bw_proxy1.offset(), bw_proxy2.offset());
    EXPECT_EQ(bw_proxy1.size(), bw_proxy2.size());
}

// =============================================================================
// Multi-Field Packet Tests
// =============================================================================

TEST_F(FieldProxyTest, MultiFieldPacket) {
    // Packet with bandwidth + sample_rate + gain
    using TestContext = ContextPacket<
        true, NoTimeStamp, NoClassId,
        cif0::BANDWIDTH | cif0::SAMPLE_RATE | cif0::GAIN,
        0, 0, 0, false
    >;

    TestContext packet(buffer.data());

    // Set all fields
    get(packet, field::bandwidth).set_raw_value(100'000'000ULL);
    get(packet, field::sample_rate).set_raw_value(50'000'000ULL);
    get(packet, field::gain).set_raw_value(0x12345678U);

    // Verify all fields accessible and not corrupted
    EXPECT_EQ(get(packet, field::bandwidth).raw_value(), 100'000'000ULL);
    EXPECT_EQ(get(packet, field::sample_rate).raw_value(), 50'000'000ULL);
    EXPECT_EQ(get(packet, field::gain).raw_value(), 0x12345678U);

    EXPECT_TRUE(has(packet, field::bandwidth));
    EXPECT_TRUE(has(packet, field::sample_rate));
    EXPECT_TRUE(has(packet, field::gain));
}

TEST_F(FieldProxyTest, MultiCIFWordPacket) {
    // Packet with fields spanning CIF0, CIF1, CIF2
    using TestContext = ContextPacket<
        true, NoTimeStamp, NoClassId,
        cif0::BANDWIDTH,       // CIF0 field
        cif1::AUX_FREQUENCY,   // CIF1 field
        cif2::CONTROLLER_UUID, // CIF2 field
        0, false
    >;

    TestContext packet(buffer.data());

    // Set bandwidth (CIF0 field)
    get(packet, field::bandwidth).set_raw_value(150'000'000ULL);

    // Verify CIF enable bits are set
    constexpr uint32_t cif_enable_mask = (1U << cif::CIF1_ENABLE_BIT) | (1U << cif::CIF2_ENABLE_BIT);
    EXPECT_EQ(packet.cif0() & cif_enable_mask, cif_enable_mask);

    // Verify bandwidth is accessible despite multi-CIF structure
    EXPECT_EQ(get(packet, field::bandwidth).raw_value(), 150'000'000ULL);
}

TEST_F(FieldProxyTest, RuntimeParserIntegration) {
    // Build packet with compile-time type
    using TestContext = ContextPacket<
        true, NoTimeStamp, NoClassId,
        cif0::BANDWIDTH, 0, 0, 0, false
    >;

    TestContext tx_packet(buffer.data());

    tx_packet.set_stream_id(0xDEADBEEF);
    get(tx_packet, field::bandwidth).set_raw_value(75'000'000ULL);

    // Parse with runtime view
    ContextPacketView view(buffer.data(), TestContext::size_bytes);
    EXPECT_EQ(view.error(), validation_error::none);

    // Verify field accessible from runtime parser
    auto bw = get(view, field::bandwidth);
    ASSERT_TRUE(bw.has_value());
    EXPECT_EQ(bw.raw_value(), 75'000'000ULL);
    EXPECT_EQ(view.stream_id().value(), 0xDEADBEEF);
}
