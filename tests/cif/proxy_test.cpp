#include <array>

#include <gtest/gtest.h>
#include <vrtigo.hpp>

using namespace vrtigo;
using namespace vrtigo::field;

// =============================================================================
// FieldProxy Interface Tests
// Tests for FieldProxy methods: bytes(), set_bytes(), encoded(),
// set_encoded(), offset(), size(), has_value(), operator bool, operator*
// =============================================================================

class FieldProxyTest : public ::testing::Test {
protected:
};

// =============================================================================
// Basic Access Tests
// =============================================================================

TEST_F(FieldProxyTest, BasicSetAndGet) {
    using TestContext = typed::ContextPacket<NoTimestamp, NoClassId, bandwidth>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);

    // Set bandwidth raw value (Q52.12 format)
    packet[bandwidth].set_encoded(20'000'000ULL);

    // Get and verify raw value
    auto bw = packet[bandwidth];
    ASSERT_TRUE(bw.has_value());
    EXPECT_EQ(bw.encoded(), 20'000'000ULL);
}

TEST_F(FieldProxyTest, FieldPresenceCheck) {
    // Create packet WITH bandwidth
    using WithBandwidth = typed::ContextPacket<NoTimestamp, NoClassId, bandwidth>;

    alignas(4) std::array<uint8_t, WithBandwidth::size_bytes()> buffer1{};
    WithBandwidth pkt1(buffer1);

    EXPECT_TRUE(pkt1[bandwidth]);
    EXPECT_TRUE(pkt1[field::bandwidth].has_value());

    // Create packet WITHOUT bandwidth
    using WithoutBandwidth =
        typed::ContextPacket<NoTimestamp, NoClassId, sample_rate>; // Has sample_rate, NOT bandwidth

    alignas(4) std::array<uint8_t, WithoutBandwidth::size_bytes()> buf2{};
    WithoutBandwidth pkt2(buf2);

    EXPECT_FALSE(pkt2[bandwidth]);
    EXPECT_FALSE(pkt2[field::bandwidth].has_value());
}

TEST_F(FieldProxyTest, UncheckedAccess) {
    using TestContext = typed::ContextPacket<NoTimestamp, NoClassId, bandwidth>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);

    // Set bandwidth
    packet[bandwidth].set_encoded(1'000'000ULL);

    // make_field_proxy_unchecked() for zero-overhead access (no presence check)
    uint64_t bw_direct = make_field_proxy_unchecked(packet, bandwidth);
    EXPECT_EQ(bw_direct, 1'000'000ULL);
}

// =============================================================================
// Raw Byte Access Tests
// =============================================================================

TEST_F(FieldProxyTest, RawBytesAccess) {
    using TestContext = typed::ContextPacket<NoTimestamp, NoClassId, bandwidth>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);

    packet[bandwidth].set_encoded(1'000'000ULL);

    // Get field proxy
    auto bw_proxy = packet[bandwidth];
    ASSERT_TRUE(bw_proxy.has_value());

    // Get raw bytes
    auto bw_raw = bw_proxy.bytes();

    // Bandwidth is 64-bit (2 words = 8 bytes)
    EXPECT_EQ(bw_raw.size(), 8);
}

TEST_F(FieldProxyTest, RawBytesManipulation) {
    using TestContext = typed::ContextPacket<NoTimestamp, NoClassId, bandwidth>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);

    // Write raw bytes directly (network byte order)
    uint8_t test_bytes[8] = {
        0x00, 0x00, 0x00, 0x00, // Upper 32 bits
        0x00, 0x0F, 0x42, 0x40  // Lower 32 bits = 1000000
    };

    auto bw_proxy = packet[bandwidth];
    bw_proxy.set_bytes(std::span<const uint8_t>(test_bytes, 8));

    // Verify value was written
    EXPECT_EQ(bw_proxy.encoded(), 1'000'000ULL);
}

TEST_F(FieldProxyTest, OffsetAndSize) {
    using TestContext = typed::ContextPacket<NoTimestamp, NoClassId, bandwidth>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);

    auto bw_proxy = packet[bandwidth];

    // Bandwidth is 8 bytes (2 words)
    EXPECT_EQ(bw_proxy.size(), 8);

    // Offset should be after header
    EXPECT_GT(bw_proxy.offset(), 0);
}

TEST_F(FieldProxyTest, MissingFieldHandling) {
    using TestContext =
        typed::ContextPacket<NoTimestamp, NoClassId, sample_rate>; // Has sample_rate, NOT bandwidth

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);

    auto missing_proxy = packet[bandwidth];

    EXPECT_FALSE(missing_proxy.has_value());
    EXPECT_FALSE(missing_proxy); // operator bool

    auto missing_data = missing_proxy.bytes();
    EXPECT_TRUE(missing_data.empty());
}

TEST_F(FieldProxyTest, DifferentFieldSizes) {
    using TestContext = typed::ContextPacket<NoTimestamp, NoClassId, bandwidth, sample_rate, gain>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);

    // Gain is 32-bit (4 bytes)
    auto gain_proxy = packet[gain];
    EXPECT_EQ(gain_proxy.size(), 4);

    // Sample rate is 64-bit (8 bytes)
    auto sr_proxy = packet[sample_rate];
    EXPECT_EQ(sr_proxy.size(), 8);

    // Bandwidth is 64-bit (8 bytes)
    auto bw_proxy = packet[bandwidth];
    EXPECT_EQ(bw_proxy.size(), 8);
}

TEST_F(FieldProxyTest, ConditionalPatternCompatibility) {
    using TestContext = typed::ContextPacket<NoTimestamp, NoClassId, gain>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);

    packet[gain].set_encoded(42U);

    // if (auto field = packet[...]) pattern works
    if (auto field = packet[gain]) {
        EXPECT_EQ(field.encoded(), 42U);
    } else {
        FAIL() << "Field should be present";
    }
}

TEST_F(FieldProxyTest, MultipleProxiesToSameField) {
    using TestContext = typed::ContextPacket<NoTimestamp, NoClassId, bandwidth>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);

    auto bw_proxy1 = packet[bandwidth];
    auto bw_proxy2 = packet[bandwidth];

    EXPECT_EQ(bw_proxy1.offset(), bw_proxy2.offset());
    EXPECT_EQ(bw_proxy1.size(), bw_proxy2.size());
}

// =============================================================================
// Multi-Field Packet Tests
// =============================================================================

TEST_F(FieldProxyTest, MultiFieldPacket) {
    // Packet with bandwidth + sample_rate + gain
    using TestContext = typed::ContextPacket<NoTimestamp, NoClassId, bandwidth, sample_rate, gain>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);

    // Set all fields
    packet[bandwidth].set_encoded(100'000'000ULL);
    packet[sample_rate].set_encoded(50'000'000ULL);
    packet[gain].set_encoded(0x12345678U);

    // Verify all fields accessible and not corrupted
    EXPECT_EQ(packet[bandwidth].encoded(), 100'000'000ULL);
    EXPECT_EQ(packet[sample_rate].encoded(), 50'000'000ULL);
    EXPECT_EQ(packet[gain].encoded(), 0x12345678U);

    EXPECT_TRUE(packet[field::bandwidth]);
    EXPECT_TRUE(packet[field::sample_rate]);
    EXPECT_TRUE(packet[field::gain]);
}

TEST_F(FieldProxyTest, MultiCIFWordPacket) {
    // Packet with fields spanning CIF0, CIF1, CIF2
    using TestContext =
        typed::ContextPacket<NoTimestamp, NoClassId, bandwidth, aux_frequency, controller_uuid>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);

    // Set bandwidth (CIF0 field)
    packet[bandwidth].set_encoded(150'000'000ULL);

    // Verify CIF enable bits are set
    constexpr uint32_t cif_enable_mask =
        (1U << cif::CIF1_ENABLE_BIT) | (1U << cif::CIF2_ENABLE_BIT);
    EXPECT_EQ(packet.cif0() & cif_enable_mask, cif_enable_mask);

    // Verify bandwidth is accessible despite multi-CIF structure
    EXPECT_EQ(packet[bandwidth].encoded(), 150'000'000ULL);
}

TEST_F(FieldProxyTest, RuntimeParserIntegration) {
    // Build packet with compile-time type
    using TestContext = typed::ContextPacket<NoTimestamp, NoClassId, bandwidth>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext tx_packet(buffer);

    tx_packet.set_stream_id(0xDEADBEEF);
    tx_packet[bandwidth].set_encoded(75'000'000ULL);

    // Parse with runtime view
    auto result = dynamic::ContextPacket::parse(buffer);
    ASSERT_TRUE(result.ok()) << result.error().message();
    const auto& view = result.value();

    // Verify field accessible from runtime parser
    auto bw = view[bandwidth];
    ASSERT_TRUE(bw.has_value());
    EXPECT_EQ(bw.encoded(), 75'000'000ULL);
    EXPECT_EQ(view.stream_id().value(), 0xDEADBEEF);
}
