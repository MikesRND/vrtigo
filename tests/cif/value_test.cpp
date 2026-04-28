#include <array>

#include <cmath>
#include <gtest/gtest.h>
#include <vrtigo.hpp>

using namespace vrtigo;
using namespace vrtigo::field;

// =============================================================================
// Interpreted Value Tests - Q44.20 Fixed-Point ↔ Hz Conversion
// Tests for CIF fields with interpreted support: bandwidth, sample_rate
// =============================================================================

namespace {
constexpr uint64_t q44_20_hz(uint64_t hz) noexcept { return hz << 20; }
} // namespace

class InterpretedValueTest : public ::testing::Test {
protected:
};

// =============================================================================
// Bandwidth Field (CIF0 Bit 29)
// =============================================================================

TEST_F(InterpretedValueTest, BandwidthInterpretedRead) {
    using TestContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, bandwidth>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);

    // Set bandwidth to Q44.20 encoding for 100 MHz
    // 100 MHz = 100'000'000 Hz
    // Q44.20: 100'000'000 * 1'048'576 = 104'857'600'000'000
    packet[bandwidth].set_encoded(q44_20_hz(100'000'000ULL));

    // Read interpreted value
    double hz = packet[bandwidth].value();

    // Should be within 1 Hz of 100 MHz
    EXPECT_NEAR(hz, 100'000'000.0, 1.0);
}

TEST_F(InterpretedValueTest, BandwidthInterpretedWrite) {
    using TestContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, bandwidth>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);

    // Write interpreted value (50 MHz)
    packet[bandwidth].set_value(50'000'000.0);

    // Verify raw value is correct Q44.20 encoding
    // 50 MHz * 1'048'576 = 52'428'800'000'000
    EXPECT_EQ(packet[bandwidth].encoded(), q44_20_hz(50'000'000ULL));
}

TEST_F(InterpretedValueTest, BandwidthRoundTrip) {
    using TestContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, bandwidth>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);

    // Test various frequencies for round-trip precision
    const double test_frequencies[] = {
        0.0,             // DC
        1'000'000.0,     // 1 MHz
        10'000'000.0,    // 10 MHz
        100'000'000.0,   // 100 MHz
        1'000'000'000.0, // 1 GHz
        6'000'000'000.0  // 6 GHz
    };

    for (double freq : test_frequencies) {
        packet[bandwidth].set_value(freq);
        double retrieved = packet[bandwidth].value();
        EXPECT_NEAR(retrieved, freq, 1.0) << "Failed for frequency: " << freq;
    }
}

TEST_F(InterpretedValueTest, BandwidthOperatorDereference) {
    using TestContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, bandwidth>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);

    // Set to 200 MHz
    packet[bandwidth].set_value(200'000'000.0);

    // operator* should return interpreted value
    auto bw_proxy = packet[bandwidth];
    double hz = *bw_proxy;

    EXPECT_NEAR(hz, 200'000'000.0, 1.0);
}

TEST_F(InterpretedValueTest, BandwidthConversionPrecision) {
    using TestContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, bandwidth>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);

    // Test Q44.20 conversion accuracy

    // Test 1: Exact integer-Hz value
    double exact_hz = 4'096'000.0;
    packet[bandwidth].set_value(exact_hz);
    EXPECT_DOUBLE_EQ(packet[bandwidth].value(), exact_hz);

    // Test 2: Non-exact value (rounding)
    double inexact_hz = 1'234'567.89;
    packet[bandwidth].set_value(inexact_hz);
    double retrieved = packet[bandwidth].value();
    // Should be within Q44.20 resolution (~0.95 microHz)
    EXPECT_NEAR(retrieved, inexact_hz, 0.000001);
}

TEST_F(InterpretedValueTest, BandwidthEdgeCases) {
    using TestContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, bandwidth>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);

    // Zero value
    packet[bandwidth].set_value(0.0);
    EXPECT_EQ(packet[bandwidth].encoded(), 0ULL);
    EXPECT_DOUBLE_EQ(packet[bandwidth].value(), 0.0);

    // Maximum positive Q44.20 value
    uint64_t max_q44_20 = 0x7FFFFFFFFFFFFFFFULL;
    packet[bandwidth].set_encoded(max_q44_20);
    EXPECT_EQ(packet[bandwidth].encoded(), max_q44_20);
    double expected_hz = static_cast<double>(max_q44_20) / 1'048'576.0;
    EXPECT_NEAR(packet[bandwidth].value(), expected_hz, 1.0);

    // Raw signed negative values decode as signed Q44.20.
    packet[bandwidth].set_encoded(0xFFFFFFFFFFFFFFFFULL);
    EXPECT_NEAR(packet[bandwidth].value(), -1.0 / 1'048'576.0, 1e-12);

    // Interpreted writes clamp negative bandwidths to zero.
    packet[bandwidth].set_value(-1.0);
    EXPECT_EQ(packet[bandwidth].encoded(), 0ULL);
    EXPECT_DOUBLE_EQ(packet[bandwidth].value(), 0.0);
}

// =============================================================================
// Sample Rate Field (CIF0 Bit 21)
// =============================================================================

TEST_F(InterpretedValueTest, SampleRateInterpretedRead) {
    using TestContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, sample_rate>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);

    // Set sample rate to Q44.20 encoding for 50 MHz (50 MSPS)
    // 50 MHz = 50'000'000 Hz
    // Q44.20: 50'000'000 * 1'048'576 = 52'428'800'000'000
    packet[sample_rate].set_encoded(q44_20_hz(50'000'000ULL));

    // Read interpreted value
    double hz = packet[sample_rate].value();

    // Should be within 1 Hz of 50 MHz
    EXPECT_NEAR(hz, 50'000'000.0, 1.0);
}

TEST_F(InterpretedValueTest, SampleRateInterpretedWrite) {
    using TestContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, sample_rate>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);

    // Write interpreted value (25 MSPS)
    packet[sample_rate].set_value(25'000'000.0);

    // Verify raw value is correct Q44.20 encoding
    // 25 MHz * 1'048'576 = 26'214'400'000'000
    EXPECT_EQ(packet[sample_rate].encoded(), q44_20_hz(25'000'000ULL));
}

TEST_F(InterpretedValueTest, SampleRateRoundTrip) {
    using TestContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, sample_rate>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);

    // Test various sample rates for round-trip precision
    const double test_rates[] = {
        0.0,             // DC
        1'000'000.0,     // 1 MSPS
        10'000'000.0,    // 10 MSPS
        50'000'000.0,    // 50 MSPS
        100'000'000.0,   // 100 MSPS
        1'000'000'000.0, // 1 GSPS
        3'000'000'000.0  // 3 GSPS
    };

    for (double rate : test_rates) {
        packet[sample_rate].set_value(rate);
        double retrieved = packet[sample_rate].value();
        EXPECT_NEAR(retrieved, rate, 1.0) << "Failed for sample rate: " << rate;
    }
}

TEST_F(InterpretedValueTest, SampleRateOperatorDereference) {
    using TestContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, sample_rate>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);

    // Set to 125 MSPS
    packet[sample_rate].set_value(125'000'000.0);

    // operator* should return interpreted value
    auto sr_proxy = packet[sample_rate];
    double hz = *sr_proxy;

    EXPECT_NEAR(hz, 125'000'000.0, 1.0);
}

TEST_F(InterpretedValueTest, SampleRateConversionPrecision) {
    using TestContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, sample_rate>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);

    // Test Q44.20 conversion accuracy

    // Test 1: Exact integer-Hz value
    double exact_hz = 4'096'000.0;
    packet[sample_rate].set_value(exact_hz);
    EXPECT_DOUBLE_EQ(packet[sample_rate].value(), exact_hz);

    // Test 2: Non-exact value (rounding)
    double inexact_hz = 12'345'678.9;
    packet[sample_rate].set_value(inexact_hz);
    double retrieved = packet[sample_rate].value();
    // Should be within Q44.20 resolution (~0.95 microHz)
    EXPECT_NEAR(retrieved, inexact_hz, 0.000001);
}

TEST_F(InterpretedValueTest, SampleRateTypicalADCRates) {
    using TestContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, sample_rate>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);

    // Test common ADC sample rates
    const double adc_rates[] = {
        48'000.0,        // Audio: 48 kHz
        192'000.0,       // High-res audio: 192 kHz
        2'400'000.0,     // 2.4 MSPS
        10'000'000.0,    // 10 MSPS
        100'000'000.0,   // 100 MSPS
        250'000'000.0,   // 250 MSPS
        500'000'000.0,   // 500 MSPS
        1'000'000'000.0, // 1 GSPS
        2'500'000'000.0  // 2.5 GSPS
    };

    for (double rate : adc_rates) {
        packet[sample_rate].set_value(rate);
        double retrieved = packet[sample_rate].value();
        EXPECT_NEAR(retrieved, rate, 1.0) << "Failed for ADC rate: " << rate;
    }
}

TEST_F(InterpretedValueTest, SampleRateEdgeCases) {
    using TestContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, sample_rate>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);

    // Zero value (stopped ADC)
    packet[sample_rate].set_value(0.0);
    EXPECT_EQ(packet[sample_rate].encoded(), 0ULL);
    EXPECT_DOUBLE_EQ(packet[sample_rate].value(), 0.0);

    // Maximum positive Q44.20 value
    uint64_t max_q44_20 = 0x7FFFFFFFFFFFFFFFULL;
    packet[sample_rate].set_encoded(max_q44_20);
    EXPECT_EQ(packet[sample_rate].encoded(), max_q44_20);
    double expected_hz = static_cast<double>(max_q44_20) / 1'048'576.0;
    EXPECT_NEAR(packet[sample_rate].value(), expected_hz, 1.0);

    // Raw signed negative values decode as signed Q44.20.
    packet[sample_rate].set_encoded(0xFFFFFFFFFFFFFFFFULL);
    EXPECT_NEAR(packet[sample_rate].value(), -1.0 / 1'048'576.0, 1e-12);

    // Interpreted writes clamp negative sample rates to zero.
    packet[sample_rate].set_value(-1.0);
    EXPECT_EQ(packet[sample_rate].encoded(), 0ULL);
    EXPECT_DOUBLE_EQ(packet[sample_rate].value(), 0.0);

    // Very low sample rate (1 Hz - theoretical minimum)
    packet[sample_rate].set_value(1.0);
    double retrieved = packet[sample_rate].value();
    EXPECT_NEAR(retrieved, 1.0, 0.000001); // Within Q44.20 resolution
}

// =============================================================================
// Multi-Field Integration Tests
// =============================================================================

TEST_F(InterpretedValueTest, BandwidthAndSampleRateTogether) {
    // Typical use case: both bandwidth and sample rate in same packet
    using TestContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, bandwidth, sample_rate>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);

    // Set bandwidth and sample rate
    // Typical: sample rate >= bandwidth (Nyquist)
    packet[bandwidth].set_value(20'000'000.0);   // 20 MHz bandwidth
    packet[sample_rate].set_value(25'000'000.0); // 25 MSPS (1.25x oversampling)

    // Verify both fields
    EXPECT_NEAR(packet[bandwidth].value(), 20'000'000.0, 1.0);
    EXPECT_NEAR(packet[sample_rate].value(), 25'000'000.0, 1.0);

    // Verify sample rate >= bandwidth (typical constraint)
    double bw = packet[bandwidth].value();
    double sr = packet[sample_rate].value();
    EXPECT_GE(sr, bw) << "Sample rate should be >= bandwidth";
}

TEST_F(InterpretedValueTest, RuntimeParserIntegration) {
    // Build packet with compile-time type
    using TestContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, bandwidth, sample_rate>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext tx_packet(buffer);

    tx_packet.set_stream_id(0xDEADBEEF);
    tx_packet[bandwidth].set_value(75'000'000.0);   // 75 MHz
    tx_packet[sample_rate].set_value(80'000'000.0); // 80 MSPS

    // Parse with runtime view
    auto result = dynamic::ContextPacketView::parse(buffer);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    const auto& view = result.value();

    // Verify values accessible from runtime parser
    auto bw = view[bandwidth];
    ASSERT_TRUE(bw.has_value());
    EXPECT_NEAR(bw.value(), 75'000'000.0, 1.0);

    auto sr = view[sample_rate];
    ASSERT_TRUE(sr.has_value());
    EXPECT_NEAR(sr.value(), 80'000'000.0, 1.0);

    EXPECT_EQ(view.stream_id().value(), 0xDEADBEEF);
}
