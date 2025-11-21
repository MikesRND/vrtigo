#include <array>

#include <cmath>
#include <gtest/gtest.h>
#include <vrtigo.hpp>

using namespace vrtigo;
using namespace vrtigo::field;

// =============================================================================
// Interpreted Value Tests - Q52.12 Fixed-Point ↔ Hz Conversion
// Tests for CIF fields with interpreted support: bandwidth, sample_rate
// =============================================================================

class InterpretedValueTest : public ::testing::Test {
protected:
    alignas(4) std::array<uint8_t, 256> buffer{};
};

// =============================================================================
// Bandwidth Field (CIF0 Bit 29)
// =============================================================================

TEST_F(InterpretedValueTest, BandwidthInterpretedRead) {
    using TestContext = ContextPacket<NoTimeStamp, NoClassId, bandwidth>;

    TestContext packet(buffer.data());

    // Set bandwidth to Q52.12 encoding for 100 MHz
    // 100 MHz = 100'000'000 Hz
    // Q52.12: 100'000'000 * 4096 = 409'600'000'000
    packet[bandwidth].set_encoded(409'600'000'000ULL);

    // Read interpreted value
    double hz = packet[bandwidth].value();

    // Should be within 1 Hz of 100 MHz
    EXPECT_NEAR(hz, 100'000'000.0, 1.0);
}

TEST_F(InterpretedValueTest, BandwidthInterpretedWrite) {
    using TestContext = ContextPacket<NoTimeStamp, NoClassId, bandwidth>;

    TestContext packet(buffer.data());

    // Write interpreted value (50 MHz)
    packet[bandwidth].set_value(50'000'000.0);

    // Verify raw value is correct Q52.12 encoding
    // 50 MHz * 4096 = 204'800'000'000
    EXPECT_EQ(packet[bandwidth].encoded(), 204'800'000'000ULL);
}

TEST_F(InterpretedValueTest, BandwidthRoundTrip) {
    using TestContext = ContextPacket<NoTimeStamp, NoClassId, bandwidth>;

    TestContext packet(buffer.data());

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
    using TestContext = ContextPacket<NoTimeStamp, NoClassId, bandwidth>;

    TestContext packet(buffer.data());

    // Set to 200 MHz
    packet[bandwidth].set_value(200'000'000.0);

    // operator* should return interpreted value
    auto bw_proxy = packet[bandwidth];
    double hz = *bw_proxy;

    EXPECT_NEAR(hz, 200'000'000.0, 1.0);
}

TEST_F(InterpretedValueTest, BandwidthConversionPrecision) {
    using TestContext = ContextPacket<NoTimeStamp, NoClassId, bandwidth>;

    TestContext packet(buffer.data());

    // Test Q52.12 conversion accuracy

    // Test 1: Exact value (divisible by 4096)
    double exact_hz = 4096.0 * 1000.0; // 4'096'000 Hz
    packet[bandwidth].set_value(exact_hz);
    EXPECT_DOUBLE_EQ(packet[bandwidth].value(), exact_hz);

    // Test 2: Non-exact value (rounding)
    double inexact_hz = 1'234'567.89;
    packet[bandwidth].set_value(inexact_hz);
    double retrieved = packet[bandwidth].value();
    // Should be within Q52.12 resolution (~0.244 Hz)
    EXPECT_NEAR(retrieved, inexact_hz, 0.25);
}

TEST_F(InterpretedValueTest, BandwidthEdgeCases) {
    using TestContext = ContextPacket<NoTimeStamp, NoClassId, bandwidth>;

    TestContext packet(buffer.data());

    // Zero value
    packet[bandwidth].set_value(0.0);
    EXPECT_EQ(packet[bandwidth].encoded(), 0ULL);
    EXPECT_DOUBLE_EQ(packet[bandwidth].value(), 0.0);

    // Maximum Q52.12 value
    uint64_t max_q52_12 = 0xFFFFFFFFFFFFFFFFULL;
    packet[bandwidth].set_encoded(max_q52_12);
    EXPECT_EQ(packet[bandwidth].encoded(), max_q52_12);
    double expected_hz = static_cast<double>(max_q52_12) / 4096.0;
    EXPECT_NEAR(packet[bandwidth].value(), expected_hz, 1.0);
}

// =============================================================================
// Sample Rate Field (CIF0 Bit 21)
// =============================================================================

TEST_F(InterpretedValueTest, SampleRateInterpretedRead) {
    using TestContext = ContextPacket<NoTimeStamp, NoClassId, sample_rate>;

    TestContext packet(buffer.data());

    // Set sample rate to Q52.12 encoding for 50 MHz (50 MSPS)
    // 50 MHz = 50'000'000 Hz
    // Q52.12: 50'000'000 * 4096 = 204'800'000'000
    packet[sample_rate].set_encoded(204'800'000'000ULL);

    // Read interpreted value
    double hz = packet[sample_rate].value();

    // Should be within 1 Hz of 50 MHz
    EXPECT_NEAR(hz, 50'000'000.0, 1.0);
}

TEST_F(InterpretedValueTest, SampleRateInterpretedWrite) {
    using TestContext = ContextPacket<NoTimeStamp, NoClassId, sample_rate>;

    TestContext packet(buffer.data());

    // Write interpreted value (25 MSPS)
    packet[sample_rate].set_value(25'000'000.0);

    // Verify raw value is correct Q52.12 encoding
    // 25 MHz * 4096 = 102'400'000'000
    EXPECT_EQ(packet[sample_rate].encoded(), 102'400'000'000ULL);
}

TEST_F(InterpretedValueTest, SampleRateRoundTrip) {
    using TestContext = ContextPacket<NoTimeStamp, NoClassId, sample_rate>;

    TestContext packet(buffer.data());

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
    using TestContext = ContextPacket<NoTimeStamp, NoClassId, sample_rate>;

    TestContext packet(buffer.data());

    // Set to 125 MSPS
    packet[sample_rate].set_value(125'000'000.0);

    // operator* should return interpreted value
    auto sr_proxy = packet[sample_rate];
    double hz = *sr_proxy;

    EXPECT_NEAR(hz, 125'000'000.0, 1.0);
}

TEST_F(InterpretedValueTest, SampleRateConversionPrecision) {
    using TestContext = ContextPacket<NoTimeStamp, NoClassId, sample_rate>;

    TestContext packet(buffer.data());

    // Test Q52.12 conversion accuracy

    // Test 1: Exact value (divisible by 4096)
    double exact_hz = 4096.0 * 1000.0; // 4'096'000 Hz
    packet[sample_rate].set_value(exact_hz);
    EXPECT_DOUBLE_EQ(packet[sample_rate].value(), exact_hz);

    // Test 2: Non-exact value (rounding)
    double inexact_hz = 12'345'678.9;
    packet[sample_rate].set_value(inexact_hz);
    double retrieved = packet[sample_rate].value();
    // Should be within Q52.12 resolution (~0.244 Hz)
    EXPECT_NEAR(retrieved, inexact_hz, 0.25);
}

TEST_F(InterpretedValueTest, SampleRateTypicalADCRates) {
    using TestContext = ContextPacket<NoTimeStamp, NoClassId, sample_rate>;

    TestContext packet(buffer.data());

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
    using TestContext = ContextPacket<NoTimeStamp, NoClassId, sample_rate>;

    TestContext packet(buffer.data());

    // Zero value (stopped ADC)
    packet[sample_rate].set_value(0.0);
    EXPECT_EQ(packet[sample_rate].encoded(), 0ULL);
    EXPECT_DOUBLE_EQ(packet[sample_rate].value(), 0.0);

    // Maximum Q52.12 value
    uint64_t max_q52_12 = 0xFFFFFFFFFFFFFFFFULL;
    packet[sample_rate].set_encoded(max_q52_12);
    EXPECT_EQ(packet[sample_rate].encoded(), max_q52_12);
    double expected_hz = static_cast<double>(max_q52_12) / 4096.0;
    EXPECT_NEAR(packet[sample_rate].value(), expected_hz, 1.0);

    // Very low sample rate (1 Hz - theoretical minimum)
    packet[sample_rate].set_value(1.0);
    double retrieved = packet[sample_rate].value();
    EXPECT_NEAR(retrieved, 1.0, 0.25); // Within Q52.12 resolution
}

// =============================================================================
// Multi-Field Integration Tests
// =============================================================================

TEST_F(InterpretedValueTest, BandwidthAndSampleRateTogether) {
    // Typical use case: both bandwidth and sample rate in same packet
    using TestContext = ContextPacket<NoTimeStamp, NoClassId, bandwidth, sample_rate>;

    TestContext packet(buffer.data());

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
    using TestContext = ContextPacket<NoTimeStamp, NoClassId, bandwidth, sample_rate>;

    TestContext tx_packet(buffer.data());

    tx_packet.set_stream_id(0xDEADBEEF);
    tx_packet[bandwidth].set_value(75'000'000.0);   // 75 MHz
    tx_packet[sample_rate].set_value(80'000'000.0); // 80 MSPS

    // Parse with runtime view
    RuntimeContextPacket view(buffer.data(), TestContext::size_bytes);
    EXPECT_EQ(view.error(), ValidationError::none);

    // Verify values accessible from runtime parser
    auto bw = view[bandwidth];
    ASSERT_TRUE(bw.has_value());
    EXPECT_NEAR(bw.value(), 75'000'000.0, 1.0);

    auto sr = view[sample_rate];
    ASSERT_TRUE(sr.has_value());
    EXPECT_NEAR(sr.value(), 80'000'000.0, 1.0);

    EXPECT_EQ(view.stream_id().value(), 0xDEADBEEF);
}
