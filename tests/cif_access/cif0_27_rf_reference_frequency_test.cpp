// Self-Documenting CIF Access Test
// Demonstrates accessing and manipulating the RF Reference Frequency field
//
// This file auto-generates: docs/cif_access/cif0_27_rf_reference_frequency.md

#include "../context_test_fixture.hpp"

using namespace vrtigo;
using namespace vrtigo::field;

// [TITLE]
// RF Reference Frequency Field (CIF0 bit 27)
// [/TITLE]

TEST_F(ContextPacketTest, CIF0_27_BasicAccess) {
    // [EXAMPLE]
    // Setting and Reading RF Reference Frequency
    // [/EXAMPLE]

    // [DESCRIPTION]
    // The RF Reference Frequency field specifies the original RF frequency that was
    // translated to the IF Reference Frequency. It uses 64-bit two's complement Q44.20
    // fixed-point format, providing ±8.79 THz range with 0.95 µHz resolution.
    // [/DESCRIPTION]

    // [SNIPPET]
    using RFRefContext =
        typed::ContextPacketBuilder<NoTimestamp, NoClassId, rf_reference_frequency>;

    alignas(4) std::array<uint8_t, RFRefContext::size_bytes()> buffer{};
    RFRefContext packet(buffer);

    // Set RF reference frequency to 2.4 GHz (WiFi band)
    packet[rf_reference_frequency].set_value(2.4e9);

    // Read back the value in Hz
    double freq_hz = packet[rf_reference_frequency].value();
    // [/SNIPPET]

    // Assertions
    EXPECT_NEAR(freq_hz, 2.4e9, 1.0); // Within 1 Hz tolerance
}

// Additional tests (not included in documentation)

TEST_F(ContextPacketTest, CIF0_27_CommonRFFrequencies) {
    using RFRefContext =
        typed::ContextPacketBuilder<NoTimestamp, NoClassId, rf_reference_frequency>;
    alignas(4) std::array<uint8_t, RFRefContext::size_bytes()> buffer{};
    RFRefContext packet(buffer);

    // Test common RF frequencies
    double test_freqs[] = {
        100.0e6, // 100 MHz (FM radio)
        915.0e6, // 915 MHz (ISM band)
        2.4e9,   // 2.4 GHz (WiFi, Bluetooth)
        5.8e9,   // 5.8 GHz (WiFi)
        10.0e9,  // 10 GHz (X-band)
        28.0e9   // 28 GHz (5G mmWave)
    };

    for (double freq : test_freqs) {
        packet[rf_reference_frequency].set_value(freq);
        double readback = packet[rf_reference_frequency].value();
        EXPECT_NEAR(readback, freq, freq * 1e-6); // Within 1 ppm
    }
}

TEST_F(ContextPacketTest, CIF0_27_SpecExamples) {
    using RFRefContext =
        typed::ContextPacketBuilder<NoTimestamp, NoClassId, rf_reference_frequency>;
    alignas(4) std::array<uint8_t, RFRefContext::size_bytes()> buffer{};
    RFRefContext packet(buffer);

    // Per VITA 49.2 spec observations:
    // 0x0000000000100000 = +1 Hz
    packet[rf_reference_frequency].set_encoded(0x0000000000100000ULL);
    EXPECT_NEAR(packet[rf_reference_frequency].value(), 1.0, 1e-6);

    // 0xFFFFFFFFFFF00000 = -1 Hz
    packet[rf_reference_frequency].set_encoded(0xFFFFFFFFFFF00000ULL);
    EXPECT_NEAR(packet[rf_reference_frequency].value(), -1.0, 1e-6);

    // 0x0000000000000001 = +0.95 µHz
    packet[rf_reference_frequency].set_encoded(0x0000000000000001ULL);
    EXPECT_NEAR(packet[rf_reference_frequency].value(), 9.5367431640625e-7, 1e-12);
}

TEST_F(ContextPacketTest, CIF0_27_RuntimeAccess) {
    // Create a compile-time packet
    using RFRefContext =
        typed::ContextPacketBuilder<NoTimestamp, NoClassId, rf_reference_frequency>;
    alignas(4) std::array<uint8_t, RFRefContext::size_bytes()> buffer{};
    RFRefContext packet(buffer);

    packet[rf_reference_frequency].set_value(2.4e9);

    // Parse with runtime packet
    auto result = dynamic::ContextPacketView::parse(buffer);
    ASSERT_TRUE(result.ok()) << result.error().message();
    const auto& runtime_packet = result.value();

    if (runtime_packet[rf_reference_frequency]) {
        EXPECT_NEAR(runtime_packet[rf_reference_frequency].value(), 2.4e9, 1.0);
    }
}
