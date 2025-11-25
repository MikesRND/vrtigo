// Self-Documenting CIF Access Test
// Demonstrates accessing and manipulating the IF Reference Frequency field
//
// This file auto-generates: docs/cif_access/cif0_28_if_reference_frequency.md

#include "../context_test_fixture.hpp"

using namespace vrtigo;
using namespace vrtigo::field;

// [TITLE]
// IF Reference Frequency Field (CIF0 bit 28)
// [/TITLE]

TEST_F(ContextPacketTest, CIF0_28_BasicAccess) {
    // [EXAMPLE]
    // Setting and Reading IF Reference Frequency
    // [/EXAMPLE]

    // [DESCRIPTION]
    // The IF Reference Frequency field indicates a frequency within the usable spectrum
    // of the described signal. It uses 64-bit two's complement Q44.20 fixed-point format
    // (radix point right of bit 20), providing ±8.79 THz range with 0.95 µHz resolution.
    // [/DESCRIPTION]

    // [SNIPPET]
    using IFRefContext = ContextPacket<NoTimestamp, NoClassId, if_reference_frequency>;

    alignas(4) std::array<uint8_t, IFRefContext::size_bytes> buffer{};
    IFRefContext packet(buffer.data());

    // Set IF reference frequency to 10.7 MHz (typical AM/FM IF)
    packet[if_reference_frequency].set_value(10.7e6);

    // Read back the value in Hz
    double freq_hz = packet[if_reference_frequency].value();
    // [/SNIPPET]

    // Assertions
    EXPECT_NEAR(freq_hz, 10.7e6, 1.0); // Within 1 Hz tolerance
}

// Additional tests (not included in documentation)

TEST_F(ContextPacketTest, CIF0_28_PositiveFrequencies) {
    using IFRefContext = ContextPacket<NoTimestamp, NoClassId, if_reference_frequency>;
    alignas(4) std::array<uint8_t, IFRefContext::size_bytes> buffer{};
    IFRefContext packet(buffer.data());

    // Test various positive frequencies
    double test_freqs[] = {
        1.0,     // 1 Hz
        1000.0,  // 1 kHz
        10.7e6,  // 10.7 MHz (AM/FM IF)
        70.0e6,  // 70 MHz
        455.0e3, // 455 kHz (AM IF)
        1.0e9    // 1 GHz
    };

    for (double freq : test_freqs) {
        packet[if_reference_frequency].set_value(freq);
        double readback = packet[if_reference_frequency].value();
        EXPECT_NEAR(readback, freq, freq * 1e-6); // Within 1 ppm
    }
}

TEST_F(ContextPacketTest, CIF0_28_NegativeFrequencies) {
    using IFRefContext = ContextPacket<NoTimestamp, NoClassId, if_reference_frequency>;
    alignas(4) std::array<uint8_t, IFRefContext::size_bytes> buffer{};
    IFRefContext packet(buffer.data());

    // Test negative frequencies (for complex samples)
    double test_freqs[] = {-1.0, -10.7e6, -1.0e9};

    for (double freq : test_freqs) {
        packet[if_reference_frequency].set_value(freq);
        double readback = packet[if_reference_frequency].value();
        EXPECT_NEAR(readback, freq, std::abs(freq) * 1e-6); // Within 1 ppm
    }
}

TEST_F(ContextPacketTest, CIF0_28_SpecExamples) {
    using IFRefContext = ContextPacket<NoTimestamp, NoClassId, if_reference_frequency>;
    alignas(4) std::array<uint8_t, IFRefContext::size_bytes> buffer{};
    IFRefContext packet(buffer.data());

    // Per VITA 49.2 spec observations:
    // 0x0000000000100000 = 1 Hz
    packet[if_reference_frequency].set_encoded(0x0000000000100000ULL);
    EXPECT_NEAR(packet[if_reference_frequency].value(), 1.0, 1e-6);

    // 0xFFFFFFFFFFF00000 = -1 Hz
    packet[if_reference_frequency].set_encoded(0xFFFFFFFFFFF00000ULL);
    EXPECT_NEAR(packet[if_reference_frequency].value(), -1.0, 1e-6);

    // 0x0000000000000001 = 0.95 µHz (minimum positive resolution)
    packet[if_reference_frequency].set_encoded(0x0000000000000001ULL);
    EXPECT_NEAR(packet[if_reference_frequency].value(), 9.5367431640625e-7, 1e-12);
}

TEST_F(ContextPacketTest, CIF0_28_RuntimeAccess) {
    // Create a compile-time packet
    using IFRefContext = ContextPacket<NoTimestamp, NoClassId, if_reference_frequency>;
    alignas(4) std::array<uint8_t, IFRefContext::size_bytes> buffer{};
    IFRefContext packet(buffer.data());

    packet[if_reference_frequency].set_value(10.7e6);

    // Parse with runtime packet
    RuntimeContextPacket runtime_packet(buffer.data(), buffer.size());

    EXPECT_TRUE(runtime_packet.is_valid());
    if (runtime_packet[if_reference_frequency]) {
        EXPECT_NEAR(runtime_packet[if_reference_frequency].value(), 10.7e6, 1.0);
    }
}
