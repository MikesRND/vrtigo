// Self-Documenting CIF Access Test
// Demonstrates accessing and manipulating the RF Frequency Offset field
//
// This file auto-generates: docs/cif_access/cif0_26_rf_frequency_offset.md

#include "../context_test_fixture.hpp"

using namespace vrtigo;
using namespace vrtigo::field;

// [TITLE]
// RF Frequency Offset Field (CIF0 bit 26)
// [/TITLE]

TEST_F(ContextPacketTest, CIF0_26_BasicAccess) {
    // [EXAMPLE]
    // Setting and Reading RF Frequency Offset
    // [/EXAMPLE]

    // [DESCRIPTION]
    // The RF Frequency Offset field works with RF Reference Frequency to describe
    // channelized signals. When present, the original frequency is RF Reference Frequency
    // + RF Frequency Offset. Uses 64-bit two's complement Q44.20 fixed-point format.
    // [/DESCRIPTION]

    // [SNIPPET]
    using RFOffsetContext =
        typed::ContextPacketBuilder<NoTimestamp, NoClassId, rf_frequency_offset>;

    alignas(4) std::array<uint8_t, RFOffsetContext::size_bytes()> buffer{};
    RFOffsetContext packet(buffer);

    // Set RF frequency offset to 1 MHz (channelizer offset)
    packet[rf_frequency_offset].set_value(1.0e6);

    // Read back the value in Hz
    double offset_hz = packet[rf_frequency_offset].value();
    // [/SNIPPET]

    // Assertions
    EXPECT_NEAR(offset_hz, 1.0e6, 1.0); // Within 1 Hz tolerance
}

// Additional tests (not included in documentation)

TEST_F(ContextPacketTest, CIF0_26_ChannelizerOffsets) {
    using RFOffsetContext =
        typed::ContextPacketBuilder<NoTimestamp, NoClassId, rf_frequency_offset>;
    alignas(4) std::array<uint8_t, RFOffsetContext::size_bytes()> buffer{};
    RFOffsetContext packet(buffer);

    // Test typical channelizer offsets
    double test_offsets[] = {
        0.0,     // No offset
        100.0e3, // 100 kHz channel spacing
        1.0e6,   // 1 MHz channel spacing
        10.0e6,  // 10 MHz channel spacing
        -1.0e6,  // Negative offset
        -10.0e6  // Negative offset
    };

    for (double offset : test_offsets) {
        packet[rf_frequency_offset].set_value(offset);
        double readback = packet[rf_frequency_offset].value();
        EXPECT_NEAR(readback, offset, std::abs(offset) * 1e-6 + 1e-3); // Within 1 ppm + 1 mHz
    }
}

TEST_F(ContextPacketTest, CIF0_26_SpecExamples) {
    using RFOffsetContext =
        typed::ContextPacketBuilder<NoTimestamp, NoClassId, rf_frequency_offset>;
    alignas(4) std::array<uint8_t, RFOffsetContext::size_bytes()> buffer{};
    RFOffsetContext packet(buffer);

    // Per VITA 49.2 spec observations:
    // 0x0000000000100000 = +1 Hz
    packet[rf_frequency_offset].set_encoded(0x0000000000100000ULL);
    EXPECT_NEAR(packet[rf_frequency_offset].value(), 1.0, 1e-6);

    // 0xFFFFFFFFFFF00000 = -1 Hz
    packet[rf_frequency_offset].set_encoded(0xFFFFFFFFFFF00000ULL);
    EXPECT_NEAR(packet[rf_frequency_offset].value(), -1.0, 1e-6);

    // 0x0000000000000001 = +0.95 µHz
    packet[rf_frequency_offset].set_encoded(0x0000000000000001ULL);
    EXPECT_NEAR(packet[rf_frequency_offset].value(), 9.5367431640625e-7, 1e-12);
}

TEST_F(ContextPacketTest, CIF0_26_RuntimeAccess) {
    // Create a compile-time packet
    using RFOffsetContext =
        typed::ContextPacketBuilder<NoTimestamp, NoClassId, rf_frequency_offset>;
    alignas(4) std::array<uint8_t, RFOffsetContext::size_bytes()> buffer{};
    RFOffsetContext packet(buffer);

    packet[rf_frequency_offset].set_value(1.0e6);

    // Parse with runtime packet
    auto result = dynamic::ContextPacketView::parse(buffer);
    ASSERT_TRUE(result.ok()) << result.error().message();
    const auto& runtime_packet = result.value();

    if (runtime_packet[rf_frequency_offset]) {
        EXPECT_NEAR(runtime_packet[rf_frequency_offset].value(), 1.0e6, 1.0);
    }
}
