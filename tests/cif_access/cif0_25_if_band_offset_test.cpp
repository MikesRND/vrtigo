// Self-Documenting CIF Access Test
// Demonstrates accessing and manipulating the IF Band Offset field
//
// This file auto-generates: docs/cif_access/cif0_25_if_band_offset.md

#include "../context_test_fixture.hpp"

using namespace vrtigo;
using namespace vrtigo::field;

// [TITLE]
// IF Band Offset Field (CIF0 bit 25)
// [/TITLE]

TEST_F(ContextPacketTest, CIF0_25_BasicAccess) {
    // [EXAMPLE]
    // Setting and Reading IF Band Offset
    // [/EXAMPLE]

    // [DESCRIPTION]
    // The IF Band Offset field specifies the frequency offset from IF Reference Frequency
    // to the center of the band. Band Center = IF Reference Frequency + IF Band Offset.
    // Uses 64-bit two's complement Q44.20 fixed-point format.
    // [/DESCRIPTION]

    // [SNIPPET]
    using IFOffsetContext = typed::ContextPacket<NoTimestamp, NoClassId, if_band_offset>;

    alignas(4) std::array<uint8_t, IFOffsetContext::size_bytes()> buffer{};
    IFOffsetContext packet(buffer);

    // Set IF band offset to 500 kHz
    packet[if_band_offset].set_value(500.0e3);

    // Read back the value in Hz
    double offset_hz = packet[if_band_offset].value();
    // [/SNIPPET]

    // Assertions
    EXPECT_NEAR(offset_hz, 500.0e3, 1.0); // Within 1 Hz tolerance
}

// Additional tests (not included in documentation)

TEST_F(ContextPacketTest, CIF0_25_PositiveAndNegativeOffsets) {
    using IFOffsetContext = typed::ContextPacket<NoTimestamp, NoClassId, if_band_offset>;
    alignas(4) std::array<uint8_t, IFOffsetContext::size_bytes()> buffer{};
    IFOffsetContext packet(buffer);

    // Test positive and negative offsets
    double test_offsets[] = {
        0.0,      // No offset (band center at IF ref freq)
        100.0e3,  // +100 kHz (band center above IF ref)
        500.0e3,  // +500 kHz
        1.0e6,    // +1 MHz
        -100.0e3, // -100 kHz (band center below IF ref)
        -500.0e3, // -500 kHz
        -1.0e6    // -1 MHz
    };

    for (double offset : test_offsets) {
        packet[if_band_offset].set_value(offset);
        double readback = packet[if_band_offset].value();
        EXPECT_NEAR(readback, offset, std::abs(offset) * 1e-6 + 1e-3); // Within 1 ppm + 1 mHz
    }
}

TEST_F(ContextPacketTest, CIF0_25_SpecExamples) {
    using IFOffsetContext = typed::ContextPacket<NoTimestamp, NoClassId, if_band_offset>;
    alignas(4) std::array<uint8_t, IFOffsetContext::size_bytes()> buffer{};
    IFOffsetContext packet(buffer);

    // Per VITA 49.2 spec observations:
    // 0x0000000000100000 = +1 Hz
    packet[if_band_offset].set_encoded(0x0000000000100000ULL);
    EXPECT_NEAR(packet[if_band_offset].value(), 1.0, 1e-6);

    // 0xFFFFFFFFFFF00000 = -1 Hz
    packet[if_band_offset].set_encoded(0xFFFFFFFFFFF00000ULL);
    EXPECT_NEAR(packet[if_band_offset].value(), -1.0, 1e-6);

    // 0x0000000000000001 = +0.95 µHz
    packet[if_band_offset].set_encoded(0x0000000000000001ULL);
    EXPECT_NEAR(packet[if_band_offset].value(), 9.5367431640625e-7, 1e-12);
}

TEST_F(ContextPacketTest, CIF0_25_RuntimeAccess) {
    // Create a compile-time packet
    using IFOffsetContext = typed::ContextPacket<NoTimestamp, NoClassId, if_band_offset>;
    alignas(4) std::array<uint8_t, IFOffsetContext::size_bytes()> buffer{};
    IFOffsetContext packet(buffer);

    packet[if_band_offset].set_value(500.0e3);

    // Parse with runtime packet
    auto result = dynamic::ContextPacket::parse(buffer);
    ASSERT_TRUE(result.ok()) << result.error().message();
    const auto& runtime_packet = result.value();

    if (runtime_packet[if_band_offset]) {
        EXPECT_NEAR(runtime_packet[if_band_offset].value(), 500.0e3, 1.0);
    }
}
