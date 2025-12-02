// Self-Documenting CIF Access Test
// Demonstrates accessing and manipulating the Reference Level field
//
// This file auto-generates: docs/cif_access/cif0_24_reference_level.md

#include "../context_test_fixture.hpp"

using namespace vrtigo;
using namespace vrtigo::field;

// [TITLE]
// Reference Level Field (CIF0 bit 24)
// [/TITLE]

TEST_F(ContextPacketTest, CIF0_24_BasicAccess) {
    // [EXAMPLE]
    // Setting and Reading Reference Level
    // [/EXAMPLE]

    // [DESCRIPTION]
    // The Reference Level field specifies the reference level at the Reference Point
    // in dBm. It uses a 16-bit two's complement Q9.7 fixed-point format stored in
    // the lower 16 bits of a 32-bit word, providing ±256 dBm range with 1/128 dBm
    // resolution (approximately 0.0078 dBm).
    // [/DESCRIPTION]

    // [SNIPPET]
    using RefLevelContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, reference_level>;

    alignas(4) std::array<uint8_t, RefLevelContext::size_bytes()> buffer{};
    RefLevelContext packet(buffer);

    // Set reference level to -10 dBm (typical receiver reference)
    packet[reference_level].set_value(-10.0);

    // Read back the value in dBm
    double level_dbm = packet[reference_level].value();
    // [/SNIPPET]

    // Assertions
    EXPECT_NEAR(level_dbm, -10.0, 0.01); // Within 0.01 dBm tolerance
}

// Additional tests (not included in documentation)

TEST_F(ContextPacketTest, CIF0_24_CommonLevels) {
    using RefLevelContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, reference_level>;
    alignas(4) std::array<uint8_t, RefLevelContext::size_bytes()> buffer{};
    RefLevelContext packet(buffer);

    // Test common reference levels
    double test_levels[] = {
        0.0,   // 0 dBm (1 mW)
        -10.0, // -10 dBm (0.1 mW)
        -30.0, // -30 dBm (1 µW)
        10.0,  // +10 dBm (10 mW)
        -50.0, // -50 dBm (10 nW)
        20.0   // +20 dBm (100 mW)
    };

    for (double level : test_levels) {
        packet[reference_level].set_value(level);
        double readback = packet[reference_level].value();
        EXPECT_NEAR(readback, level, 0.01); // Within 0.01 dBm
    }
}

TEST_F(ContextPacketTest, CIF0_24_SpecExamples) {
    using RefLevelContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, reference_level>;
    alignas(4) std::array<uint8_t, RefLevelContext::size_bytes()> buffer{};
    RefLevelContext packet(buffer);

    // Per VITA 49.2 spec observations for Q9.7:
    // Scale factor = 2^7 = 128

    // 0x0080 = +1 dBm (128/128)
    packet[reference_level].set_encoded(0x00000080); // Upper 16 bits reserved
    EXPECT_NEAR(packet[reference_level].value(), 1.0, 0.001);

    // 0xFF80 = -1 dBm (two's complement: -128/128)
    packet[reference_level].set_encoded(0x0000FF80);
    EXPECT_NEAR(packet[reference_level].value(), -1.0, 0.001);

    // 0x0001 = +1/128 dBm ≈ 0.0078 dBm
    packet[reference_level].set_encoded(0x00000001);
    EXPECT_NEAR(packet[reference_level].value(), 0.0078125, 0.0001);

    // 0xFFFF = -1/128 dBm ≈ -0.0078 dBm
    packet[reference_level].set_encoded(0x0000FFFF);
    EXPECT_NEAR(packet[reference_level].value(), -0.0078125, 0.0001);

    // Maximum positive: 0x7FFF = +255.992 dBm
    packet[reference_level].set_encoded(0x00007FFF);
    EXPECT_NEAR(packet[reference_level].value(), 255.9921875, 0.001);

    // Maximum negative: 0x8000 = -256 dBm
    packet[reference_level].set_encoded(0x00008000);
    EXPECT_NEAR(packet[reference_level].value(), -256.0, 0.001);
}

TEST_F(ContextPacketTest, CIF0_24_Saturation) {
    using RefLevelContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, reference_level>;
    alignas(4) std::array<uint8_t, RefLevelContext::size_bytes()> buffer{};
    RefLevelContext packet(buffer);

    // Test saturation at bounds

    // Above maximum (+255.992 dBm)
    packet[reference_level].set_value(300.0);
    EXPECT_NEAR(packet[reference_level].value(), 255.9921875, 0.001);

    // Below minimum (-256 dBm)
    packet[reference_level].set_value(-300.0);
    EXPECT_NEAR(packet[reference_level].value(), -256.0, 0.001);

    // Within range
    packet[reference_level].set_value(50.0);
    EXPECT_NEAR(packet[reference_level].value(), 50.0, 0.01);
}

TEST_F(ContextPacketTest, CIF0_24_RuntimeAccess) {
    // Create a compile-time packet
    using RefLevelContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, reference_level>;
    alignas(4) std::array<uint8_t, RefLevelContext::size_bytes()> buffer{};
    RefLevelContext packet(buffer);

    packet[reference_level].set_value(-20.0);

    // Parse with runtime packet
    auto result = dynamic::ContextPacketView::parse(buffer);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    const auto& runtime_packet = result.value();

    if (runtime_packet[reference_level]) {
        EXPECT_NEAR(runtime_packet[reference_level].value(), -20.0, 0.01);
    }
}

TEST_F(ContextPacketTest, CIF0_24_ReservedBits) {
    using RefLevelContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, reference_level>;
    alignas(4) std::array<uint8_t, RefLevelContext::size_bytes()> buffer{};
    RefLevelContext packet(buffer);

    // Verify that upper 16 bits are preserved as reserved (set to 0)
    packet[reference_level].set_value(-10.0);
    uint32_t raw = packet[reference_level].encoded();

    // Upper 16 bits should be 0
    EXPECT_EQ(raw & 0xFFFF0000, 0u);

    // Lower 16 bits should contain the Q9.7 value
    EXPECT_NE(raw & 0x0000FFFF, 0u);
}