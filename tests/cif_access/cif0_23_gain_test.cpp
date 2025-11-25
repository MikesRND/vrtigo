// Self-Documenting CIF Access Test
// Demonstrates accessing and manipulating the Gain/Attenuation field
//
// This file auto-generates: docs/cif_access/cif0_23_gain.md

#include "../context_test_fixture.hpp"

using namespace vrtigo;
using namespace vrtigo::field;

// [TITLE]
// Gain/Attenuation Field (CIF0 bit 23)
// [/TITLE]

TEST_F(ContextPacketTest, CIF0_23_BasicAccess) {
    // [EXAMPLE]
    // Setting and Reading Gain Values
    // [/EXAMPLE]

    // [DESCRIPTION]
    // The Gain field describes the signal gain or attenuation from the Reference Point
    // to the Described Signal. It contains two 16-bit Q9.7 fixed-point subfields:
    // - Stage 1 Gain (lower 16 bits): Front-end or RF gain
    // - Stage 2 Gain (upper 16 bits): Back-end or IF gain
    //
    // Each subfield provides ±256 dB range with 1/128 dB resolution (0.0078125 dB).
    // Single-stage equipment uses Stage 1 only with Stage 2 set to zero.
    // [/DESCRIPTION]

    // [SNIPPET]
    using GainContext = ContextPacket<NoTimestamp, NoClassId, gain>;

    alignas(4) std::array<uint8_t, GainContext::size_bytes> buffer{};
    GainContext packet(buffer.data());

    // Set single-stage gain of +10 dB (Stage 2 = 0)
    packet[gain].set_value({10.0, 0.0});

    // Read back the gain value
    auto gain_value = packet[gain].value();
    double stage1 = gain_value.stage1_db;
    double stage2 = gain_value.stage2_db;
    double total = gain_value.total_db();
    // [/SNIPPET]

    // Assertions
    EXPECT_NEAR(stage1, 10.0, 0.01);
    EXPECT_NEAR(stage2, 0.0, 0.01);
    EXPECT_NEAR(total, 10.0, 0.01);
}

// Additional tests (not included in documentation)

TEST_F(ContextPacketTest, CIF0_23_DualStageGain) {
    using GainContext = ContextPacket<NoTimestamp, NoClassId, gain>;
    alignas(4) std::array<uint8_t, GainContext::size_bytes> buffer{};
    GainContext packet(buffer.data());

    // Set dual-stage gains: RF gain +20 dB, IF gain +15 dB
    packet[gain].set_value({20.0, 15.0});

    auto gain_value = packet[gain].value();
    EXPECT_NEAR(gain_value.stage1_db, 20.0, 0.01);
    EXPECT_NEAR(gain_value.stage2_db, 15.0, 0.01);
    EXPECT_NEAR(gain_value.total_db(), 35.0, 0.01); // Total = 20 + 15
}

TEST_F(ContextPacketTest, CIF0_23_Attenuation) {
    using GainContext = ContextPacket<NoTimestamp, NoClassId, gain>;
    alignas(4) std::array<uint8_t, GainContext::size_bytes> buffer{};
    GainContext packet(buffer.data());

    // Test negative gains (attenuation)
    packet[gain].set_value({-10.0, -5.0});

    auto gain_value = packet[gain].value();
    EXPECT_NEAR(gain_value.stage1_db, -10.0, 0.01);
    EXPECT_NEAR(gain_value.stage2_db, -5.0, 0.01);
    EXPECT_NEAR(gain_value.total_db(), -15.0, 0.01);
}

TEST_F(ContextPacketTest, CIF0_23_SpecExamples) {
    using GainContext = ContextPacket<NoTimestamp, NoClassId, gain>;
    alignas(4) std::array<uint8_t, GainContext::size_bytes> buffer{};
    GainContext packet(buffer.data());

    // Per VITA 49.2 Observation 9.5.3-4:

    // 0x00000080 = Stage 1: +1 dB, Stage 2: 0 dB
    packet[gain].set_encoded(0x00000080);
    {
        auto g = packet[gain].value();
        EXPECT_NEAR(g.stage1_db, 1.0, 0.001);
        EXPECT_NEAR(g.stage2_db, 0.0, 0.001);
    }

    // 0x0000FF80 = Stage 1: -1 dB, Stage 2: 0 dB
    packet[gain].set_encoded(0x0000FF80);
    {
        auto g = packet[gain].value();
        EXPECT_NEAR(g.stage1_db, -1.0, 0.001);
        EXPECT_NEAR(g.stage2_db, 0.0, 0.001);
    }

    // 0x00000001 = Stage 1: +0.0078125 dB (1/128), Stage 2: 0 dB
    packet[gain].set_encoded(0x00000001);
    {
        auto g = packet[gain].value();
        EXPECT_NEAR(g.stage1_db, 0.0078125, 0.0001);
        EXPECT_NEAR(g.stage2_db, 0.0, 0.001);
    }

    // 0x0000FFFF = Stage 1: -0.0078125 dB (-1/128), Stage 2: 0 dB
    packet[gain].set_encoded(0x0000FFFF);
    {
        auto g = packet[gain].value();
        EXPECT_NEAR(g.stage1_db, -0.0078125, 0.0001);
        EXPECT_NEAR(g.stage2_db, 0.0, 0.001);
    }

    // Per VITA 49.2 Observation 9.5.3-5:

    // 0x00800080 = Both stages: +1 dB
    packet[gain].set_encoded(0x00800080);
    {
        auto g = packet[gain].value();
        EXPECT_NEAR(g.stage1_db, 1.0, 0.001);
        EXPECT_NEAR(g.stage2_db, 1.0, 0.001);
        EXPECT_NEAR(g.total_db(), 2.0, 0.001);
    }

    // 0xFF80FF80 = Both stages: -1 dB
    packet[gain].set_encoded(0xFF80FF80);
    {
        auto g = packet[gain].value();
        EXPECT_NEAR(g.stage1_db, -1.0, 0.001);
        EXPECT_NEAR(g.stage2_db, -1.0, 0.001);
        EXPECT_NEAR(g.total_db(), -2.0, 0.001);
    }

    // 0x00010001 = Both stages: +0.0078125 dB
    packet[gain].set_encoded(0x00010001);
    {
        auto g = packet[gain].value();
        EXPECT_NEAR(g.stage1_db, 0.0078125, 0.0001);
        EXPECT_NEAR(g.stage2_db, 0.0078125, 0.0001);
        EXPECT_NEAR(g.total_db(), 0.015625, 0.0001);
    }

    // 0xFFFFFFFF = Both stages: -0.0078125 dB
    packet[gain].set_encoded(0xFFFFFFFF);
    {
        auto g = packet[gain].value();
        EXPECT_NEAR(g.stage1_db, -0.0078125, 0.0001);
        EXPECT_NEAR(g.stage2_db, -0.0078125, 0.0001);
        EXPECT_NEAR(g.total_db(), -0.015625, 0.0001);
    }
}

TEST_F(ContextPacketTest, CIF0_23_CommonGainValues) {
    using GainContext = ContextPacket<NoTimestamp, NoClassId, gain>;
    alignas(4) std::array<uint8_t, GainContext::size_bytes> buffer{};
    GainContext packet(buffer.data());

    // Test typical single-stage gain values
    struct TestCase {
        double gain_db;
        const char* description;
    };

    TestCase test_cases[] = {
        {0.0, "Unity gain"}, {10.0, "Low gain"},         {30.0, "Medium gain"},
        {60.0, "High gain"}, {-3.0, "3 dB attenuation"}, {-10.0, "10 dB attenuation"},
    };

    for (const auto& tc : test_cases) {
        packet[gain].set_value({tc.gain_db, 0.0}); // Single stage
        auto gain_value = packet[gain].value();
        EXPECT_NEAR(gain_value.stage1_db, tc.gain_db, 0.01) << tc.description;
        EXPECT_NEAR(gain_value.stage2_db, 0.0, 0.01) << tc.description;
        EXPECT_NEAR(gain_value.total_db(), tc.gain_db, 0.01) << tc.description;
    }
}

TEST_F(ContextPacketTest, CIF0_23_Saturation) {
    using GainContext = ContextPacket<NoTimestamp, NoClassId, gain>;
    alignas(4) std::array<uint8_t, GainContext::size_bytes> buffer{};
    GainContext packet(buffer.data());

    // Test saturation at bounds (±256 dB per stage)

    // Above maximum
    packet[gain].set_value({300.0, 300.0});
    {
        auto g = packet[gain].value();
        EXPECT_NEAR(g.stage1_db, 255.9921875, 0.001); // Max Q9.7
        EXPECT_NEAR(g.stage2_db, 255.9921875, 0.001);
    }

    // Below minimum
    packet[gain].set_value({-300.0, -300.0});
    {
        auto g = packet[gain].value();
        EXPECT_NEAR(g.stage1_db, -256.0, 0.001); // Min Q9.7
        EXPECT_NEAR(g.stage2_db, -256.0, 0.001);
    }
}

TEST_F(ContextPacketTest, CIF0_23_RuntimeAccess) {
    // Create a compile-time packet
    using GainContext = ContextPacket<NoTimestamp, NoClassId, gain>;
    alignas(4) std::array<uint8_t, GainContext::size_bytes> buffer{};
    GainContext packet(buffer.data());

    packet[gain].set_value({15.0, 10.0});

    // Parse with runtime packet
    RuntimeContextPacket runtime_packet(buffer.data(), buffer.size());

    EXPECT_TRUE(runtime_packet.is_valid());
    if (runtime_packet[gain]) {
        auto g = runtime_packet[gain].value();
        EXPECT_NEAR(g.stage1_db, 15.0, 0.01);
        EXPECT_NEAR(g.stage2_db, 10.0, 0.01);
        EXPECT_NEAR(g.total_db(), 25.0, 0.01);
    }
}
