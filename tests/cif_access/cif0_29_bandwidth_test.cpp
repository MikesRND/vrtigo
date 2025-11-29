// Self-Documenting CIF Access Test
// Demonstrates accessing and manipulating the Bandwidth CIF field
//
// This file auto-generates: docs/cif_access/cif0_29_bandwidth.md

#include "../context_test_fixture.hpp"

using namespace vrtigo;
using namespace vrtigo::field;

// [TITLE]
// Bandwidth Field (CIF0 bit 29)
// [/TITLE]

TEST_F(ContextPacketTest, CIF0_29_BasicAccess) {
    // [EXAMPLE]
    // Setting and Reading Bandwidth
    // [/EXAMPLE]

    // [DESCRIPTION]
    // The Bandwidth field is a 64-bit Q52.12 fixed-point value representing
    // the signal bandwidth in Hz.
    // [/DESCRIPTION]

    // [SNIPPET]
    using BandwidthContext = typed::ContextPacket<NoTimestamp, NoClassId, bandwidth>;

    alignas(4) std::array<uint8_t, BandwidthContext::size_bytes()> buffer{};
    BandwidthContext packet(buffer);

    // Set bandwidth to 20 MHz
    packet[bandwidth].set_value(20'000'000.0);

    // Read back the value in Hz
    double bw_hz = packet[bandwidth].value();

    // Can also access the encoded Q52.12 value directly
    uint64_t encoded = packet[bandwidth].encoded();
    // [/SNIPPET]

    // Assertions
    EXPECT_DOUBLE_EQ(bw_hz, 20'000'000.0);
    EXPECT_EQ(encoded >> 12, 20'000'000ULL);
}
