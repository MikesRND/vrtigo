// Self-Documenting CIF Access Test
// Demonstrates accessing and manipulating the Sample Rate CIF field
//
// This file auto-generates: docs/cif_access/cif0_21_sample_rate.md

#include "../context_test_fixture.hpp"

using namespace vrtigo;
using namespace vrtigo::field;

// [TITLE]
// Sample Rate Field (CIF0 bit 21)
// [/TITLE]

TEST_F(ContextPacketTest, CIF0_21_BasicAccess) {
    // [EXAMPLE]
    // Setting and Reading Sample Rate
    // [/EXAMPLE]

    // [DESCRIPTION]
    // The Sample Rate field is a 64-bit Q52.12 fixed-point value representing
    // the sample rate in Hz. Set and read the encoded value directly.
    // [/DESCRIPTION]

    // [SNIPPET]
    using SampleRateContext = typed::ContextPacket<NoTimestamp, NoClassId, sample_rate>;

    alignas(4) std::array<uint8_t, SampleRateContext::size_bytes()> buffer{};
    SampleRateContext packet(buffer);

    // Set sample rate to 10 MHz
    packet[sample_rate].set_value(10'000'000.0);

    // Read back the value in Hz
    double rate_hz = packet[sample_rate].value();

    // Can also access the encoded Q52.12 value directly
    uint64_t encoded = packet[sample_rate].encoded();
    // [/SNIPPET]

    // Assertions
    EXPECT_DOUBLE_EQ(rate_hz, 10'000'000.0);
    EXPECT_EQ(encoded >> 12, 10'000'000ULL);
}
