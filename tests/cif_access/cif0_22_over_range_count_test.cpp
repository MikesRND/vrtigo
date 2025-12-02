// Self-Documenting CIF Access Test
// Demonstrates accessing and manipulating the Over-Range Count field
//
// This file auto-generates: docs/cif_access/cif0_22_over_range_count.md

#include "../context_test_fixture.hpp"

using namespace vrtigo;
using namespace vrtigo::field;

// [TITLE]
// Over-Range Count Field (CIF0 bit 22)
// [/TITLE]

TEST_F(ContextPacketTest, CIF0_22_BasicAccess) {
    // [EXAMPLE]
    // Setting and Reading Over-Range Count
    // [/EXAMPLE]

    // [DESCRIPTION]
    // The Over-Range Count field contains the number of Data Samples in the paired
    // Data Packet whose amplitudes were beyond the range of the Data Item format.
    // Per VITA 49.2 Rule 9.10.6-1, this count applies only to the paired Data Packet
    // with the corresponding Timestamp and does not accumulate over multiple packets.
    // For complex Cartesian samples, the count includes samples where either the real
    // or imaginary component was beyond range (Rule 9.10.6-2).
    // [/DESCRIPTION]

    // [SNIPPET]
    using OverRangeContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, over_range_count>;

    alignas(4) std::array<uint8_t, OverRangeContext::size_bytes()> buffer{};
    OverRangeContext packet(buffer);

    // Set Over-Range Count to indicate 42 over-range samples
    uint32_t count = 42;
    packet[over_range_count].set_value(count);

    // Read back the count
    uint32_t read_count = packet[over_range_count].value();
    // [/SNIPPET]

    // Assertions
    EXPECT_EQ(read_count, count);
}

// Additional tests (not included in documentation)

TEST_F(ContextPacketTest, CIF0_22_ZeroCount) {
    using OverRangeContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, over_range_count>;
    alignas(4) std::array<uint8_t, OverRangeContext::size_bytes()> buffer{};
    OverRangeContext packet(buffer);

    // Zero indicates no over-range samples in the paired Data packet
    packet[over_range_count].set_value(0);
    EXPECT_EQ(packet[over_range_count].value(), 0);
}

TEST_F(ContextPacketTest, CIF0_22_MaxCount) {
    using OverRangeContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, over_range_count>;
    alignas(4) std::array<uint8_t, OverRangeContext::size_bytes()> buffer{};
    OverRangeContext packet(buffer);

    // Maximum 32-bit value per Rule 9.10.6-3
    uint32_t max_count = 0xFFFFFFFF;
    packet[over_range_count].set_value(max_count);
    EXPECT_EQ(packet[over_range_count].value(), max_count);
}

TEST_F(ContextPacketTest, CIF0_22_MultipleValues) {
    using OverRangeContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, over_range_count>;
    alignas(4) std::array<uint8_t, OverRangeContext::size_bytes()> buffer{};
    OverRangeContext packet(buffer);

    // Test various count values
    uint32_t test_counts[] = {0, 1, 1000, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF};

    for (uint32_t count : test_counts) {
        packet[over_range_count].set_value(count);
        EXPECT_EQ(packet[over_range_count].value(), count);
    }
}

TEST_F(ContextPacketTest, CIF0_22_RuntimeAccess) {
    // Create a compile-time packet with Over-Range Count
    using OverRangeContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, over_range_count>;
    alignas(4) std::array<uint8_t, OverRangeContext::size_bytes()> buffer{};
    OverRangeContext packet(buffer);

    uint32_t count = 12345;
    packet[over_range_count].set_value(count);

    // Parse with runtime packet
    auto result = dynamic::ContextPacketView::parse(buffer);
    ASSERT_TRUE(result.has_value()) << result.error().message();
    const auto& runtime_packet = result.value();

    // Runtime packet can read the Over-Range Count
    if (runtime_packet[over_range_count]) {
        EXPECT_EQ(runtime_packet[over_range_count].value(), count);
    }
}

TEST_F(ContextPacketTest, CIF0_22_WithOtherFields) {
    // Over-Range Count combined with related fields
    using CombinedContext =
        typed::ContextPacketBuilder<NoTimestamp, NoClassId, sample_rate, over_range_count, gain>;

    alignas(4) std::array<uint8_t, CombinedContext::size_bytes()> buffer{};
    CombinedContext packet(buffer);

    // Set all fields
    packet[sample_rate].set_value(100'000'000.0); // 100 MHz
    packet[over_range_count].set_value(500);
    packet[gain].set_value({10.0, 5.0}); // +10 dB stage1, +5 dB stage2

    // Verify over_range_count is independent
    EXPECT_EQ(packet[over_range_count].value(), 500);

    // Verify all fields maintained
    EXPECT_NEAR(packet[sample_rate].value(), 100'000'000.0, 1.0);
    auto gain_value = packet[gain].value();
    EXPECT_NEAR(gain_value.stage1_db, 10.0, 0.01);
    EXPECT_NEAR(gain_value.stage2_db, 5.0, 0.01);
}
