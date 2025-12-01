// Self-Documenting CIF Access Test
// Demonstrates accessing and manipulating the Context Field Change Indicator
//
// This file auto-generates: docs/cif_access/cif0_31_change_indicator.md

#include "../context_test_fixture.hpp"

using namespace vrtigo;
using namespace vrtigo::field;

// [TITLE]
// Context Field Change Indicator (CIF0 bit 31)
// [/TITLE]

TEST_F(ContextPacketTest, CIF0_31_BasicAccess) {
    // [EXAMPLE]
    // Setting and Reading the Change Indicator
    // [/EXAMPLE]

    // [DESCRIPTION]
    // The Context Field Change Indicator (CIF0 bit 31) indicates whether any
    // context field values have changed since the previous context packet.
    // Per VITA 49.2: false = no changes, true = at least one field changed.
    //
    // > **Special Case:** Accessed via packet-level methods, not the field proxy pattern.
    // [/DESCRIPTION]

    // [SNIPPET]
    using TestContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, bandwidth>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);

    // Indicate that context fields have changed
    packet.set_change_indicator(true);

    // Read the indicator
    bool has_changes = packet.change_indicator();
    // [/SNIPPET]

    // Assertions
    EXPECT_TRUE(has_changes);
}

// Additional tests (not included in documentation)

TEST_F(ContextPacketTest, CIF0_31_ToggleIndicator) {
    using TestContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, bandwidth>;
    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);

    // Initially false after packet creation
    EXPECT_FALSE(packet.change_indicator());

    // Set to true
    packet.set_change_indicator(true);
    EXPECT_TRUE(packet.change_indicator());

    // Clear back to false
    packet.set_change_indicator(false);
    EXPECT_FALSE(packet.change_indicator());
}

TEST_F(ContextPacketTest, CIF0_31_RuntimeAccess) {
    // Create a compile-time packet with change indicator set
    using TestContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, bandwidth>;
    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);
    packet.set_change_indicator(true);

    // Parse with runtime packet
    auto result = dynamic::ContextPacketView::parse(buffer);
    ASSERT_TRUE(result.ok()) << result.error().message();
    const auto& runtime_packet = result.value();

    // Runtime packet can read the change indicator
    EXPECT_TRUE(runtime_packet.change_indicator());
}
