// Self-Documenting CIF Access Test
// Demonstrates accessing and manipulating the Reference Point Identifier field
//
// This file auto-generates: docs/cif_access/cif0_30_reference_point_id.md

#include "../context_test_fixture.hpp"

using namespace vrtigo;
using namespace vrtigo::field;

// [TITLE]
// Reference Point Identifier Field (CIF0 bit 30)
// [/TITLE]

TEST_F(ContextPacketTest, CIF0_30_BasicAccess) {
    // [EXAMPLE]
    // Setting and Reading Reference Point ID
    // [/EXAMPLE]

    // [DESCRIPTION]
    // The Reference Point Identifier field contains the Stream ID of the Reference
    // Point where VRT timestamps are measured. Per VITA 49.2, this identifies the
    // point in the signal processing architecture (often an analog signal like an
    // air interface) where the timing reference applies.
    // [/DESCRIPTION]

    // [SNIPPET]
    using RefPointContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, reference_point_id>;

    alignas(4) std::array<uint8_t, RefPointContext::size_bytes()> buffer{};
    RefPointContext packet(buffer);

    // Set Reference Point ID to a Stream ID
    uint32_t ref_point_sid = 0x12345678;
    packet[reference_point_id].set_value(ref_point_sid);

    // Read back the Stream ID
    uint32_t read_sid = packet[reference_point_id].value();
    // [/SNIPPET]

    // Assertions
    EXPECT_EQ(read_sid, ref_point_sid);
}

// Additional tests (not included in documentation)

TEST_F(ContextPacketTest, CIF0_30_MultipleValues) {
    using RefPointContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, reference_point_id>;
    alignas(4) std::array<uint8_t, RefPointContext::size_bytes()> buffer{};
    RefPointContext packet(buffer);

    // Test different Stream IDs
    uint32_t test_sids[] = {0x00000000, 0xFFFFFFFF, 0xABCD1234, 0x80000001};

    for (uint32_t sid : test_sids) {
        packet[reference_point_id].set_value(sid);
        EXPECT_EQ(packet[reference_point_id].value(), sid);
    }
}

TEST_F(ContextPacketTest, CIF0_30_RuntimeAccess) {
    // Create a compile-time packet with Reference Point ID
    using RefPointContext = typed::ContextPacketBuilder<NoTimestamp, NoClassId, reference_point_id>;
    alignas(4) std::array<uint8_t, RefPointContext::size_bytes()> buffer{};
    RefPointContext packet(buffer);

    uint32_t ref_point_sid = 0xCAFEBABE;
    packet[reference_point_id].set_value(ref_point_sid);

    // Parse with runtime packet
    auto result = dynamic::ContextPacketView::parse(buffer);
    ASSERT_TRUE(result.ok()) << result.error().message();
    const auto& runtime_packet = result.value();

    // Runtime packet can read the Reference Point ID
    if (runtime_packet[reference_point_id]) {
        EXPECT_EQ(runtime_packet[reference_point_id].value(), ref_point_sid);
    }
}
