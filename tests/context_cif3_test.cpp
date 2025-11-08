#include "context_test_fixture.hpp"

TEST_F(ContextPacketTest, CIF3FieldsBasic) {
    // Test a selection of 1-word and 2-word CIF3 fields
    constexpr uint32_t cif3_mask = cif3::NETWORK_ID | cif3::TROPOSPHERIC_STATE |
                                   cif3::JITTER | cif3::PULSE_WIDTH;
    using TestContext = ContextPacket<
        true,
        NoTimeStamp,
        NoClassId,
        0, 0, 0,        // No CIF0, CIF1, CIF2
        cif3_mask,      // CIF3 with environmental and temporal fields
        false
    >;

    TestContext packet(buffer.data());

    // Test 1-word fields
    set(packet, field::network_id, 0x11111111);
    EXPECT_EQ(get(packet, field::network_id).value(), 0x11111111);

    set(packet, field::tropospheric_state, 0x22222222);
    EXPECT_EQ(get(packet, field::tropospheric_state).value(), 0x22222222);

    // Test 2-word (64-bit) fields
    set(packet, field::jitter, 0x3333333344444444ULL);
    EXPECT_EQ(get(packet, field::jitter).value(), 0x3333333344444444ULL);

    set(packet, field::pulse_width, 0x5555555566666666ULL);
    EXPECT_EQ(get(packet, field::pulse_width).value(), 0x5555555566666666ULL);
}

TEST_F(ContextPacketTest, RuntimeParseCIF3) {
    // Build a packet with CIF3 enabled
    // Structure: header(1) + CIF0(1) + CIF3(1) + network_id(1) = 4 words
    uint32_t header = (static_cast<uint32_t>(packet_type::context) << header::PACKET_TYPE_SHIFT) | 4;
    cif::write_u32_safe(buffer.data(), 0, header);

    // CIF0 with CIF3 enable bit set
    uint32_t cif0_val = (1U << cif::CIF3_ENABLE_BIT);
    cif::write_u32_safe(buffer.data(), 4, cif0_val);

    // CIF3 with network_id field
    uint32_t cif3_val = cif3::NETWORK_ID;
    cif::write_u32_safe(buffer.data(), 8, cif3_val);

    // Network ID value
    cif::write_u32_safe(buffer.data(), 12, 0xDEADBEEF);

    // Parse and validate
    ContextPacketView view(buffer.data(), 4 * 4);
    EXPECT_EQ(view.validate(), validation_error::none);
    EXPECT_TRUE(view.is_valid());

    // Verify CIF3 is present
    EXPECT_EQ(view.cif3(), cif3::NETWORK_ID);

    // Verify field value
    EXPECT_TRUE(has(view, field::network_id));
    EXPECT_EQ(get(view, field::network_id).value(), 0xDEADBEEF);
}

TEST_F(ContextPacketTest, CompileTimeCIF3) {
    constexpr uint32_t cif3_mask = cif3::NETWORK_ID | cif3::TROPOSPHERIC_STATE;
    using TestContext = ContextPacket<
        true,           // Has stream ID
        NoTimeStamp,
        NoClassId,
        0, 0, 0,        // No CIF0, CIF1, CIF2
        cif3_mask,      // CIF3 with two fields
        false
    >;

    TestContext packet(buffer.data());

    // Set field values
    set(packet, field::network_id, 0x11111111);
    set(packet, field::tropospheric_state, 0x22222222);

    // Verify CIF3 word is written correctly
    EXPECT_EQ(packet.cif3(), cif3_mask);

    // Verify field values are correct (not overwriting CIF3 word)
    EXPECT_EQ(get(packet, field::network_id).value(), 0x11111111);
    EXPECT_EQ(get(packet, field::tropospheric_state).value(), 0x22222222);

    // Parse as runtime packet to verify structure
    ContextPacketView view(buffer.data(), TestContext::size_bytes);
    EXPECT_EQ(view.validate(), validation_error::none);
    EXPECT_EQ(view.cif3(), cif3_mask);
    EXPECT_EQ(get(view, field::network_id).value(), 0x11111111);
    EXPECT_EQ(get(view, field::tropospheric_state).value(), 0x22222222);
}

