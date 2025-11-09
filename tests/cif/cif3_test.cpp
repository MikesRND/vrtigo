#include "../context_test_fixture.hpp"

TEST_F(ContextPacketTest, CIF3FieldsBasic) {
    // Test a selection of 1-word and 2-word CIF3 fields
    constexpr uint32_t cif3_mask = cif3::NETWORK_ID | cif3::TROPOSPHERIC_STATE |
                                   cif3::JITTER | cif3::PULSE_WIDTH;
    using TestContext = ContextPacket<
        true,
        NoTimeStamp,
        NoClassId,
        0, 0, 0,        // No CIF0, CIF1, CIF2
        cif3_mask>;

    TestContext packet(buffer.data());

    // Test 1-word fields
    get(packet, field::network_id).set_raw_value(0x11111111);
    EXPECT_EQ(get(packet, field::network_id).raw_value(), 0x11111111);

    get(packet, field::tropospheric_state).set_raw_value(0x22222222);
    EXPECT_EQ(get(packet, field::tropospheric_state).raw_value(), 0x22222222);

    // Test 2-word (64-bit) fields
    get(packet, field::jitter).set_raw_value(0x3333333344444444ULL);
    EXPECT_EQ(get(packet, field::jitter).raw_value(), 0x3333333344444444ULL);

    get(packet, field::pulse_width).set_raw_value(0x5555555566666666ULL);
    EXPECT_EQ(get(packet, field::pulse_width).raw_value(), 0x5555555566666666ULL);
}

TEST_F(ContextPacketTest, RuntimeParseCIF3) {
    // Build a packet with CIF3 enabled
    // Type 4 has stream ID: header(1) + stream_id(1) + CIF0(1) + CIF3(1) + network_id(1) = 5 words
    uint32_t header = (static_cast<uint32_t>(PacketType::Context) << header::PACKET_TYPE_SHIFT) | 5;
    cif::write_u32_safe(buffer.data(), 0, header);

    // Stream ID (type 4 has stream ID per VITA 49.2)
    cif::write_u32_safe(buffer.data(), 4, 0x12345678);

    // CIF0 with CIF3 enable bit set
    uint32_t cif0_val = (1U << cif::CIF3_ENABLE_BIT);
    cif::write_u32_safe(buffer.data(), 8, cif0_val);

    // CIF3 with network_id field
    uint32_t cif3_val = cif3::NETWORK_ID;
    cif::write_u32_safe(buffer.data(), 12, cif3_val);

    // Network ID value
    cif::write_u32_safe(buffer.data(), 16, 0xDEADBEEF);

    // Parse and validate
    ContextPacketView view(buffer.data(), 5 * 4);
    EXPECT_EQ(view.error(), validation_error::none);
    EXPECT_TRUE(view.is_valid());

    // Verify CIF3 is present
    EXPECT_EQ(view.cif3(), cif3::NETWORK_ID);

    // Verify field value
    EXPECT_TRUE(has(view, field::network_id));
    EXPECT_EQ(get(view, field::network_id).raw_value(), 0xDEADBEEF);
}

TEST_F(ContextPacketTest, CompileTimeCIF3) {
    constexpr uint32_t cif3_mask = cif3::NETWORK_ID | cif3::TROPOSPHERIC_STATE;
    using TestContext = ContextPacket<
        true,           // Has stream ID
        NoTimeStamp,
        NoClassId,
        0, 0, 0,        // No CIF0, CIF1, CIF2
        cif3_mask>;

    TestContext packet(buffer.data());

    // Set field values
    get(packet, field::network_id).set_raw_value(0x11111111);
    get(packet, field::tropospheric_state).set_raw_value(0x22222222);

    // Verify CIF3 word is written correctly
    EXPECT_EQ(packet.cif3(), cif3_mask);

    // Verify field values are correct (not overwriting CIF3 word)
    EXPECT_EQ(get(packet, field::network_id).raw_value(), 0x11111111);
    EXPECT_EQ(get(packet, field::tropospheric_state).raw_value(), 0x22222222);

    // Parse as runtime packet to verify structure
    ContextPacketView view(buffer.data(), TestContext::size_bytes);
    EXPECT_EQ(view.error(), validation_error::none);
    EXPECT_EQ(view.cif3(), cif3_mask);
    EXPECT_EQ(get(view, field::network_id).raw_value(), 0x11111111);
    EXPECT_EQ(get(view, field::tropospheric_state).raw_value(), 0x22222222);
}

