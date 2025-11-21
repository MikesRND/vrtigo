#include "../context_test_fixture.hpp"

TEST_F(ContextPacketTest, CIF3FieldsBasic) {
    // Test a selection of 1-word and 2-word CIF3 fields using field-based API
    using namespace vrtigo::field;
    using TestContext =
        ContextPacket<NoTimeStamp, NoClassId, network_id, tropospheric_state, jitter, pulse_width>;

    TestContext packet(buffer.data());

    // Test 1-word fields
    packet[field::network_id].set_encoded(0x11111111);
    EXPECT_EQ(packet[field::network_id].encoded(), 0x11111111);

    packet[field::tropospheric_state].set_encoded(0x22222222);
    EXPECT_EQ(packet[field::tropospheric_state].encoded(), 0x22222222);

    // Test 2-word (64-bit) fields
    packet[field::jitter].set_encoded(0x3333333344444444ULL);
    EXPECT_EQ(packet[field::jitter].encoded(), 0x3333333344444444ULL);

    packet[field::pulse_width].set_encoded(0x5555555566666666ULL);
    EXPECT_EQ(packet[field::pulse_width].encoded(), 0x5555555566666666ULL);
}

TEST_F(ContextPacketTest, RuntimeParseCIF3) {
    // Build a packet with CIF3 enabled
    // Type 4 has stream ID: header(1) + stream_id(1) + CIF0(1) + CIF3(1) + network_id(1) = 5 words
    uint32_t header = (static_cast<uint32_t>(PacketType::context) << header::packet_type_shift) | 5;
    cif::write_u32_safe(buffer.data(), 0, header);

    // Stream ID (type 4 has stream ID per VITA 49.2)
    cif::write_u32_safe(buffer.data(), 4, 0x12345678);

    // CIF0 with CIF3 enable bit set
    uint32_t cif0_val = (1U << cif::CIF3_ENABLE_BIT);
    cif::write_u32_safe(buffer.data(), 8, cif0_val);

    // CIF3 with network_id field
    using namespace vrtigo::field;
    uint32_t cif3_val = vrtigo::detail::field_bitmask<network_id>();
    cif::write_u32_safe(buffer.data(), 12, cif3_val);

    // Network ID value
    cif::write_u32_safe(buffer.data(), 16, 0xDEADBEEF);

    // Parse and validate
    RuntimeContextPacket view(buffer.data(), 5 * 4);
    EXPECT_EQ(view.error(), ValidationError::none);
    EXPECT_TRUE(view.is_valid());

    // Verify CIF3 is present
    EXPECT_EQ(view.cif3(), vrtigo::detail::field_bitmask<network_id>());

    // Verify field value
    EXPECT_TRUE(view[field::network_id]);
    EXPECT_EQ(view[field::network_id].encoded(), 0xDEADBEEF);
}

TEST_F(ContextPacketTest, CompileTimeCIF3) {
    using namespace vrtigo::field;
    using TestContext = ContextPacket<NoTimeStamp, NoClassId, network_id, tropospheric_state>;

    TestContext packet(buffer.data());

    // Set field values
    packet[network_id].set_encoded(0x11111111);
    packet[tropospheric_state].set_encoded(0x22222222);

    // Verify CIF3 word is written correctly
    constexpr uint32_t expected_cif3 = vrtigo::detail::field_bitmask<network_id>() |
                                       vrtigo::detail::field_bitmask<tropospheric_state>();
    EXPECT_EQ(packet.cif3(), expected_cif3);

    // Verify field values are correct (not overwriting CIF3 word)
    EXPECT_EQ(packet[network_id].encoded(), 0x11111111);
    EXPECT_EQ(packet[tropospheric_state].encoded(), 0x22222222);

    // Parse as runtime packet to verify structure
    RuntimeContextPacket view(buffer.data(), TestContext::size_bytes);
    EXPECT_EQ(view.error(), ValidationError::none);
    EXPECT_EQ(view.cif3(), expected_cif3);
    EXPECT_EQ(view[network_id].encoded(), 0x11111111);
    EXPECT_EQ(view[tropospheric_state].encoded(), 0x22222222);
}
