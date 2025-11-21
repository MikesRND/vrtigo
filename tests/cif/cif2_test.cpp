#include "../context_test_fixture.hpp"

TEST_F(ContextPacketTest, CIF2Fields) {
    // Create packet with CIF2 controller UUID using field-based API
    // Note: Context packets always have Stream ID per VITA 49.2 spec
    using namespace vrtigo::field;
    using TestContext = ContextPacket<NoTimeStamp,      // No timestamp
                                      NoClassId,        // No class ID
                                      controller_uuid>; // CIF2 field

    TestContext packet(buffer.data());

    // Check size includes CIF2 field
    EXPECT_EQ(TestContext::size_words,
              1 + 1 + 1 + 1 + 4); // header + stream_id + cif0 + cif2 + uuid
}

TEST_F(ContextPacketTest, RuntimeParseCIF2) {
    // Build a packet with CIF2 enabled and Controller UUID field
    // Type 4 has stream ID: header(1) + stream_id(1) + CIF0(1) + CIF2(1) + Controller UUID(4) = 8
    // words
    uint32_t header = (static_cast<uint32_t>(PacketType::context) << header::packet_type_shift) | 8;
    cif::write_u32_safe(buffer.data(), 0, header);

    // Stream ID (type 4 has stream ID per VITA 49.2)
    cif::write_u32_safe(buffer.data(), 4, 0x12345678);

    // CIF0 with CIF2 enable bit set
    uint32_t cif0_mask = (1U << 2); // Enable CIF2
    cif::write_u32_safe(buffer.data(), 8, cif0_mask);

    // CIF2 with Controller UUID enabled
    using namespace vrtigo::field;
    uint32_t cif2_mask = vrtigo::detail::field_bitmask<controller_uuid>();
    cif::write_u32_safe(buffer.data(), 12, cif2_mask);

    // Controller UUID: 4 words (128 bits)
    uint32_t expected_uuid[4] = {0x12345678, 0x9ABCDEF0, 0x11111111, 0x22222222};
    cif::write_u32_safe(buffer.data(), 16, expected_uuid[0]);
    cif::write_u32_safe(buffer.data(), 20, expected_uuid[1]);
    cif::write_u32_safe(buffer.data(), 24, expected_uuid[2]);
    cif::write_u32_safe(buffer.data(), 28, expected_uuid[3]);

    // Parse and validate
    RuntimeContextPacket view(buffer.data(), 8 * 4);
    EXPECT_EQ(view.error(), ValidationError::none);

    // Verify CIF0 and CIF2 are read correctly
    EXPECT_EQ(view.cif0(), cif0_mask);
    EXPECT_EQ(view.cif2(), cif2_mask);

    // Verify we can read back the Controller UUID field using operator[] API
    auto uuid_proxy = view[field::controller_uuid];
    ASSERT_TRUE(uuid_proxy.has_value());
    auto uuid = uuid_proxy.bytes();
    ASSERT_EQ(uuid.size(), 16); // 128 bits = 16 bytes

    // Verify UUID bytes match what we wrote
    for (int i = 0; i < 4; i++) {
        uint32_t word = cif::read_u32_safe(uuid.data(), i * 4);
        EXPECT_EQ(word, expected_uuid[i]);
    }
}

TEST_F(ContextPacketTest, CompileTimeCIF2RuntimeParse) {
    // Create packet with CIF2 field at compile time using field-based API
    using namespace vrtigo::field;
    using TestContext = ContextPacket<NoTimeStamp, NoClassId,
                                      controller_uuid>; // CIF2 field

    // Compile-time assertion: verify CIF2 enable bit is auto-set
    static_assert((TestContext::cif0_value & 0x04) != 0,
                  "CIF2 enable bit (bit 2) should be auto-set when CIF2 != 0");
    static_assert((TestContext::cif0_value & 0x02) == 0,
                  "CIF1 enable bit (bit 1) should NOT be set when CIF1 == 0");

    TestContext tx_packet(buffer.data());
    tx_packet.set_stream_id(0xAABBCCDD);

    // Parse with runtime view
    RuntimeContextPacket view(buffer.data(), TestContext::size_bytes);
    EXPECT_EQ(view.error(), ValidationError::none);
    EXPECT_EQ(view.stream_id().value(), 0xAABBCCDD);
    // CIF0 should have bit 2 set (CIF2 enable)
    EXPECT_EQ(view.cif0() & 0x04, 0x04);
    EXPECT_EQ(view.cif2(), vrtigo::detail::field_bitmask<controller_uuid>());
}
