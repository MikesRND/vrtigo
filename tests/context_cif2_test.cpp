#include "context_test_fixture.hpp"

TEST_F(ContextPacketTest, CIF2Fields) {
    // Create packet with CIF2 controller UUID
    constexpr uint32_t cif0_mask = 0;  // No CIF0 fields (CIF2 enable bit set automatically)
    constexpr uint32_t cif2_mask = cif2::CONTROLLER_UUID;

    using TestContext = ContextPacket<
        false,          // No stream ID
        NoTimeStamp,    // No timestamp
        NoClassId,      // No class ID
        cif0_mask,      // CIF0
        0,              // No CIF1
        cif2_mask,      // CIF2
        false           // No trailer
    >;

    TestContext packet(buffer.data());

    // Check size includes CIF2 field
    EXPECT_EQ(TestContext::size_words, 1 + 1 + 1 + 4);  // header + cif0 + cif2 + uuid
}

TEST_F(ContextPacketTest, RuntimeParseCIF2) {
    // Build a packet with CIF2 enabled and Controller UUID field
    // Structure: header(1) + CIF0(1) + CIF2(1) + Controller UUID(4) = 7 words
    uint32_t header = (static_cast<uint32_t>(packet_type::context) << header::PACKET_TYPE_SHIFT) | 7;
    cif::write_u32_safe(buffer.data(), 0, header);

    // CIF0 with CIF2 enable bit set
    uint32_t cif0_mask = (1U << 2);  // Enable CIF2
    cif::write_u32_safe(buffer.data(), 4, cif0_mask);

    // CIF2 with Controller UUID enabled
    uint32_t cif2_mask = cif2::CONTROLLER_UUID;
    cif::write_u32_safe(buffer.data(), 8, cif2_mask);

    // Controller UUID: 4 words (128 bits)
    uint32_t expected_uuid[4] = {0x12345678, 0x9ABCDEF0, 0x11111111, 0x22222222};
    cif::write_u32_safe(buffer.data(), 12, expected_uuid[0]);
    cif::write_u32_safe(buffer.data(), 16, expected_uuid[1]);
    cif::write_u32_safe(buffer.data(), 20, expected_uuid[2]);
    cif::write_u32_safe(buffer.data(), 24, expected_uuid[3]);

    // Parse and validate
    ContextPacketView view(buffer.data(), 7 * 4);
    EXPECT_EQ(view.validate(), validation_error::none);

    // Verify CIF0 and CIF2 are read correctly
    EXPECT_EQ(view.cif0(), cif0_mask);
    EXPECT_EQ(view.cif2(), cif2_mask);

    // Verify we can read back the Controller UUID field using the accessor
    auto uuid = view.controller_uuid();
    ASSERT_TRUE(uuid.has_value());
    ASSERT_EQ(uuid->size(), 16);  // 128 bits = 16 bytes

    // Verify UUID bytes match what we wrote
    for (int i = 0; i < 4; i++) {
        uint32_t word = cif::read_u32_safe(uuid->data(), i * 4);
        EXPECT_EQ(word, expected_uuid[i]);
    }
}

TEST_F(ContextPacketTest, CompileTimeCIF2RuntimeParse) {
    // Create packet with CIF2 field at compile time
    constexpr uint32_t cif2_mask = cif2::CONTROLLER_UUID;
    using TestContext = ContextPacket<
        false,          // No stream ID
        NoTimeStamp,
        NoClassId,
        0,              // No CIF0 data fields (CIF2 enable bit auto-set)
        0,              // No CIF1
        cif2_mask,      // CIF2
        false
    >;

    // Compile-time assertion: verify CIF2 enable bit is auto-set
    static_assert((TestContext::cif0_value & 0x04) != 0,
        "CIF2 enable bit (bit 2) should be auto-set when CIF2 != 0");
    static_assert((TestContext::cif0_value & 0x02) == 0,
        "CIF1 enable bit (bit 1) should NOT be set when CIF1 == 0");

    TestContext tx_packet(buffer.data());

    // Parse with runtime view
    ContextPacketView view(buffer.data(), TestContext::size_bytes);
    EXPECT_EQ(view.validate(), validation_error::none);
    // CIF0 should have bit 2 set (CIF2 enable)
    EXPECT_EQ(view.cif0() & 0x04, 0x04);
    EXPECT_EQ(view.cif2(), cif2_mask);
}

