#include "../context_test_fixture.hpp"

using namespace vrtigo::field;

TEST_F(ContextPacketTest, CIF1Fields) {
    // Create packet with CIF1 spectrum field
    // Note: Context packets always have Stream ID per VITA 49.2 spec
    using TestContext = ContextPacket<NoTimestamp, NoClassId, spectrum>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);

    // Check size includes CIF1 spectrum field
    EXPECT_EQ(TestContext::size_words(),
              1 + 1 + 1 + 1 + 13); // header + stream_id + cif0 + cif1 + spectrum
}

TEST_F(ContextPacketTest, NewCIF1Fields) {
    using TestContext = ContextPacket<NoTimestamp, NoClassId, health_status, phase_offset,
                                      polarization, pointing_vector_3d_single>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);

    // Set and verify Health Status (1 word)
    packet[health_status].set_encoded(0xABCDEF01);
    EXPECT_EQ(packet[health_status].encoded(), 0xABCDEF01);

    // Set and verify Phase Offset (1 word)
    packet[phase_offset].set_encoded(0x12345678);
    EXPECT_EQ(packet[phase_offset].encoded(), 0x12345678);

    // Set and verify Polarization (1 word)
    packet[polarization].set_encoded(0x87654321);
    EXPECT_EQ(packet[polarization].encoded(), 0x87654321);

    // Set and verify 3D Pointing Single (1 word)
    packet[pointing_vector_3d_single].set_encoded(0xFEDCBA98);
    EXPECT_EQ(packet[pointing_vector_3d_single].encoded(), 0xFEDCBA98);
}

TEST_F(ContextPacketTest, RuntimeParseCIF1) {
    // Build a packet with CIF1 enabled and Aux Frequency field
    // Type 4 has stream ID: header(1) + stream_id(1) + CIF0(1) + CIF1(1) + Aux Frequency(2) = 6
    // words
    alignas(4) std::array<uint8_t, 6 * 4> buffer{};
    uint32_t header = (static_cast<uint32_t>(PacketType::context) << header::packet_type_shift) | 6;
    cif::write_u32_safe(buffer.data(), 0, header);

    // Stream ID (type 4 has stream ID per VITA 49.2)
    cif::write_u32_safe(buffer.data(), 4, 0x12345678);

    // CIF0 with CIF1 enable bit set
    uint32_t cif0_mask = (1U << cif::CIF1_ENABLE_BIT);
    cif::write_u32_safe(buffer.data(), 8, cif0_mask);

    // CIF1 with Aux Frequency enabled
    uint32_t cif1_mask = vrtigo::detail::field_bitmask<aux_frequency>();
    cif::write_u32_safe(buffer.data(), 12, cif1_mask);

    // Aux Frequency: 10 MHz = 10,000,000 Hz
    uint64_t expected_freq = 10'000'000;
    cif::write_u64_safe(buffer.data(), 16, expected_freq);

    // Parse and validate
    auto result = RuntimeContextPacket::parse(buffer);
    ASSERT_TRUE(result.ok()) << result.error().message();
    const auto& view = result.value();

    // Verify CIF0 and CIF1 are read correctly
    EXPECT_EQ(view.cif0(), cif0_mask);
    EXPECT_EQ(view.cif1(), cif1_mask);

    // Verify we can read back the Aux Frequency field using the accessor
    auto aux_freq = view[aux_frequency];
    ASSERT_TRUE(aux_freq.has_value());
    EXPECT_EQ(aux_freq.encoded(), expected_freq);
}

TEST_F(ContextPacketTest, CompileTimeCIF1RuntimeParse) {
    // Create packet with CIF1 field at compile time
    using TestContext = ContextPacket<NoTimestamp, NoClassId, aux_frequency>;

    // Compile-time assertion: verify CIF1 enable bit is auto-set
    static_assert((TestContext::cif0_value & (1U << cif::CIF1_ENABLE_BIT)) != 0,
                  "CIF1 enable bit should be auto-set when CIF1 != 0");
    static_assert((TestContext::cif0_value & (1U << cif::CIF2_ENABLE_BIT)) == 0,
                  "CIF2 enable bit should NOT be set when CIF2 == 0");

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext tx_packet(buffer);
    tx_packet.set_stream_id(0xAABBCCDD);
    tx_packet[aux_frequency].set_encoded(15'000'000ULL); // 15 MHz

    // Parse with runtime view
    auto result = RuntimeContextPacket::parse(buffer);
    ASSERT_TRUE(result.ok()) << result.error().message();
    const auto& view = result.value();
    EXPECT_EQ(view.stream_id().value(), 0xAABBCCDD);
    // CIF0 should have CIF1 enable bit set
    EXPECT_EQ(view.cif0() & (1U << cif::CIF1_ENABLE_BIT), (1U << cif::CIF1_ENABLE_BIT));
    EXPECT_EQ(view.cif1(), vrtigo::detail::field_bitmask<aux_frequency>());
}
