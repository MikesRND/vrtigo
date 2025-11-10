#include "../context_test_fixture.hpp"

using namespace vrtio::field;

TEST_F(ContextPacketTest, CIF1Fields) {
    // Create packet with CIF1 spectrum field
    // Note: Context packets always have Stream ID per VITA 49.2 spec
    using TestContext = ContextPacket<NoTimeStamp, NoClassId, spectrum>;

    TestContext packet(buffer.data());

    // Check size includes CIF1 spectrum field
    EXPECT_EQ(TestContext::size_words,
              1 + 1 + 1 + 1 + 13); // header + stream_id + cif0 + cif1 + spectrum
}

TEST_F(ContextPacketTest, NewCIF1Fields) {
    using TestContext = ContextPacket<NoTimeStamp, NoClassId, health_status, phase_offset,
                                      polarization, pointing_vector_3d_single>;

    TestContext packet(buffer.data());

    // Set and verify Health Status (1 word)
    get(packet, health_status).set_raw_value(0xABCDEF01);
    EXPECT_EQ(get(packet, health_status).raw_value(), 0xABCDEF01);

    // Set and verify Phase Offset (1 word)
    get(packet, phase_offset).set_raw_value(0x12345678);
    EXPECT_EQ(get(packet, phase_offset).raw_value(), 0x12345678);

    // Set and verify Polarization (1 word)
    get(packet, polarization).set_raw_value(0x87654321);
    EXPECT_EQ(get(packet, polarization).raw_value(), 0x87654321);

    // Set and verify 3D Pointing Single (1 word)
    get(packet, pointing_vector_3d_single).set_raw_value(0xFEDCBA98);
    EXPECT_EQ(get(packet, pointing_vector_3d_single).raw_value(), 0xFEDCBA98);
}

TEST_F(ContextPacketTest, RuntimeParseCIF1) {
    // Build a packet with CIF1 enabled and Aux Frequency field
    // Type 4 has stream ID: header(1) + stream_id(1) + CIF0(1) + CIF1(1) + Aux Frequency(2) = 6
    // words
    uint32_t header = (static_cast<uint32_t>(PacketType::Context) << header::packet_type_shift) | 6;
    cif::write_u32_safe(buffer.data(), 0, header);

    // Stream ID (type 4 has stream ID per VITA 49.2)
    cif::write_u32_safe(buffer.data(), 4, 0x12345678);

    // CIF0 with CIF1 enable bit set
    uint32_t cif0_mask = (1U << cif::CIF1_ENABLE_BIT);
    cif::write_u32_safe(buffer.data(), 8, cif0_mask);

    // CIF1 with Aux Frequency enabled
    uint32_t cif1_mask = vrtio::detail::field_bitmask<aux_frequency>();
    cif::write_u32_safe(buffer.data(), 12, cif1_mask);

    // Aux Frequency: 10 MHz = 10,000,000 Hz
    uint64_t expected_freq = 10'000'000;
    cif::write_u64_safe(buffer.data(), 16, expected_freq);

    // Parse and validate
    ContextPacketView view(buffer.data(), 6 * 4);
    EXPECT_EQ(view.error(), ValidationError::none);

    // Verify CIF0 and CIF1 are read correctly
    EXPECT_EQ(view.cif0(), cif0_mask);
    EXPECT_EQ(view.cif1(), cif1_mask);

    // Verify we can read back the Aux Frequency field using the accessor
    auto aux_freq = get(view, aux_frequency);
    ASSERT_TRUE(aux_freq.has_value());
    EXPECT_EQ(aux_freq.raw_value(), expected_freq);
}

TEST_F(ContextPacketTest, CompileTimeCIF1RuntimeParse) {
    // Create packet with CIF1 field at compile time
    using TestContext = ContextPacket<NoTimeStamp, NoClassId, aux_frequency>;

    // Compile-time assertion: verify CIF1 enable bit is auto-set
    static_assert((TestContext::cif0_value & (1U << cif::CIF1_ENABLE_BIT)) != 0,
                  "CIF1 enable bit should be auto-set when CIF1 != 0");
    static_assert((TestContext::cif0_value & (1U << cif::CIF2_ENABLE_BIT)) == 0,
                  "CIF2 enable bit should NOT be set when CIF2 == 0");

    TestContext tx_packet(buffer.data());
    tx_packet.set_stream_id(0xAABBCCDD);
    get(tx_packet, aux_frequency).set_raw_value(15'000'000ULL); // 15 MHz

    // Parse with runtime view
    ContextPacketView view(buffer.data(), TestContext::size_bytes);
    EXPECT_EQ(view.error(), ValidationError::none);
    EXPECT_TRUE(view.has_stream_id());
    EXPECT_EQ(view.stream_id().value(), 0xAABBCCDD);
    // CIF0 should have CIF1 enable bit set
    EXPECT_EQ(view.cif0() & (1U << cif::CIF1_ENABLE_BIT), (1U << cif::CIF1_ENABLE_BIT));
    EXPECT_EQ(view.cif1(), vrtio::detail::field_bitmask<aux_frequency>());
}
