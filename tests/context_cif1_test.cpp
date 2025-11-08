#include "context_test_fixture.hpp"

TEST_F(ContextPacketTest, CIF1Fields) {
    // Create packet with CIF1 spectrum field
    constexpr uint32_t cif0_mask = 0;  // No CIF0 fields (CIF1 enable bit set automatically)
    constexpr uint32_t cif1_mask = cif1::SPECTRUM;

    using TestContext = ContextPacket<
        false,          // No stream ID
        NoTimeStamp,    // No timestamp
        NoClassId,      // No class ID
        cif0_mask,      // CIF0
        cif1_mask,      // CIF1
        0,              // No CIF2
        0,              // No CIF3
        false           // No trailer
    >;

    TestContext packet(buffer.data());

    // Check size includes CIF1 spectrum field
    EXPECT_EQ(TestContext::size_words, 1 + 1 + 1 + 13);  // header + cif0 + cif1 + spectrum
}

TEST_F(ContextPacketTest, NewCIF1Fields) {
    constexpr uint32_t cif1_mask = cif1::HEALTH_STATUS | cif1::PHASE_OFFSET |
                                   cif1::POLARIZATION | cif1::POINTING_VECTOR_3D_SINGLE;
    using TestContext = ContextPacket<
        true,           // Has stream ID
        NoTimeStamp,
        NoClassId,
        0,              // No CIF0 fields
        cif1_mask,      // CIF1 with new fields
        0,              // No CIF2
        0,              // No CIF3
        false
    >;

    TestContext packet(buffer.data());

    // Set and verify Health Status (1 word)
    set(packet, field::health_status, 0xABCDEF01);
    EXPECT_EQ(get(packet, field::health_status).value(), 0xABCDEF01);

    // Set and verify Phase Offset (1 word)
    set(packet, field::phase_offset, 0x12345678);
    EXPECT_EQ(get(packet, field::phase_offset).value(), 0x12345678);

    // Set and verify Polarization (1 word)
    set(packet, field::polarization, 0x87654321);
    EXPECT_EQ(get(packet, field::polarization).value(), 0x87654321);

    // Set and verify 3D Pointing Single (1 word)
    set(packet, field::pointing_vector_3d_single, 0xFEDCBA98);
    EXPECT_EQ(get(packet, field::pointing_vector_3d_single).value(), 0xFEDCBA98);
}

TEST_F(ContextPacketTest, RuntimeParseCIF1) {
    // Build a packet with CIF1 enabled and Aux Frequency field
    // Structure: header(1) + CIF0(1) + CIF1(1) + Aux Frequency(2) = 5 words
    uint32_t header = (static_cast<uint32_t>(packet_type::context) << header::PACKET_TYPE_SHIFT) | 5;
    cif::write_u32_safe(buffer.data(), 0, header);

    // CIF0 with CIF1 enable bit set
    uint32_t cif0_mask = (1U << cif::CIF1_ENABLE_BIT);
    cif::write_u32_safe(buffer.data(), 4, cif0_mask);

    // CIF1 with Aux Frequency enabled
    uint32_t cif1_mask = cif1::AUX_FREQUENCY;
    cif::write_u32_safe(buffer.data(), 8, cif1_mask);

    // Aux Frequency: 10 MHz = 10,000,000 Hz
    uint64_t expected_freq = 10'000'000;
    cif::write_u64_safe(buffer.data(), 12, expected_freq);

    // Parse and validate
    ContextPacketView view(buffer.data(), 5 * 4);
    EXPECT_EQ(view.validate(), validation_error::none);

    // Verify CIF0 and CIF1 are read correctly
    EXPECT_EQ(view.cif0(), cif0_mask);
    EXPECT_EQ(view.cif1(), cif1_mask);

    // Verify we can read back the Aux Frequency field using the accessor
    auto aux_freq = get(view, field::aux_frequency);
    ASSERT_TRUE(aux_freq.has_value());
    EXPECT_EQ(*aux_freq, expected_freq);
}

TEST_F(ContextPacketTest, CompileTimeCIF1RuntimeParse) {
    // Create packet with CIF1 field at compile time
    constexpr uint32_t cif1_mask = cif1::AUX_FREQUENCY;
    using TestContext = ContextPacket<
        true,           // Has stream ID
        NoTimeStamp,
        NoClassId,
        0,              // No CIF0 data fields (CIF1 enable bit auto-set)
        cif1_mask,      // CIF1
        0,              // No CIF2
        0,              // No CIF3
        false
    >;

    // Compile-time assertion: verify CIF1 enable bit is auto-set
    static_assert((TestContext::cif0_value & (1U << cif::CIF1_ENABLE_BIT)) != 0,
        "CIF1 enable bit should be auto-set when CIF1 != 0");
    static_assert((TestContext::cif0_value & (1U << cif::CIF2_ENABLE_BIT)) == 0,
        "CIF2 enable bit should NOT be set when CIF2 == 0");

    TestContext tx_packet(buffer.data());
    tx_packet.set_stream_id(0xAABBCCDD);
    set(tx_packet, field::aux_frequency, 15'000'000ULL);  // 15 MHz

    // Parse with runtime view
    ContextPacketView view(buffer.data(), TestContext::size_bytes);
    EXPECT_EQ(view.validate(), validation_error::none);
    EXPECT_TRUE(view.has_stream_id());
    EXPECT_EQ(view.stream_id().value(), 0xAABBCCDD);
    // CIF0 should have CIF1 enable bit set
    EXPECT_EQ(view.cif0() & (1U << cif::CIF1_ENABLE_BIT), (1U << cif::CIF1_ENABLE_BIT));
    EXPECT_EQ(view.cif1(), cif1_mask);
}

