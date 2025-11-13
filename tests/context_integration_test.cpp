#include "context_test_fixture.hpp"

using namespace vrtio::field;

TEST_F(ContextPacketTest, RoundTrip) {
    // Create packet with template
    // Note: Context packets always have Stream ID per VITA 49.2 spec
    using TestContext = ContextPacket<NoTimeStamp, NoClassId, bandwidth, gain>;

    TestContext tx_packet(buffer.data());
    tx_packet.set_stream_id(0xDEADBEEF);
    tx_packet[bandwidth].set_value(100'000'000.0); // 100 MHz
    tx_packet[gain].set_raw_value(0x12345678U);

    // Parse same buffer with view
    ContextPacketView view(buffer.data(), TestContext::size_bytes);
    EXPECT_EQ(view.error(), ValidationError::none);

    EXPECT_EQ(view.stream_id().value(), 0xDEADBEEF);
    EXPECT_DOUBLE_EQ(view[bandwidth].value(), 100'000'000.0);
    EXPECT_EQ(view[gain].raw_value(), 0x12345678);
}

TEST_F(ContextPacketTest, CombinedCIF1AndCIF2CompileTime) {
    // Create packet with both CIF1 and CIF2 fields
    using TestContext =
        ContextPacket<NoTimeStamp, NoClassId, bandwidth, aux_frequency, controller_uuid>;

    // Compile-time assertions: verify both enable bits are auto-set
    static_assert((TestContext::cif0_value & (1U << cif::CIF1_ENABLE_BIT)) != 0,
                  "CIF1 enable bit should be set when CIF1 != 0");
    static_assert((TestContext::cif0_value & (1U << cif::CIF2_ENABLE_BIT)) != 0,
                  "CIF2 enable bit should be set when CIF2 != 0");
    static_assert((TestContext::cif0_value & (1U << 29)) != 0,
                  "Bandwidth bit should be preserved from CIF0 parameter");

    TestContext tx_packet(buffer.data());
    tx_packet.set_stream_id(0x11223344);
    tx_packet[bandwidth].set_value(50'000'000.0);          // 50 MHz (has interpreted support)
    tx_packet[aux_frequency].set_raw_value(25'000'000ULL); // Raw (no interpreted support)

    // Parse with runtime view
    ContextPacketView view(buffer.data(), TestContext::size_bytes);
    EXPECT_EQ(view.error(), ValidationError::none);

    // Verify CIF0 has both enable bits set
    constexpr uint32_t cif_enable_mask =
        (1U << cif::CIF1_ENABLE_BIT) | (1U << cif::CIF2_ENABLE_BIT);
    EXPECT_EQ(view.cif0() & cif_enable_mask, cif_enable_mask);
    EXPECT_EQ(view.cif1(), vrtio::detail::field_bitmask<aux_frequency>());
    EXPECT_EQ(view.cif2(), vrtio::detail::field_bitmask<controller_uuid>());

    // Verify fields
    EXPECT_EQ(view.stream_id().value(), 0x11223344);
    EXPECT_DOUBLE_EQ(view[bandwidth].value(), 50'000'000.0);
    EXPECT_EQ(view[aux_frequency].raw_value(), 25'000'000);
}

TEST_F(ContextPacketTest, CombinedCIF1AndCIF2Runtime) {
    // Manually build a packet with both CIF1 and CIF2
    // Structure: header(1) + stream_id(1) + CIF0(1) + CIF1(1) + CIF2(1) + bandwidth(2) +
    // aux_freq(2) + uuid(4) = 13 words Use ext_context (type 5) for packets with stream ID
    uint32_t header =
        (static_cast<uint32_t>(PacketType::ExtensionContext) << header::packet_type_shift) | 13;
    cif::write_u32_safe(buffer.data(), 0, header);

    // Stream ID
    cif::write_u32_safe(buffer.data(), 4, 0xAABBCCDD);

    // CIF0: Enable CIF1, CIF2, and Bandwidth
    uint32_t cif0_mask = (1U << 1) | (1U << 2) | vrtio::detail::field_bitmask<bandwidth>();
    cif::write_u32_safe(buffer.data(), 8, cif0_mask);

    // CIF1: Aux Frequency
    uint32_t cif1_mask = vrtio::detail::field_bitmask<aux_frequency>();
    cif::write_u32_safe(buffer.data(), 12, cif1_mask);

    // CIF2: Controller UUID
    uint32_t cif2_mask = vrtio::detail::field_bitmask<controller_uuid>();
    cif::write_u32_safe(buffer.data(), 16, cif2_mask);

    // Bandwidth (2 words)
    cif::write_u64_safe(buffer.data(), 20, 100'000'000);

    // Aux Frequency (2 words)
    cif::write_u64_safe(buffer.data(), 28, 75'000'000);

    // Controller UUID (4 words)
    cif::write_u32_safe(buffer.data(), 36, 0x12345678);
    cif::write_u32_safe(buffer.data(), 40, 0x9ABCDEF0);
    cif::write_u32_safe(buffer.data(), 44, 0x11111111);
    cif::write_u32_safe(buffer.data(), 48, 0x22222222);

    // Parse and validate
    ContextPacketView view(buffer.data(), 13 * 4);
    EXPECT_EQ(view.error(), ValidationError::none);

    // Verify structure
    EXPECT_EQ(view.cif0(), cif0_mask);
    EXPECT_EQ(view.cif1(), cif1_mask);
    EXPECT_EQ(view.cif2(), cif2_mask);

    // Verify fields
    EXPECT_EQ(view.stream_id().value(), 0xAABBCCDD);
    EXPECT_EQ(view[bandwidth].raw_value(), 100'000'000);
    EXPECT_EQ(view[aux_frequency].raw_value(), 75'000'000);

    auto uuid_proxy = view[controller_uuid];
    ASSERT_TRUE(uuid_proxy.has_value());
    auto uuid = uuid_proxy.raw_bytes();
    EXPECT_EQ(cif::read_u32_safe(uuid.data(), 0), 0x12345678);
}

TEST_F(ContextPacketTest, MultiWordFieldWrite) {
    // Create a compile-time packet with Data Payload Format (2 words, FieldView<2>)
    using TestContext = ContextPacket<NoTimeStamp, NoClassId, data_payload_format>;

    TestContext packet(buffer.data());

    // Create source data for the field (2 words = 8 bytes)
    alignas(4) uint8_t source_data[8];
    cif::write_u32_safe(source_data, 0, 0xAABBCCDD);
    cif::write_u32_safe(source_data, 4, 0x11223344);

    // Create a FieldView from the source data
    FieldView<2> field_value(source_data, 0);

    // Write the field to the packet
    packet[data_payload_format].set_raw_value(field_value);

    // Read it back and verify
    auto read_value = packet[data_payload_format];
    ASSERT_TRUE(read_value.has_value());
    EXPECT_EQ(read_value.raw_value().word(0), 0xAABBCCDD);
    EXPECT_EQ(read_value.raw_value().word(1), 0x11223344);

    // Verify round-trip through runtime parser
    ContextPacketView view(buffer.data(), TestContext::size_bytes);
    EXPECT_EQ(view.error(), ValidationError::none);

    auto runtime_value = view[data_payload_format];
    ASSERT_TRUE(runtime_value.has_value());
    EXPECT_EQ(runtime_value.raw_value().word(0), 0xAABBCCDD);
    EXPECT_EQ(runtime_value.raw_value().word(1), 0x11223344);
}
