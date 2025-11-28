#include "context_test_fixture.hpp"

using namespace vrtigo::field;

TEST_F(ContextPacketTest, BasicCompileTimePacket) {
    // Create a simple context packet with bandwidth and sample rate
    // Note: Context packets always have Stream ID per VITA 49.2 spec
    using TestContext = ContextPacket<NoTimestamp, NoClassId, bandwidth, sample_rate>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> pkt_buffer{};
    TestContext packet(pkt_buffer);

    // Check packet size
    EXPECT_EQ(TestContext::size_words(),
              1 + 1 + 1 + 2 + 2); // header + stream + cif0 + bandwidth + sample_rate
    EXPECT_EQ(TestContext::size_bytes(), TestContext::size_words() * 4);

    // Set fields
    packet.set_stream_id(0x12345678);
    packet[bandwidth].set_value(20'000'000.0);   // 20 MHz
    packet[sample_rate].set_value(10'000'000.0); // 10 MSPS

    // Verify fields
    EXPECT_EQ(packet.stream_id(), 0x12345678);
    EXPECT_DOUBLE_EQ(packet[bandwidth].value(), 20'000'000.0);
    EXPECT_DOUBLE_EQ(packet[sample_rate].value(), 10'000'000.0);
}

TEST_F(ContextPacketTest, PacketWithClassId) {
    using TestContext = ContextPacket<NoTimestamp, ClassId, bandwidth>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> pkt_buffer{};
    TestContext packet(pkt_buffer);

    // Check that class ID increases packet size
    EXPECT_EQ(TestContext::size_words(),
              1 + 1 + 2 + 1 + 2); // header + stream + class_id + cif0 + bandwidth

    packet.set_stream_id(0x87654321);

    // Set class ID: OUI=0x123456, ICC=0x5678, PCC=0xABCD
    ClassIdValue cid(0x123456, 0x5678, 0xABCD);
    packet.set_class_id(cid);

    packet[bandwidth].set_value(40'000'000.0); // 40 MHz

    EXPECT_EQ(packet.stream_id(), 0x87654321);

    // Verify class ID
    auto read_cid = packet.class_id();
    EXPECT_EQ(read_cid.oui(), 0x123456U);
    EXPECT_EQ(read_cid.icc(), 0x5678U);
    EXPECT_EQ(read_cid.pcc(), 0xABCDU);

    EXPECT_DOUBLE_EQ(packet[bandwidth].value(), 40'000'000.0);
}

TEST_F(ContextPacketTest, RuntimeParserBasic) {
    // Manually construct a context packet in the buffer
    // Header: extension context packet (type 5), which has stream ID per spec
    // Stream ID presence is determined by packet type (odd=has, even=no), not bit 25
    uint32_t header =
        (static_cast<uint32_t>(PacketType::extension_context) << header::packet_type_shift) | 7;
    cif::write_u32_safe(buffer.data(), 0, header);

    // Stream ID
    cif::write_u32_safe(buffer.data(), 4, 0xAABBCCDD);

    // CIF0 - enable bandwidth and sample rate
    uint32_t cif0_mask =
        vrtigo::detail::field_bitmask<bandwidth>() | vrtigo::detail::field_bitmask<sample_rate>();
    cif::write_u32_safe(buffer.data(), 8, cif0_mask);

    // Bandwidth (64-bit)
    cif::write_u64_safe(buffer.data(), 12, 25'000'000);

    // Sample rate (64-bit)
    cif::write_u64_safe(buffer.data(), 20, 12'500'000);

    // Parse with RuntimeContextPacket
    auto result = RuntimeContextPacket::parse(std::span<const uint8_t>(buffer.data(), 7 * 4));
    ASSERT_TRUE(result.ok()) << result.error().message();
    const auto& view = result.value();

    // Check parsed values
    EXPECT_EQ(view.stream_id().value(), 0xAABBCCDD);

    EXPECT_EQ(view.cif0(), cif0_mask);
    EXPECT_EQ(view.cif1(), 0);
    EXPECT_EQ(view.cif2(), 0);

    auto bw = view[bandwidth];
    EXPECT_TRUE(bw.has_value());
    EXPECT_EQ(bw.encoded(), 25'000'000);

    auto sr = view[sample_rate];
    EXPECT_TRUE(sr.has_value());
    EXPECT_EQ(sr.encoded(), 12'500'000);

    // Field that's not present should return nullopt
    EXPECT_FALSE(view[gain].has_value());
}

TEST_F(ContextPacketTest, SizeFieldValidation) {
    // Create packet with wrong size field in header
    // Actual structure: header (1) + stream_id (1) + CIF0 (1) + bandwidth (2) = 5 words
    // But header claims 10 words (mismatch!)
    uint32_t header = (static_cast<uint32_t>(PacketType::context) << header::packet_type_shift) |
                      10; // type=4, WRONG size=10 words
    cif::write_u32_safe(buffer.data(), 0, header);

    // Stream ID (always present per spec)
    cif::write_u32_safe(buffer.data(), 4, 0);

    uint32_t cif0_mask = vrtigo::detail::field_bitmask<bandwidth>();
    cif::write_u32_safe(buffer.data(), 8, cif0_mask);
    cif::write_u64_safe(buffer.data(), 12, 25'000'000);

    // Provide buffer large enough for header's claim, so we get past buffer_too_small check
    auto result = RuntimeContextPacket::parse(std::span<const uint8_t>(buffer.data(), 10 * 4));
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, ValidationError::size_field_mismatch);
}

TEST_F(ContextPacketTest, BufferTooSmall) {
    uint32_t header = (static_cast<uint32_t>(PacketType::context) << header::packet_type_shift) |
                      10; // type=4, size=10 words
    cif::write_u32_safe(buffer.data(), 0, header);

    // Provide buffer smaller than declared size
    auto result = RuntimeContextPacket::parse(
        std::span<const uint8_t>(buffer.data(), 3 * 4)); // Only 3 words provided
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, ValidationError::buffer_too_small);
}

TEST_F(ContextPacketTest, InvalidPacketType) {
    uint32_t header = (0U << 28) | 3; // type=0 (not context), size=3
    cif::write_u32_safe(buffer.data(), 0, header);

    auto result = RuntimeContextPacket::parse(std::span<const uint8_t>(buffer.data(), 3 * 4));
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, ValidationError::packet_type_mismatch);
}

TEST_F(ContextPacketTest, PacketCountAccessors) {
    // Test packet_count getter/setter for compile-time context packets
    using TestContext = ContextPacket<NoTimestamp, NoClassId, bandwidth>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> pkt_buffer{};
    TestContext packet(pkt_buffer);

    // Test initial packet count (should be 0 after initialization)
    EXPECT_EQ(packet.packet_count(), 0);

    // Test setting valid values (0-15)
    packet.set_packet_count(5);
    EXPECT_EQ(packet.packet_count(), 5);

    packet.set_packet_count(0);
    EXPECT_EQ(packet.packet_count(), 0);

    packet.set_packet_count(15);
    EXPECT_EQ(packet.packet_count(), 15);

    // Test modulo-16 wrapping for values > 15
    packet.set_packet_count(16);
    EXPECT_EQ(packet.packet_count(), 0); // 16 % 16 = 0

    packet.set_packet_count(17);
    EXPECT_EQ(packet.packet_count(), 1); // 17 % 16 = 1

    packet.set_packet_count(31);
    EXPECT_EQ(packet.packet_count(), 15); // 31 % 16 = 15

    packet.set_packet_count(255);
    EXPECT_EQ(packet.packet_count(), 15); // 255 % 16 = 15
}

TEST_F(ContextPacketTest, PacketCountParsing) {
    // Test packet_count parsing in RuntimeContextPacket
    // Manually construct a context packet with specific packet count

    // Create header with packet count = 7
    uint32_t header = (static_cast<uint32_t>(PacketType::context) << header::packet_type_shift) |
                      (7U << 16) | // packet_count = 7
                      5;           // size = 5 words
    cif::write_u32_safe(buffer.data(), 0, header);

    // Stream ID (always present for context packets per spec)
    cif::write_u32_safe(buffer.data(), 4, 0x12345678);

    // CIF0 - enable bandwidth
    uint32_t cif0_mask = vrtigo::detail::field_bitmask<bandwidth>();
    cif::write_u32_safe(buffer.data(), 8, cif0_mask);

    // Bandwidth field (64-bit)
    cif::write_u64_safe(buffer.data(), 12, 20'000'000);

    // Parse with RuntimeContextPacket
    auto result = RuntimeContextPacket::parse(std::span<const uint8_t>(buffer.data(), 5 * 4));
    ASSERT_TRUE(result.ok()) << result.error().message();
    const auto& view = result.value();

    // Verify packet_count was correctly parsed
    EXPECT_EQ(view.packet_count(), 7);

    // Test with different packet count values
    for (uint8_t count = 0; count <= 15; ++count) {
        // Update header with new packet count
        header = (static_cast<uint32_t>(PacketType::context) << header::packet_type_shift) |
                 (static_cast<uint32_t>(count) << 16) | 5;
        cif::write_u32_safe(buffer.data(), 0, header);

        // Re-parse
        auto result2 = RuntimeContextPacket::parse(std::span<const uint8_t>(buffer.data(), 5 * 4));
        ASSERT_TRUE(result2.ok()) << result2.error().message();
        EXPECT_EQ(result2.value().packet_count(), count);
    }
}
