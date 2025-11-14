#include "context_test_fixture.hpp"

using namespace vrtio::field;

TEST_F(ContextPacketTest, BasicCompileTimePacket) {
    // Create a simple context packet with bandwidth and sample rate
    // Note: Context packets always have Stream ID per VITA 49.2 spec
    using TestContext = ContextPacket<NoTimeStamp, NoClassId, bandwidth, sample_rate>;

    TestContext packet(buffer.data());

    // Check packet size
    EXPECT_EQ(TestContext::size_words,
              1 + 1 + 1 + 2 + 2); // header + stream + cif0 + bandwidth + sample_rate
    EXPECT_EQ(TestContext::size_bytes, TestContext::size_words * 4);

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
    using TestContext = ContextPacket<NoTimeStamp, ClassId, bandwidth>;

    TestContext packet(buffer.data());

    // Check that class ID increases packet size
    EXPECT_EQ(TestContext::size_words,
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
        (static_cast<uint32_t>(PacketType::ExtensionContext) << header::packet_type_shift) | 7;
    cif::write_u32_safe(buffer.data(), 0, header);

    // Stream ID
    cif::write_u32_safe(buffer.data(), 4, 0xAABBCCDD);

    // CIF0 - enable bandwidth and sample rate
    uint32_t cif0_mask =
        vrtio::detail::field_bitmask<bandwidth>() | vrtio::detail::field_bitmask<sample_rate>();
    cif::write_u32_safe(buffer.data(), 8, cif0_mask);

    // Bandwidth (64-bit)
    cif::write_u64_safe(buffer.data(), 12, 25'000'000);

    // Sample rate (64-bit)
    cif::write_u64_safe(buffer.data(), 20, 12'500'000);

    // Parse with ContextPacketView
    ContextPacketView view(buffer.data(), 7 * 4);
    EXPECT_EQ(view.error(), ValidationError::none);

    // Check parsed values
    EXPECT_EQ(view.stream_id().value(), 0xAABBCCDD);

    EXPECT_EQ(view.cif0(), cif0_mask);
    EXPECT_EQ(view.cif1(), 0);
    EXPECT_EQ(view.cif2(), 0);

    auto bw = view[bandwidth];
    EXPECT_TRUE(bw.has_value());
    EXPECT_EQ(bw.raw_value(), 25'000'000);

    auto sr = view[sample_rate];
    EXPECT_TRUE(sr.has_value());
    EXPECT_EQ(sr.raw_value(), 12'500'000);

    // Field that's not present should return nullopt
    EXPECT_FALSE(view[gain].has_value());
}

TEST_F(ContextPacketTest, SizeFieldValidation) {
    // Create packet with wrong size field in header
    // Actual structure: header (1) + stream_id (1) + CIF0 (1) + bandwidth (2) = 5 words
    // But header claims 10 words (mismatch!)
    uint32_t header = (static_cast<uint32_t>(PacketType::Context) << header::packet_type_shift) |
                      10; // type=4, WRONG size=10 words
    cif::write_u32_safe(buffer.data(), 0, header);

    // Stream ID (always present per spec)
    cif::write_u32_safe(buffer.data(), 4, 0);

    uint32_t cif0_mask = vrtio::detail::field_bitmask<bandwidth>();
    cif::write_u32_safe(buffer.data(), 8, cif0_mask);
    cif::write_u64_safe(buffer.data(), 12, 25'000'000);

    // Provide buffer large enough for header's claim, so we get past buffer_too_small check
    ContextPacketView view(buffer.data(), 10 * 4);
    EXPECT_EQ(view.error(), ValidationError::size_field_mismatch);
}

TEST_F(ContextPacketTest, BufferTooSmall) {
    uint32_t header = (static_cast<uint32_t>(PacketType::Context) << header::packet_type_shift) |
                      10; // type=4, size=10 words
    cif::write_u32_safe(buffer.data(), 0, header);

    // Provide buffer smaller than declared size
    ContextPacketView view(buffer.data(), 3 * 4); // Only 3 words provided
    EXPECT_EQ(view.error(), ValidationError::buffer_too_small);
}

TEST_F(ContextPacketTest, InvalidPacketType) {
    uint32_t header = (0U << 28) | 3; // type=0 (not context), size=3
    cif::write_u32_safe(buffer.data(), 0, header);

    ContextPacketView view(buffer.data(), 3 * 4);
    EXPECT_EQ(view.error(), ValidationError::invalid_packet_type);
}
