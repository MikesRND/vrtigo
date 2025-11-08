#include "context_test_fixture.hpp"

TEST_F(ContextPacketTest, BasicCompileTimePacket) {
    // Create a simple context packet with bandwidth and sample rate
    constexpr uint32_t cif0_mask = cif0::BANDWIDTH | cif0::SAMPLE_RATE;
    using TestContext = ContextPacket<
        true,           // Has stream ID
        NoTimeStamp,    // No timestamp
        NoClassId,      // No class ID
        cif0_mask,      // CIF0
        0,              // No CIF1
        0,              // No CIF2
        false           // No trailer
    >;

    TestContext packet(buffer.data());

    // Check packet size
    EXPECT_EQ(TestContext::size_words, 1 + 1 + 1 + 2 + 2);  // header + stream + cif0 + bandwidth + sample_rate
    EXPECT_EQ(TestContext::size_bytes, TestContext::size_words * 4);

    // Set fields
    packet.set_stream_id(0x12345678);
    set(packet, field::bandwidth, 20'000'000ULL);  // 20 MHz
    set(packet, field::sample_rate, 10'000'000ULL);  // 10 MSPS

    // Verify fields
    EXPECT_EQ(packet.stream_id(), 0x12345678);
    EXPECT_EQ(get(packet, field::bandwidth).value(), 20'000'000);
    EXPECT_EQ(get(packet, field::sample_rate).value(), 10'000'000);
}

TEST_F(ContextPacketTest, PacketWithClassId) {
    // Define a class ID
    using TestClassId = ClassId<0x123456, 0xABCDEF00>;

    constexpr uint32_t cif0_mask = cif0::BANDWIDTH;
    using TestContext = ContextPacket<
        true,           // Has stream ID
        NoTimeStamp,    // No timestamp
        TestClassId,    // Has class ID
        cif0_mask,      // CIF0
        0,              // No CIF1
        0,              // No CIF2
        false           // No trailer
    >;

    TestContext packet(buffer.data());

    // Check that class ID increases packet size
    EXPECT_EQ(TestContext::size_words, 1 + 1 + 2 + 1 + 2);  // header + stream + class_id + cif0 + bandwidth

    packet.set_stream_id(0x87654321);
    set(packet, field::bandwidth, 40'000'000ULL);

    EXPECT_EQ(packet.stream_id(), 0x87654321);
    EXPECT_EQ(get(packet, field::bandwidth).value(), 40'000'000);
}

TEST_F(ContextPacketTest, RuntimeParserBasic) {
    // Manually construct a context packet in the buffer
    // Header: context packet (type 4), has stream ID, size
    uint32_t header =
        (static_cast<uint32_t>(packet_type::context) << header::PACKET_TYPE_SHIFT) |
        header::STREAM_ID_INDICATOR |
        7;
    cif::write_u32_safe(buffer.data(), 0, header);

    // Stream ID
    cif::write_u32_safe(buffer.data(), 4, 0xAABBCCDD);

    // CIF0 - enable bandwidth and sample rate
    uint32_t cif0_mask = cif0::BANDWIDTH | cif0::SAMPLE_RATE;
    cif::write_u32_safe(buffer.data(), 8, cif0_mask);

    // Bandwidth (64-bit)
    cif::write_u64_safe(buffer.data(), 12, 25'000'000);

    // Sample rate (64-bit)
    cif::write_u64_safe(buffer.data(), 20, 12'500'000);

    // Parse with ContextPacketView
    ContextPacketView view(buffer.data(), 7 * 4);
    EXPECT_EQ(view.validate(), validation_error::none);

    // Check parsed values
    EXPECT_TRUE(view.has_stream_id());
    EXPECT_EQ(view.stream_id().value(), 0xAABBCCDD);

    EXPECT_EQ(view.cif0(), cif0_mask);
    EXPECT_EQ(view.cif1(), 0);
    EXPECT_EQ(view.cif2(), 0);

    auto bw = get(view, field::bandwidth);
    EXPECT_TRUE(bw.has_value());
    EXPECT_EQ(bw.value(), 25'000'000);

    auto sr = get(view, field::sample_rate);
    EXPECT_TRUE(sr.has_value());
    EXPECT_EQ(sr.value(), 12'500'000);

    // Field that's not present should return nullopt
    EXPECT_FALSE(get(view, field::gain).has_value());
}

TEST_F(ContextPacketTest, SizeFieldValidation) {
    // Create packet with wrong size field in header
    // Actual structure: header (1) + CIF0 (1) + bandwidth (2) = 4 words
    // But header claims 10 words (mismatch!)
    uint32_t header = (static_cast<uint32_t>(packet_type::context) << header::PACKET_TYPE_SHIFT) | 10;  // type=4, WRONG size=10 words
    cif::write_u32_safe(buffer.data(), 0, header);

    uint32_t cif0_mask = cif0::BANDWIDTH;
    cif::write_u32_safe(buffer.data(), 4, cif0_mask);
    cif::write_u64_safe(buffer.data(), 8, 25'000'000);

    // Provide buffer large enough for header's claim, so we get past buffer_too_small check
    ContextPacketView view(buffer.data(), 10 * 4);
    EXPECT_EQ(view.validate(), validation_error::size_field_mismatch);
}

TEST_F(ContextPacketTest, BufferTooSmall) {
    uint32_t header = (static_cast<uint32_t>(packet_type::context) << header::PACKET_TYPE_SHIFT) | 10;  // type=4, size=10 words
    cif::write_u32_safe(buffer.data(), 0, header);

    // Provide buffer smaller than declared size
    ContextPacketView view(buffer.data(), 3 * 4);  // Only 3 words provided
    EXPECT_EQ(view.validate(), validation_error::buffer_too_small);
}

TEST_F(ContextPacketTest, InvalidPacketType) {
    uint32_t header = (0U << 28) | 3;  // type=0 (not context), size=3
    cif::write_u32_safe(buffer.data(), 0, header);

    ContextPacketView view(buffer.data(), 3 * 4);
    EXPECT_EQ(view.validate(), validation_error::invalid_packet_type);
}

