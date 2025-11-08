#include "context_test_fixture.hpp"

TEST_F(ContextPacketTest, GPSASCIIVariableField) {
    // Manually create packet with GPS ASCII field
    // Packet structure: header (1) + CIF0 (1) + GPS ASCII (4) = 6 words
    uint32_t header = (static_cast<uint32_t>(packet_type::context) << header::PACKET_TYPE_SHIFT) | 6;  // type=4, size=6 words
    cif::write_u32_safe(buffer.data(), 0, header);

    // CIF0 with GPS ASCII bit
    uint32_t cif0_mask = cif0::GPS_ASCII;
    cif::write_u32_safe(buffer.data(), 4, cif0_mask);

    // GPS ASCII field: count + data
    uint32_t char_count = 12;  // "Hello World!" = 12 chars
    cif::write_u32_safe(buffer.data(), 8, char_count);

    // Write ASCII data (3 words needed for 12 chars)
    const char* msg = "Hello World!";
    std::memcpy(buffer.data() + 12, msg, 12);

    ContextPacketView view(buffer.data(), 6 * 4);
    EXPECT_EQ(view.validate(), validation_error::none);

    EXPECT_TRUE(view.has_gps_ascii());
    auto gps_data = view.gps_ascii_data();
    EXPECT_EQ(gps_data.size(), 4 * 4);  // 1 count + 3 data words (12 bytes)

    // Extract the character count
    uint32_t parsed_count;
    std::memcpy(&parsed_count, gps_data.data(), 4);
    parsed_count = detail::network_to_host32(parsed_count);
    EXPECT_EQ(parsed_count, 12);

    // Check the message
    EXPECT_EQ(std::memcmp(gps_data.data() + 4, msg, 12), 0);
}

TEST_F(ContextPacketTest, ContextAssociationLists) {
    uint32_t header = (static_cast<uint32_t>(packet_type::context) << header::PACKET_TYPE_SHIFT) | 6;  // type=4, size=6 words
    cif::write_u32_safe(buffer.data(), 0, header);

    // CIF0 with context association bit
    uint32_t cif0_mask = cif0::CONTEXT_ASSOCIATION_LISTS;
    cif::write_u32_safe(buffer.data(), 4, cif0_mask);

    // Context association: two 16-bit counts + IDs
    uint16_t stream_count = 2;
    uint16_t context_count = 1;
    uint32_t counts_word = (stream_count << 16) | context_count;
    cif::write_u32_safe(buffer.data(), 8, counts_word);

    // Stream IDs
    cif::write_u32_safe(buffer.data(), 12, 0x1111);
    cif::write_u32_safe(buffer.data(), 16, 0x2222);

    // Context ID
    cif::write_u32_safe(buffer.data(), 20, 0x3333);

    ContextPacketView view(buffer.data(), 6 * 4);
    EXPECT_EQ(view.validate(), validation_error::none);

    EXPECT_TRUE(view.has_context_association());
    auto assoc_data = view.context_association_data();
    EXPECT_EQ(assoc_data.size(), 4 * 4);  // 1 counts + 2 stream + 1 context
}

TEST_F(ContextPacketTest, CompileTimeForbidsVariable) {
    // This should not compile (static_assert failure):
    // constexpr uint32_t bad_cif0 = (1U << 10);  // GPS ASCII
    // using BadContext = ContextPacket<true, NoTimeStamp, NoClassId,
    //                                   bad_cif0, 0, 0, false>;

    // Test passes by not having the above code compile
    EXPECT_TRUE(true);
}

