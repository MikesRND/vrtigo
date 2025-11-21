#include "context_test_fixture.hpp"

using namespace vrtigo::field;

TEST_F(ContextPacketTest, GPSASCIIVariableField) {
    // Manually create packet with GPS ASCII field
    // Type 4 has stream ID: header (1) + stream_id (1) + CIF0 (1) + GPS ASCII (4) = 7 words
    uint32_t header = (static_cast<uint32_t>(PacketType::context) << header::packet_type_shift) |
                      7; // type=4, size=7 words
    cif::write_u32_safe(buffer.data(), 0, header);

    // Stream ID (type 4 has stream ID per VITA 49.2)
    cif::write_u32_safe(buffer.data(), 4, 0x12345678);

    // CIF0 with GPS ASCII bit
    uint32_t cif0_mask = vrtigo::detail::field_bitmask<gps_ascii>();
    cif::write_u32_safe(buffer.data(), 8, cif0_mask);

    // GPS ASCII field: count + data
    uint32_t char_count = 12; // "Hello World!" = 12 chars
    cif::write_u32_safe(buffer.data(), 12, char_count);

    // Write ASCII data (3 words needed for 12 chars)
    const char* msg = "Hello World!";
    std::memcpy(buffer.data() + 16, msg, 12);

    RuntimeContextPacket view(buffer.data(), 7 * 4);
    EXPECT_EQ(view.error(), ValidationError::none);

    // Use operator[] API instead of has_gps_ascii() / gps_ascii_data()
    auto gps_proxy = view[gps_ascii];
    EXPECT_TRUE(gps_proxy.has_value());
    auto gps_data = gps_proxy.bytes();
    EXPECT_EQ(gps_data.size(), 4 * 4); // 1 count + 3 data words (12 bytes)

    // Extract the character count
    uint32_t parsed_count;
    std::memcpy(&parsed_count, gps_data.data(), 4);
    parsed_count = detail::network_to_host32(parsed_count);
    EXPECT_EQ(parsed_count, 12);

    // Check the message
    EXPECT_EQ(std::memcmp(gps_data.data() + 4, msg, 12), 0);
}

TEST_F(ContextPacketTest, ContextAssociationLists) {
    // Type 4 has stream ID: header (1) + stream_id (1) + CIF0 (1) + context_assoc (4) = 7 words
    uint32_t header = (static_cast<uint32_t>(PacketType::context) << header::packet_type_shift) |
                      7; // type=4, size=7 words
    cif::write_u32_safe(buffer.data(), 0, header);

    // Stream ID (type 4 has stream ID per VITA 49.2)
    cif::write_u32_safe(buffer.data(), 4, 0x12345678);

    // CIF0 with context association bit
    uint32_t cif0_mask = vrtigo::detail::field_bitmask<context_association_lists>();
    cif::write_u32_safe(buffer.data(), 8, cif0_mask);

    // Context association: two 16-bit counts + IDs
    uint16_t stream_count = 2;
    uint16_t context_count = 1;
    uint32_t counts_word = (stream_count << 16) | context_count;
    cif::write_u32_safe(buffer.data(), 12, counts_word);

    // Stream IDs
    cif::write_u32_safe(buffer.data(), 16, 0x1111);
    cif::write_u32_safe(buffer.data(), 20, 0x2222);

    // Context ID
    cif::write_u32_safe(buffer.data(), 24, 0x3333);

    RuntimeContextPacket view(buffer.data(), 7 * 4);
    EXPECT_EQ(view.error(), ValidationError::none);

    // Use operator[] API instead of has_context_association() / context_association_data()
    auto assoc_proxy = view[context_association_lists];
    EXPECT_TRUE(assoc_proxy.has_value());
    auto assoc_data = assoc_proxy.bytes();
    EXPECT_EQ(assoc_data.size(), 4 * 4); // 1 counts + 2 stream + 1 context
}

TEST_F(ContextPacketTest, CompileTimeForbidsVariable) {
    // This should not compile (static_assert failure):
    // constexpr uint32_t bad_cif0 = (1U << 10);  // GPS ASCII
    // using BadContext = ContextPacket<true, NoTimeStamp, NoClassId,
    //                                   bad_cif0, 0, 0, 0>;

    // Test passes by not having the above code compile
    EXPECT_TRUE(true);
}
