#include "context_test_fixture.hpp"

TEST_F(ContextPacketTest, RejectUnsupportedFields) {
    // Try to create packet with unsupported CIF0 bit 7 (Field Attributes)
    // Type 4 (context) has stream ID, so structure: header + stream_id + CIF0 = 3 words
    uint32_t header = (static_cast<uint32_t>(PacketType::context) << header::packet_type_shift) | 3;
    cif::write_u32_safe(buffer.data(), 0, header);

    // Stream ID (type 4 has stream ID per VITA 49.2)
    cif::write_u32_safe(buffer.data(), 4, 0x12345678);

    // CIF0 with unsupported bit 7 set
    uint32_t bad_cif0 = (1U << 7);
    cif::write_u32_safe(buffer.data(), 8, bad_cif0);

    RuntimeContextPacket view(buffer.data(), 3 * 4);
    EXPECT_EQ(view.error(), ValidationError::unsupported_field);
}

TEST_F(ContextPacketTest, RejectReservedBits) {
    // Type 4 (context) has stream ID, so structure: header + stream_id + CIF0 = 3 words
    uint32_t header = (static_cast<uint32_t>(PacketType::context) << header::packet_type_shift) |
                      3; // type=4, size=3 words
    cif::write_u32_safe(buffer.data(), 0, header);

    // Stream ID (type 4 has stream ID per VITA 49.2)
    cif::write_u32_safe(buffer.data(), 4, 0x12345678);

    // CIF0 with reserved bit 4 set
    uint32_t bad_cif0 = (1U << 4);
    cif::write_u32_safe(buffer.data(), 8, bad_cif0);

    RuntimeContextPacket view(buffer.data(), 3 * 4);
    EXPECT_EQ(view.error(), ValidationError::unsupported_field);
}

TEST_F(ContextPacketTest, RejectReservedCIF1Bits) {
    // Create packet with CIF1 enabled but reserved bit set
    // Type 4 has stream ID: header(1) + stream_id(1) + CIF0(1) + CIF1(1) = 4 words
    uint32_t header = (static_cast<uint32_t>(PacketType::context) << header::packet_type_shift) | 4;
    cif::write_u32_safe(buffer.data(), 0, header);

    // Stream ID (type 4 has stream ID per VITA 49.2)
    cif::write_u32_safe(buffer.data(), 4, 0x12345678);

    // CIF0 with CIF1 enable
    uint32_t cif0_mask = (1U << 1);
    cif::write_u32_safe(buffer.data(), 8, cif0_mask);

    // CIF1 with reserved bit 0 set
    uint32_t bad_cif1 = (1U << 0);
    cif::write_u32_safe(buffer.data(), 12, bad_cif1);

    RuntimeContextPacket view(buffer.data(), 4 * 4);
    EXPECT_EQ(view.error(), ValidationError::unsupported_field);
}

TEST_F(ContextPacketTest, RejectReservedCIF2Bits) {
    // Create packet with CIF2 enabled but reserved bit set
    // Type 4 has stream ID: header(1) + stream_id(1) + CIF0(1) + CIF2(1) = 4 words
    uint32_t header = (static_cast<uint32_t>(PacketType::context) << header::packet_type_shift) | 4;
    cif::write_u32_safe(buffer.data(), 0, header);

    // Stream ID (type 4 has stream ID per VITA 49.2)
    cif::write_u32_safe(buffer.data(), 4, 0x12345678);

    // CIF0 with CIF2 enable
    uint32_t cif0_mask = (1U << 2);
    cif::write_u32_safe(buffer.data(), 8, cif0_mask);

    // CIF2 with reserved bit 0 set
    uint32_t bad_cif2 = (1U << 0);
    cif::write_u32_safe(buffer.data(), 12, bad_cif2);

    RuntimeContextPacket view(buffer.data(), 4 * 4);
    EXPECT_EQ(view.error(), ValidationError::unsupported_field);
}
