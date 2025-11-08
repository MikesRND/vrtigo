#include "context_test_fixture.hpp"

TEST_F(ContextPacketTest, RejectUnsupportedFields) {
    // Try to create packet with unsupported CIF0 bit 7 (Field Attributes)
    uint32_t header =
        (static_cast<uint32_t>(packet_type::context) << header::PACKET_TYPE_SHIFT) | 3;
    cif::write_u32_safe(buffer.data(), 0, header);

    // CIF0 with unsupported bit 7 set
    uint32_t bad_cif0 = (1U << 7);
    cif::write_u32_safe(buffer.data(), 4, bad_cif0);

    ContextPacketView view(buffer.data(), 3 * 4);
    EXPECT_EQ(view.validate(), validation_error::unsupported_field);
}

TEST_F(ContextPacketTest, RejectReservedBits) {
    uint32_t header = (static_cast<uint32_t>(packet_type::context) << header::PACKET_TYPE_SHIFT) | 3;  // type=4, size=3 words
    cif::write_u32_safe(buffer.data(), 0, header);

    // CIF0 with reserved bit 4 set
    uint32_t bad_cif0 = (1U << 4);
    cif::write_u32_safe(buffer.data(), 4, bad_cif0);

    ContextPacketView view(buffer.data(), 3 * 4);
    EXPECT_EQ(view.validate(), validation_error::unsupported_field);
}

TEST_F(ContextPacketTest, RejectReservedCIF1Bits) {
    // Create packet with CIF1 enabled but reserved bit set
    // Structure: header(1) + CIF0(1) + CIF1(1) = 3 words
    uint32_t header = (static_cast<uint32_t>(packet_type::context) << header::PACKET_TYPE_SHIFT) | 3;
    cif::write_u32_safe(buffer.data(), 0, header);

    // CIF0 with CIF1 enable
    uint32_t cif0_mask = (1U << 1);
    cif::write_u32_safe(buffer.data(), 4, cif0_mask);

    // CIF1 with reserved bit 0 set
    uint32_t bad_cif1 = (1U << 0);
    cif::write_u32_safe(buffer.data(), 8, bad_cif1);

    ContextPacketView view(buffer.data(), 3 * 4);
    EXPECT_EQ(view.validate(), validation_error::unsupported_field);
}

TEST_F(ContextPacketTest, RejectReservedCIF2Bits) {
    // Create packet with CIF2 enabled but reserved bit set
    // Structure: header(1) + CIF0(1) + CIF2(1) = 3 words
    uint32_t header = (static_cast<uint32_t>(packet_type::context) << header::PACKET_TYPE_SHIFT) | 3;
    cif::write_u32_safe(buffer.data(), 0, header);

    // CIF0 with CIF2 enable
    uint32_t cif0_mask = (1U << 2);
    cif::write_u32_safe(buffer.data(), 4, cif0_mask);

    // CIF2 with reserved bit 0 set
    uint32_t bad_cif2 = (1U << 0);
    cif::write_u32_safe(buffer.data(), 8, bad_cif2);

    ContextPacketView view(buffer.data(), 3 * 4);
    EXPECT_EQ(view.validate(), validation_error::unsupported_field);
}

