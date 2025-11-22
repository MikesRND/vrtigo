#include <array>

#include <cstring>
#include <gtest/gtest.h>
#include <vrtigo.hpp>

// Test fixture for security validation tests
class SecurityTest : public ::testing::Test {
protected:
    // Helper to corrupt a specific field in the header
    void corrupt_header_field(uint8_t* buffer, uint32_t mask, uint32_t value) {
        uint32_t header;
        std::memcpy(&header, buffer, 4);
        header = vrtigo::detail::network_to_host32(header);
        header = (header & ~mask) | value;
        header = vrtigo::detail::host_to_network32(header);
        std::memcpy(buffer, &header, 4);
    }
};

// Test 1: Valid packet should pass validation
TEST_F(SecurityTest, ValidPacketPassesValidation) {
    using PacketType = vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::UtcRealTimestamp,
                                                vrtigo::Trailer::included, 256>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer;
    PacketType packet(buffer.data());

    // Validate with exact buffer size
    EXPECT_EQ(packet.validate(buffer.size()), vrtigo::ValidationError::none);

    // Validate with larger buffer (typical network scenario)
    EXPECT_EQ(packet.validate(buffer.size() + 1000), vrtigo::ValidationError::none);
}

// Test 2: Buffer too small
TEST_F(SecurityTest, BufferTooSmall) {
    using PacketType = vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::NoTimestamp,
                                                vrtigo::Trailer::none, 128>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer;
    PacketType packet(buffer.data());

    // Validate with buffer smaller than packet
    EXPECT_EQ(packet.validate(buffer.size() - 1), vrtigo::ValidationError::buffer_too_small);
    EXPECT_EQ(packet.validate(0), vrtigo::ValidationError::buffer_too_small);
    EXPECT_EQ(packet.validate(4), vrtigo::ValidationError::buffer_too_small);
}

// Test 3: Wrong packet type
TEST_F(SecurityTest, PacketTypeMismatch) {
    using PacketType = vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::NoTimestamp,
                                                vrtigo::Trailer::none, 128>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer;
    PacketType packet(buffer.data());

    // Corrupt packet type field (bits 31-28) to type 0
    corrupt_header_field(buffer.data(), 0xF0000000, 0x00000000);

    EXPECT_EQ(packet.validate(buffer.size()), vrtigo::ValidationError::packet_type_mismatch);

    // Try other invalid types
    corrupt_header_field(buffer.data(), 0xF0000000, 0x20000000); // Type 2
    EXPECT_EQ(packet.validate(buffer.size()), vrtigo::ValidationError::packet_type_mismatch);

    corrupt_header_field(buffer.data(), 0xF0000000, 0x40000000); // Type 4
    EXPECT_EQ(packet.validate(buffer.size()), vrtigo::ValidationError::packet_type_mismatch);
}

// Test 4: Wrong TSI field
TEST_F(SecurityTest, TSIMismatch) {
    using PacketType =
        vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::UtcRealTimestamp, // TSI = 1, TSF = none
                                 vrtigo::Trailer::none, 128>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer;
    PacketType packet(buffer.data());

    // Corrupt TSI field (bits 23-22) to none (0)
    corrupt_header_field(buffer.data(), 0x00C00000, 0x00000000);
    EXPECT_EQ(packet.validate(buffer.size()), vrtigo::ValidationError::tsi_mismatch);

    // Try TSI = 2 (GPS)
    corrupt_header_field(buffer.data(), 0x00C00000, 0x00800000);
    EXPECT_EQ(packet.validate(buffer.size()), vrtigo::ValidationError::tsi_mismatch);

    // Try TSI = 3 (other)
    corrupt_header_field(buffer.data(), 0x00C00000, 0x00C00000);
    EXPECT_EQ(packet.validate(buffer.size()), vrtigo::ValidationError::tsi_mismatch);
}

// Test 5: Wrong TSF field
TEST_F(SecurityTest, TSFMismatch) {
    using PacketType = vrtigo::SignalDataPacket<
        vrtigo::NoClassId,
        vrtigo::Timestamp<vrtigo::TsiType::none, vrtigo::TsfType::real_time>, // Unusual combo for
                                                                              // validation test
        vrtigo::Trailer::none, 128>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer;
    PacketType packet(buffer.data());

    // Corrupt TSF field (bits 21-20) to none (0)
    corrupt_header_field(buffer.data(), 0x00300000, 0x00000000);
    EXPECT_EQ(packet.validate(buffer.size()), vrtigo::ValidationError::tsf_mismatch);

    // Try TSF = 1 (sample count)
    corrupt_header_field(buffer.data(), 0x00300000, 0x00100000);
    EXPECT_EQ(packet.validate(buffer.size()), vrtigo::ValidationError::tsf_mismatch);

    // Try TSF = 3 (free running)
    corrupt_header_field(buffer.data(), 0x00300000, 0x00300000);
    EXPECT_EQ(packet.validate(buffer.size()), vrtigo::ValidationError::tsf_mismatch);
}

// Test 6: Wrong trailer bit
TEST_F(SecurityTest, TrailerBitMismatch) {
    using PacketType = vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::NoTimestamp,
                                                vrtigo::Trailer::included, // Has trailer
                                                128>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer;
    PacketType packet(buffer.data());

    // Corrupt trailer bit (bit 26) to 0
    corrupt_header_field(buffer.data(), 0x04000000, 0x00000000);
    EXPECT_EQ(packet.validate(buffer.size()), vrtigo::ValidationError::trailer_bit_mismatch);
}

// Test 7: No trailer when expected
TEST_F(SecurityTest, NoTrailerWhenExpected) {
    using PacketType = vrtigo::SignalDataPacketNoId<vrtigo::NoClassId, vrtigo::NoTimestamp,
                                                    vrtigo::Trailer::none, // No trailer
                                                    128>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer;
    PacketType packet(buffer.data());

    // Corrupt trailer bit (bit 26) to 1
    corrupt_header_field(buffer.data(), 0x04000000, 0x04000000);
    EXPECT_EQ(packet.validate(buffer.size()), vrtigo::ValidationError::trailer_bit_mismatch);
}

// Test 8: Wrong size field
TEST_F(SecurityTest, SizeFieldMismatch) {
    using PacketType = vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::UtcRealTimestamp,
                                                vrtigo::Trailer::none, 256>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer;
    PacketType packet(buffer.data());

    // Corrupt size field (bits 15-0) to wrong value
    corrupt_header_field(buffer.data(), 0x0000FFFF, PacketType::size_words + 1);
    EXPECT_EQ(packet.validate(buffer.size()), vrtigo::ValidationError::size_field_mismatch);

    corrupt_header_field(buffer.data(), 0x0000FFFF, PacketType::size_words - 1);
    EXPECT_EQ(packet.validate(buffer.size()), vrtigo::ValidationError::size_field_mismatch);

    corrupt_header_field(buffer.data(), 0x0000FFFF, 0);
    EXPECT_EQ(packet.validate(buffer.size()), vrtigo::ValidationError::size_field_mismatch);

    corrupt_header_field(buffer.data(), 0x0000FFFF, 65535);
    EXPECT_EQ(packet.validate(buffer.size()), vrtigo::ValidationError::size_field_mismatch);
}

// Test 9: Minimal packet validation
TEST_F(SecurityTest, MinimalPacketValidation) {
    using PacketType =
        vrtigo::SignalDataPacketNoId<vrtigo::NoClassId, vrtigo::NoTimestamp, vrtigo::Trailer::none,
                                     0 // Zero payload
                                     >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer;
    PacketType packet(buffer.data());

    EXPECT_EQ(packet.validate(buffer.size()), vrtigo::ValidationError::none);
    EXPECT_EQ(PacketType::size_bytes, 4); // Just header
}

// Test 10: Maximum configuration validation
TEST_F(SecurityTest, MaximumConfigurationValidation) {
    using PacketType = vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::UtcRealTimestamp,
                                                vrtigo::Trailer::included,
                                                1024 // Large payload
                                                >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer;
    PacketType packet(buffer.data());

    EXPECT_EQ(packet.validate(buffer.size()), vrtigo::ValidationError::none);
}

// Test 11: Multiple validation errors (first error should be reported)
TEST_F(SecurityTest, MultipleErrors) {
    using PacketType = vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::UtcRealTimestamp,
                                                vrtigo::Trailer::included, 256>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer;
    PacketType packet(buffer.data());

    // Corrupt multiple fields - buffer too small should be caught first
    corrupt_header_field(buffer.data(), 0xF0000000, 0x00000000); // Wrong type
    corrupt_header_field(buffer.data(), 0x00C00000, 0x00000000); // Wrong TSI

    // Buffer size error takes priority
    EXPECT_EQ(packet.validate(4), vrtigo::ValidationError::buffer_too_small);

    // With sufficient buffer, type mismatch should be reported
    EXPECT_EQ(packet.validate(buffer.size()), vrtigo::ValidationError::packet_type_mismatch);
}

// Test 12: Validation error string conversion
TEST_F(SecurityTest, ErrorStringConversion) {
    EXPECT_STREQ(vrtigo::validation_error_string(vrtigo::ValidationError::none), "No error");
    EXPECT_STREQ(vrtigo::validation_error_string(vrtigo::ValidationError::buffer_too_small),
                 "Buffer size smaller than declared packet size");
    EXPECT_STREQ(vrtigo::validation_error_string(vrtigo::ValidationError::packet_type_mismatch),
                 "Packet type doesn't match template configuration");
    EXPECT_STREQ(vrtigo::validation_error_string(vrtigo::ValidationError::tsi_mismatch),
                 "TSI field doesn't match template configuration");
    EXPECT_STREQ(vrtigo::validation_error_string(vrtigo::ValidationError::tsf_mismatch),
                 "TSF field doesn't match template configuration");
    EXPECT_STREQ(vrtigo::validation_error_string(vrtigo::ValidationError::trailer_bit_mismatch),
                 "Trailer indicator doesn't match template configuration");
    EXPECT_STREQ(vrtigo::validation_error_string(vrtigo::ValidationError::size_field_mismatch),
                 "Size field doesn't match expected packet size");
}

// Test 13: Type 0 packet validation
TEST_F(SecurityTest, Type0PacketValidation) {
    using PacketType = vrtigo::SignalDataPacketNoId<vrtigo::NoClassId, vrtigo::UtcRealTimestamp,
                                                    vrtigo::Trailer::none, 256>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer;
    PacketType packet(buffer.data());

    EXPECT_EQ(packet.validate(buffer.size()), vrtigo::ValidationError::none);

    // Corrupt to type 1 (should fail)
    corrupt_header_field(buffer.data(), 0xF0000000, 0x10000000);
    EXPECT_EQ(packet.validate(buffer.size()), vrtigo::ValidationError::packet_type_mismatch);
}

// Test 14: Parsing untrusted network data pattern
TEST_F(SecurityTest, UntrustedNetworkDataPattern) {
    using PacketType =
        vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::UtcRealTimestamp, vrtigo::Trailer::none,
                                 505 // Fits in 2048 byte buffer: (1+1+1+2+505)*4 = 2040 bytes
                                 >;

    // Simulate receiving packet from network
    alignas(4) std::array<uint8_t, 2048> network_buffer;

    // Build valid packet
    PacketType tx_packet(network_buffer.data());
    tx_packet.set_stream_id(0x12345678);
    auto ts = vrtigo::UtcRealTimestamp(1234567890, 999999999999ULL);
    tx_packet.set_timestamp(ts);

    // Parse as untrusted data (typical usage pattern)
    PacketType rx_packet(network_buffer.data(), false);

    // MUST validate before trusting data
    auto validation_result = rx_packet.validate(network_buffer.size());
    EXPECT_EQ(validation_result, vrtigo::ValidationError::none);

    if (validation_result == vrtigo::ValidationError::none) {
        // Safe to use packet data
        EXPECT_EQ(rx_packet.stream_id(), 0x12345678);
        auto read_ts = rx_packet.timestamp();
        EXPECT_EQ(read_ts.tsi(), 1234567890);
        EXPECT_EQ(read_ts.tsf(), 999999999999ULL);
    }
}

// Test 15: Defense against size field manipulation
TEST_F(SecurityTest, SizeFieldManipulationDefense) {
    using PacketType = vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::NoTimestamp,
                                                vrtigo::Trailer::none, 128>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer;
    PacketType packet(buffer.data());

    // Attacker tries to claim packet is larger than it really is
    corrupt_header_field(buffer.data(), 0x0000FFFF, 65535);

    // Should detect size mismatch
    EXPECT_EQ(packet.validate(buffer.size()), vrtigo::ValidationError::size_field_mismatch);

    // Attacker tries to claim packet is smaller
    corrupt_header_field(buffer.data(), 0x0000FFFF, 1);
    EXPECT_EQ(packet.validate(buffer.size()), vrtigo::ValidationError::size_field_mismatch);
}
