#include <array>

#include <cstring>
#include <gtest/gtest.h>
#include <vrtigo.hpp>

// Test fixture for security validation tests
// Tests dynamic::DataPacketView::parse() validation of untrusted data
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

// Test 1: Valid packet should parse successfully
TEST_F(SecurityTest, ValidPacketParsesSuccessfully) {
    using PacketType =
        vrtigo::typed::SignalDataPacketBuilder<256, vrtigo::UtcRealTimestamp, vrtigo::NoClassId,
                                               vrtigo::WithTrailer>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};
    PacketType packet(buffer);

    // Parse with dynamic::DataPacketView
    auto result = vrtigo::dynamic::DataPacketView::parse(buffer);
    EXPECT_TRUE(result.has_value());

    if (result.has_value()) {
        const auto& view = result.value();
        EXPECT_EQ(view.type(), vrtigo::PacketType::signal_data);
        EXPECT_EQ(view.size_bytes(), PacketType::size_bytes());
        EXPECT_TRUE(view.has_trailer());
        EXPECT_TRUE(view.has_timestamp());
    }
}

// Test 2: Buffer too small (less than header)
TEST_F(SecurityTest, BufferTooSmallForHeader) {
    std::array<uint8_t, 3> tiny_buffer{};

    auto result = vrtigo::dynamic::DataPacketView::parse(tiny_buffer);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, vrtigo::ValidationError::buffer_too_small);
}

// Test 3: Buffer too small for declared size
TEST_F(SecurityTest, BufferTooSmallForDeclaredSize) {
    using PacketType = vrtigo::typed::SignalDataPacketBuilder<128>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};
    PacketType packet(buffer);

    // Parse with truncated buffer
    std::span<const uint8_t> truncated(buffer.data(), buffer.size() - 4);
    auto result = vrtigo::dynamic::DataPacketView::parse(truncated);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, vrtigo::ValidationError::buffer_too_small);
}

// Test 4: Wrong packet type (context packet)
TEST_F(SecurityTest, ContextPacketTypeMismatch) {
    using PacketType = vrtigo::typed::SignalDataPacketBuilder<128>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};
    PacketType packet(buffer);

    // Corrupt packet type field (bits 31-28) to type 4 (context)
    corrupt_header_field(buffer.data(), 0xF0000000, 0x40000000);

    auto result = vrtigo::dynamic::DataPacketView::parse(buffer);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, vrtigo::ValidationError::packet_type_mismatch);
}

// Test 5: Wrong packet type (extension context)
TEST_F(SecurityTest, ExtensionContextPacketTypeMismatch) {
    using PacketType = vrtigo::typed::SignalDataPacketBuilder<128>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};
    PacketType packet(buffer);

    // Corrupt packet type field to type 5 (extension context)
    corrupt_header_field(buffer.data(), 0xF0000000, 0x50000000);

    auto result = vrtigo::dynamic::DataPacketView::parse(buffer);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, vrtigo::ValidationError::packet_type_mismatch);
}

// Test 6: Size field mismatch (too large)
TEST_F(SecurityTest, SizeFieldTooLarge) {
    using PacketType = vrtigo::typed::SignalDataPacketBuilder<128>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};
    PacketType packet(buffer);

    // Corrupt size field (bits 15-0) to claim larger size than buffer
    corrupt_header_field(buffer.data(), 0x0000FFFF, 65535);

    auto result = vrtigo::dynamic::DataPacketView::parse(buffer);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, vrtigo::ValidationError::buffer_too_small);
}

// Test 7: Size field mismatch (zero)
TEST_F(SecurityTest, SizeFieldZero) {
    using PacketType = vrtigo::typed::SignalDataPacketBuilder<128>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};
    PacketType packet(buffer);

    // Corrupt size field to 0
    corrupt_header_field(buffer.data(), 0x0000FFFF, 0);

    auto result = vrtigo::dynamic::DataPacketView::parse(buffer);
    EXPECT_FALSE(result.has_value());
    // Zero size should trigger size_field_mismatch (minimum is 1 word for header)
    EXPECT_EQ(result.error().code, vrtigo::ValidationError::size_field_mismatch);
}

// Test 8: Minimal packet validation (header only)
TEST_F(SecurityTest, MinimalPacketValidation) {
    using PacketType = vrtigo::typed::SignalDataPacketBuilderNoId<0>; // Zero payload

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};
    PacketType packet(buffer);

    auto result = vrtigo::dynamic::DataPacketView::parse(buffer);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(PacketType::size_bytes(), 4); // Just header

    if (result.has_value()) {
        EXPECT_EQ(result.value().payload_size_bytes(), 0);
    }
}

// Test 9: Maximum configuration validation
TEST_F(SecurityTest, MaximumConfigurationValidation) {
    using PacketType =
        vrtigo::typed::SignalDataPacketBuilder<1024, vrtigo::UtcRealTimestamp, vrtigo::NoClassId,
                                               vrtigo::WithTrailer>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};
    PacketType packet(buffer);

    auto result = vrtigo::dynamic::DataPacketView::parse(buffer);
    EXPECT_TRUE(result.has_value());

    if (result.has_value()) {
        EXPECT_EQ(result.value().payload_size_bytes(), 1024 * 4); // In bytes
    }
}

// Test 10: Multiple errors (buffer too small takes priority)
TEST_F(SecurityTest, MultipleErrors) {
    using PacketType =
        vrtigo::typed::SignalDataPacketBuilder<256, vrtigo::UtcRealTimestamp, vrtigo::NoClassId,
                                               vrtigo::WithTrailer>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};
    PacketType packet(buffer);

    // Corrupt type field to context
    corrupt_header_field(buffer.data(), 0xF0000000, 0x40000000);

    // Buffer size error takes priority (parse with tiny buffer)
    std::span<const uint8_t> tiny_span(buffer.data(), 2);
    auto result = vrtigo::dynamic::DataPacketView::parse(tiny_span);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, vrtigo::ValidationError::buffer_too_small);

    // With sufficient buffer, type mismatch should be reported
    auto result2 = vrtigo::dynamic::DataPacketView::parse(buffer);
    EXPECT_FALSE(result2.has_value());
    EXPECT_EQ(result2.error().code, vrtigo::ValidationError::packet_type_mismatch);
}

// Test 11: Validation error string conversion
TEST_F(SecurityTest, ErrorStringConversion) {
    EXPECT_STREQ(vrtigo::validation_error_string(vrtigo::ValidationError::none), "No error");
    EXPECT_STREQ(vrtigo::validation_error_string(vrtigo::ValidationError::buffer_too_small),
                 "Buffer size smaller than declared packet size");
    EXPECT_STREQ(vrtigo::validation_error_string(vrtigo::ValidationError::packet_type_mismatch),
                 "Packet type doesn't match template configuration");
    EXPECT_STREQ(vrtigo::validation_error_string(vrtigo::ValidationError::size_field_mismatch),
                 "Size field doesn't match expected packet size");
}

// Test 12: Type 0 (signal_data_no_id) packet parses successfully
TEST_F(SecurityTest, Type0PacketValidation) {
    using PacketType = vrtigo::typed::SignalDataPacketBuilderNoId<256, vrtigo::UtcRealTimestamp>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};
    PacketType packet(buffer);

    auto result = vrtigo::dynamic::DataPacketView::parse(buffer);
    EXPECT_TRUE(result.has_value());

    if (result.has_value()) {
        EXPECT_EQ(result.value().type(), vrtigo::PacketType::signal_data_no_id);
        EXPECT_FALSE(result.value().has_stream_id());
    }
}

// Test 13: Type 1 (signal_data) packet parses successfully
TEST_F(SecurityTest, Type1PacketValidation) {
    using PacketType = vrtigo::typed::SignalDataPacketBuilder<256, vrtigo::UtcRealTimestamp>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};
    PacketType packet(buffer);
    packet.set_stream_id(0x12345678);

    auto result = vrtigo::dynamic::DataPacketView::parse(buffer);
    EXPECT_TRUE(result.has_value());

    if (result.has_value()) {
        EXPECT_EQ(result.value().type(), vrtigo::PacketType::signal_data);
        EXPECT_TRUE(result.value().has_stream_id());
        EXPECT_EQ(result.value().stream_id().value(), 0x12345678);
    }
}

// Test 14: Parsing untrusted network data pattern
TEST_F(SecurityTest, UntrustedNetworkDataPattern) {
    using PacketType =
        vrtigo::typed::SignalDataPacketBuilder<505, vrtigo::UtcRealTimestamp>; // Fits in 2048 byte
                                                                               // buffer

    // Simulate receiving packet from network
    alignas(4) std::array<uint8_t, 2048> network_buffer{};

    // Build valid packet
    std::span<uint8_t, PacketType::size_bytes()> packet_span{network_buffer.data(),
                                                             PacketType::size_bytes()};
    PacketType tx_packet(packet_span);
    tx_packet.set_stream_id(0x12345678);
    auto ts = vrtigo::UtcRealTimestamp(1234567890, 999999999999ULL);
    tx_packet.set_timestamp(ts);

    // Parse as untrusted data using dynamic::DataPacketView
    auto result = vrtigo::dynamic::DataPacketView::parse(network_buffer);
    EXPECT_TRUE(result.has_value());

    if (result.has_value()) {
        const auto& view = result.value();
        EXPECT_EQ(view.stream_id().value(), 0x12345678);
        auto read_ts = view.timestamp();
        ASSERT_TRUE(read_ts.has_value());
        EXPECT_EQ(read_ts->tsi(), 1234567890);
        EXPECT_EQ(read_ts->tsf(), 999999999999ULL);
    }
}

// Test 15: Defense against size field manipulation
TEST_F(SecurityTest, SizeFieldManipulationDefense) {
    using PacketType = vrtigo::typed::SignalDataPacketBuilder<128>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};
    PacketType packet(buffer);

    // Attacker tries to claim packet is larger than buffer
    corrupt_header_field(buffer.data(), 0x0000FFFF, 65535);

    auto result = vrtigo::dynamic::DataPacketView::parse(buffer);
    EXPECT_FALSE(result.has_value());
    // Should detect that declared size exceeds buffer
    EXPECT_EQ(result.error().code, vrtigo::ValidationError::buffer_too_small);
}

// Test 16: ParseError provides useful information
TEST_F(SecurityTest, ParseErrorInformation) {
    using PacketType = vrtigo::typed::SignalDataPacketBuilder<128>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};
    PacketType packet(buffer);

    // Corrupt to context packet type
    corrupt_header_field(buffer.data(), 0xF0000000, 0x40000000);

    auto result = vrtigo::dynamic::DataPacketView::parse(buffer);
    EXPECT_FALSE(result.has_value());

    const auto& error = result.error();
    EXPECT_EQ(error.code, vrtigo::ValidationError::packet_type_mismatch);
    EXPECT_EQ(error.attempted_type, vrtigo::PacketType::context);
    EXPECT_NE(error.message(), nullptr);
    EXPECT_EQ(error.raw_bytes.size(), buffer.size());
}

// Test 17: Extension data packet types parse successfully
TEST_F(SecurityTest, ExtensionDataPacketValidation) {
    using PacketType = vrtigo::typed::ExtensionDataPacketBuilder<64>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};
    PacketType packet(buffer);

    auto result = vrtigo::dynamic::DataPacketView::parse(buffer);
    EXPECT_TRUE(result.has_value());

    if (result.has_value()) {
        EXPECT_EQ(result.value().type(), vrtigo::PacketType::extension_data);
    }
}

// Test 18: Timestamp parsing variations
TEST_F(SecurityTest, TimestampVariations) {
    // Test TSI only
    {
        using PacketType = vrtigo::typed::SignalDataPacketBuilder<
            32, vrtigo::Timestamp<vrtigo::TsiType::utc, vrtigo::TsfType::none>>;

        alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};
        PacketType packet(buffer);

        auto result = vrtigo::dynamic::DataPacketView::parse(buffer);
        EXPECT_TRUE(result.has_value());
        if (result.has_value()) {
            auto ts = result.value().timestamp();
            ASSERT_TRUE(ts.has_value());
            EXPECT_EQ(ts->tsi_kind(), vrtigo::TsiType::utc);
            EXPECT_EQ(ts->tsf_kind(), vrtigo::TsfType::none);
        }
    }

    // Test TSF only
    {
        using PacketType = vrtigo::typed::SignalDataPacketBuilder<
            32, vrtigo::Timestamp<vrtigo::TsiType::none, vrtigo::TsfType::real_time>>;

        alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};
        PacketType packet(buffer);

        auto result = vrtigo::dynamic::DataPacketView::parse(buffer);
        EXPECT_TRUE(result.has_value());
        if (result.has_value()) {
            auto ts = result.value().timestamp();
            ASSERT_TRUE(ts.has_value());
            EXPECT_EQ(ts->tsi_kind(), vrtigo::TsiType::none);
            EXPECT_EQ(ts->tsf_kind(), vrtigo::TsfType::real_time);
        }
    }

    // Test no timestamp
    {
        using PacketType = vrtigo::typed::SignalDataPacketBuilder<32>;

        alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};
        PacketType packet(buffer);

        auto result = vrtigo::dynamic::DataPacketView::parse(buffer);
        EXPECT_TRUE(result.has_value());
        if (result.has_value()) {
            EXPECT_FALSE(result.value().timestamp().has_value());
        }
    }
}
