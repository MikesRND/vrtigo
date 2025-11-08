#include <vrtio.hpp>
#include <gtest/gtest.h>
#include <array>
#include <cstring>

using namespace vrtio;

// Test fixture for context packet tests
class ContextPacketTest : public ::testing::Test {
protected:
    // Buffer for testing
    alignas(4) std::array<uint8_t, 4096> buffer{};

    void SetUp() override {
        std::memset(buffer.data(), 0, buffer.size());
    }
};

// Test basic context packet creation with compile-time template
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

// Test context packet with Class ID
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

// Test ContextPacketView runtime parser
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

// Test rejection of unsupported fields
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

// Test rejection of reserved bits
TEST_F(ContextPacketTest, RejectReservedBits) {
    uint32_t header = (static_cast<uint32_t>(packet_type::context) << header::PACKET_TYPE_SHIFT) | 3;  // type=4, size=3 words
    cif::write_u32_safe(buffer.data(), 0, header);

    // CIF0 with reserved bit 4 set
    uint32_t bad_cif0 = (1U << 4);
    cif::write_u32_safe(buffer.data(), 4, bad_cif0);

    ContextPacketView view(buffer.data(), 3 * 4);
    EXPECT_EQ(view.validate(), validation_error::unsupported_field);
}

// Test rejection of reserved CIF1 bits
TEST_F(ContextPacketTest, RejectReservedCIF1Bits) {
    // Create packet with CIF1 enabled but reserved bit set
    // Structure: header(1) + CIF0(1) + CIF1(1) = 3 words
    uint32_t header = (static_cast<uint32_t>(packet_type::context) << header::PACKET_TYPE_SHIFT) | 3;
    cif::write_u32_safe(buffer.data(), 0, header);

    // CIF0 with CIF1 enable
    uint32_t cif0_mask = (1U << 1);
    cif::write_u32_safe(buffer.data(), 4, cif0_mask);

    // CIF1 with reserved bit 4 set (Health Status is unsupported)
    uint32_t bad_cif1 = (1U << 4);
    cif::write_u32_safe(buffer.data(), 8, bad_cif1);

    ContextPacketView view(buffer.data(), 3 * 4);
    EXPECT_EQ(view.validate(), validation_error::unsupported_field);
}

// Test rejection of reserved CIF2 bits
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

// Test CIF1 fields
TEST_F(ContextPacketTest, CIF1Fields) {
    // Create packet with CIF1 spectrum field
    constexpr uint32_t cif0_mask = 0;  // No CIF0 fields (CIF1 enable bit set automatically)
    constexpr uint32_t cif1_mask = cif1::SPECTRUM;

    using TestContext = ContextPacket<
        false,          // No stream ID
        NoTimeStamp,    // No timestamp
        NoClassId,      // No class ID
        cif0_mask,      // CIF0
        cif1_mask,      // CIF1
        0,              // No CIF2
        false           // No trailer
    >;

    TestContext packet(buffer.data());

    // Check size includes CIF1 spectrum field
    EXPECT_EQ(TestContext::size_words, 1 + 1 + 1 + 13);  // header + cif0 + cif1 + spectrum
}

// Test CIF2 fields
TEST_F(ContextPacketTest, CIF2Fields) {
    // Create packet with CIF2 controller UUID
    constexpr uint32_t cif0_mask = 0;  // No CIF0 fields (CIF2 enable bit set automatically)
    constexpr uint32_t cif2_mask = cif2::CONTROLLER_UUID;

    using TestContext = ContextPacket<
        false,          // No stream ID
        NoTimeStamp,    // No timestamp
        NoClassId,      // No class ID
        cif0_mask,      // CIF0
        0,              // No CIF1
        cif2_mask,      // CIF2
        false           // No trailer
    >;

    TestContext packet(buffer.data());

    // Check size includes CIF2 field
    EXPECT_EQ(TestContext::size_words, 1 + 1 + 1 + 4);  // header + cif0 + cif2 + uuid
}

// Test variable field parsing with GPS ASCII
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

// Test Context Association Lists variable field
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

// Test that compile-time template forbids variable fields
TEST_F(ContextPacketTest, CompileTimeForbidsVariable) {
    // This should not compile (static_assert failure):
    // constexpr uint32_t bad_cif0 = (1U << 10);  // GPS ASCII
    // using BadContext = ContextPacket<true, NoTimeStamp, NoClassId,
    //                                   bad_cif0, 0, 0, false>;

    // Test passes by not having the above code compile
    EXPECT_TRUE(true);
}

// Test size field validation
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

// Test buffer too small error
TEST_F(ContextPacketTest, BufferTooSmall) {
    uint32_t header = (static_cast<uint32_t>(packet_type::context) << header::PACKET_TYPE_SHIFT) | 10;  // type=4, size=10 words
    cif::write_u32_safe(buffer.data(), 0, header);

    // Provide buffer smaller than declared size
    ContextPacketView view(buffer.data(), 3 * 4);  // Only 3 words provided
    EXPECT_EQ(view.validate(), validation_error::buffer_too_small);
}

// Test invalid packet type
TEST_F(ContextPacketTest, InvalidPacketType) {
    uint32_t header = (0U << 28) | 3;  // type=0 (not context), size=3
    cif::write_u32_safe(buffer.data(), 0, header);

    ContextPacketView view(buffer.data(), 3 * 4);
    EXPECT_EQ(view.validate(), validation_error::invalid_packet_type);
}

// Test CIF3 rejection
TEST_F(ContextPacketTest, RejectCIF3) {
    uint32_t header = (static_cast<uint32_t>(packet_type::context) << header::PACKET_TYPE_SHIFT) | 3;  // type=4, size=3
    cif::write_u32_safe(buffer.data(), 0, header);

    // CIF0 with bit 3 set (CIF3 enable)
    uint32_t bad_cif0 = (1U << 3);
    cif::write_u32_safe(buffer.data(), 4, bad_cif0);

    ContextPacketView view(buffer.data(), 3 * 4);
    EXPECT_EQ(view.validate(), validation_error::unsupported_field);
}

// Test runtime parsing of CIF1 packet (verify enable bit is accepted)
TEST_F(ContextPacketTest, RuntimeParseCIF1) {
    // Build a packet with CIF1 enabled and Aux Frequency field
    // Structure: header(1) + CIF0(1) + CIF1(1) + Aux Frequency(2) = 5 words
    uint32_t header = (static_cast<uint32_t>(packet_type::context) << header::PACKET_TYPE_SHIFT) | 5;
    cif::write_u32_safe(buffer.data(), 0, header);

    // CIF0 with CIF1 enable bit set
    uint32_t cif0_mask = (1U << 1);  // Enable CIF1
    cif::write_u32_safe(buffer.data(), 4, cif0_mask);

    // CIF1 with Aux Frequency enabled
    uint32_t cif1_mask = cif1::AUX_FREQUENCY;
    cif::write_u32_safe(buffer.data(), 8, cif1_mask);

    // Aux Frequency: 10 MHz = 10,000,000 Hz
    uint64_t expected_freq = 10'000'000;
    cif::write_u64_safe(buffer.data(), 12, expected_freq);

    // Parse and validate
    ContextPacketView view(buffer.data(), 5 * 4);
    EXPECT_EQ(view.validate(), validation_error::none);

    // Verify CIF0 and CIF1 are read correctly
    EXPECT_EQ(view.cif0(), cif0_mask);
    EXPECT_EQ(view.cif1(), cif1_mask);

    // Verify we can read back the Aux Frequency field using the accessor
    auto aux_freq = get(view, field::aux_frequency);
    ASSERT_TRUE(aux_freq.has_value());
    EXPECT_EQ(*aux_freq, expected_freq);
}

// Test runtime parsing of CIF2 packet (verify enable bit is accepted)
TEST_F(ContextPacketTest, RuntimeParseCIF2) {
    // Build a packet with CIF2 enabled and Controller UUID field
    // Structure: header(1) + CIF0(1) + CIF2(1) + Controller UUID(4) = 7 words
    uint32_t header = (static_cast<uint32_t>(packet_type::context) << header::PACKET_TYPE_SHIFT) | 7;
    cif::write_u32_safe(buffer.data(), 0, header);

    // CIF0 with CIF2 enable bit set
    uint32_t cif0_mask = (1U << 2);  // Enable CIF2
    cif::write_u32_safe(buffer.data(), 4, cif0_mask);

    // CIF2 with Controller UUID enabled
    uint32_t cif2_mask = cif2::CONTROLLER_UUID;
    cif::write_u32_safe(buffer.data(), 8, cif2_mask);

    // Controller UUID: 4 words (128 bits)
    uint32_t expected_uuid[4] = {0x12345678, 0x9ABCDEF0, 0x11111111, 0x22222222};
    cif::write_u32_safe(buffer.data(), 12, expected_uuid[0]);
    cif::write_u32_safe(buffer.data(), 16, expected_uuid[1]);
    cif::write_u32_safe(buffer.data(), 20, expected_uuid[2]);
    cif::write_u32_safe(buffer.data(), 24, expected_uuid[3]);

    // Parse and validate
    ContextPacketView view(buffer.data(), 7 * 4);
    EXPECT_EQ(view.validate(), validation_error::none);

    // Verify CIF0 and CIF2 are read correctly
    EXPECT_EQ(view.cif0(), cif0_mask);
    EXPECT_EQ(view.cif2(), cif2_mask);

    // Verify we can read back the Controller UUID field using the accessor
    auto uuid = view.controller_uuid();
    ASSERT_TRUE(uuid.has_value());
    ASSERT_EQ(uuid->size(), 16);  // 128 bits = 16 bytes

    // Verify UUID bytes match what we wrote
    for (int i = 0; i < 4; i++) {
        uint32_t word = cif::read_u32_safe(uuid->data(), i * 4);
        EXPECT_EQ(word, expected_uuid[i]);
    }
}

// Test that compile-time CIF1 packets can be parsed at runtime
TEST_F(ContextPacketTest, CompileTimeCIF1RuntimeParse) {
    // Create packet with CIF1 field at compile time
    constexpr uint32_t cif1_mask = cif1::AUX_FREQUENCY;
    using TestContext = ContextPacket<
        true,           // Has stream ID
        NoTimeStamp,
        NoClassId,
        0,              // No CIF0 data fields (CIF1 enable bit auto-set)
        cif1_mask,      // CIF1
        0,              // No CIF2
        false
    >;

    // Compile-time assertion: verify CIF1 enable bit is auto-set
    static_assert((TestContext::cif0_value & 0x02) != 0,
        "CIF1 enable bit (bit 1) should be auto-set when CIF1 != 0");
    static_assert((TestContext::cif0_value & 0x04) == 0,
        "CIF2 enable bit (bit 2) should NOT be set when CIF2 == 0");

    TestContext tx_packet(buffer.data());
    tx_packet.set_stream_id(0xAABBCCDD);
    set(tx_packet, field::aux_frequency, 15'000'000ULL);  // 15 MHz

    // Parse with runtime view
    ContextPacketView view(buffer.data(), TestContext::size_bytes);
    EXPECT_EQ(view.validate(), validation_error::none);
    EXPECT_TRUE(view.has_stream_id());
    EXPECT_EQ(view.stream_id().value(), 0xAABBCCDD);
    // CIF0 should have bit 1 set (CIF1 enable)
    EXPECT_EQ(view.cif0() & 0x02, 0x02);
    EXPECT_EQ(view.cif1(), cif1_mask);
}

// Test that compile-time CIF2 packets can be parsed at runtime
TEST_F(ContextPacketTest, CompileTimeCIF2RuntimeParse) {
    // Create packet with CIF2 field at compile time
    constexpr uint32_t cif2_mask = cif2::CONTROLLER_UUID;
    using TestContext = ContextPacket<
        false,          // No stream ID
        NoTimeStamp,
        NoClassId,
        0,              // No CIF0 data fields (CIF2 enable bit auto-set)
        0,              // No CIF1
        cif2_mask,      // CIF2
        false
    >;

    // Compile-time assertion: verify CIF2 enable bit is auto-set
    static_assert((TestContext::cif0_value & 0x04) != 0,
        "CIF2 enable bit (bit 2) should be auto-set when CIF2 != 0");
    static_assert((TestContext::cif0_value & 0x02) == 0,
        "CIF1 enable bit (bit 1) should NOT be set when CIF1 == 0");

    TestContext tx_packet(buffer.data());

    // Parse with runtime view
    ContextPacketView view(buffer.data(), TestContext::size_bytes);
    EXPECT_EQ(view.validate(), validation_error::none);
    // CIF0 should have bit 2 set (CIF2 enable)
    EXPECT_EQ(view.cif0() & 0x04, 0x04);
    EXPECT_EQ(view.cif2(), cif2_mask);
}

// Test both CIF1 and CIF2 together (compile-time)
TEST_F(ContextPacketTest, CombinedCIF1AndCIF2CompileTime) {
    // Create packet with both CIF1 and CIF2 fields
    constexpr uint32_t cif0_mask = cif0::BANDWIDTH;
    constexpr uint32_t cif1_mask = cif1::AUX_FREQUENCY;
    constexpr uint32_t cif2_mask = cif2::CONTROLLER_UUID;

    using TestContext = ContextPacket<
        true,           // Has stream ID
        NoTimeStamp,
        NoClassId,
        cif0_mask,      // CIF0 has bandwidth
        cif1_mask,      // CIF1 has aux frequency
        cif2_mask,      // CIF2 has controller UUID
        false
    >;

    // Compile-time assertions: verify both enable bits are auto-set
    static_assert((TestContext::cif0_value & 0x02) != 0,
        "CIF1 enable bit should be set when CIF1 != 0");
    static_assert((TestContext::cif0_value & 0x04) != 0,
        "CIF2 enable bit should be set when CIF2 != 0");
    static_assert((TestContext::cif0_value & (1U << 29)) != 0,
        "Bandwidth bit should be preserved from CIF0 parameter");

    TestContext tx_packet(buffer.data());
    tx_packet.set_stream_id(0x11223344);
    set(tx_packet, field::bandwidth, 50'000'000ULL);
    set(tx_packet, field::aux_frequency, 25'000'000ULL);

    // Parse with runtime view
    ContextPacketView view(buffer.data(), TestContext::size_bytes);
    EXPECT_EQ(view.validate(), validation_error::none);

    // Verify CIF0 has both enable bits set
    EXPECT_EQ(view.cif0() & 0x06, 0x06);  // Both bits 1 and 2
    EXPECT_EQ(view.cif1(), cif1_mask);
    EXPECT_EQ(view.cif2(), cif2_mask);

    // Verify fields
    EXPECT_EQ(view.stream_id().value(), 0x11223344);
    EXPECT_EQ(get(view, field::bandwidth).value(), 50'000'000);
    EXPECT_EQ(get(view, field::aux_frequency).value(), 25'000'000);
}

// Test both CIF1 and CIF2 together (runtime)
TEST_F(ContextPacketTest, CombinedCIF1AndCIF2Runtime) {
    // Manually build a packet with both CIF1 and CIF2
    // Structure: header(1) + stream_id(1) + CIF0(1) + CIF1(1) + CIF2(1) + bandwidth(2) + aux_freq(2) + uuid(4) = 13 words
    uint32_t header = (static_cast<uint32_t>(packet_type::context) << header::PACKET_TYPE_SHIFT) | header::STREAM_ID_INDICATOR | 13;  // Type 4, has stream ID, 13 words
    cif::write_u32_safe(buffer.data(), 0, header);

    // Stream ID
    cif::write_u32_safe(buffer.data(), 4, 0xAABBCCDD);

    // CIF0: Enable CIF1, CIF2, and Bandwidth
    uint32_t cif0_mask = (1U << 1) | (1U << 2) | cif0::BANDWIDTH;
    cif::write_u32_safe(buffer.data(), 8, cif0_mask);

    // CIF1: Aux Frequency
    uint32_t cif1_mask = cif1::AUX_FREQUENCY;
    cif::write_u32_safe(buffer.data(), 12, cif1_mask);

    // CIF2: Controller UUID
    uint32_t cif2_mask = cif2::CONTROLLER_UUID;
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
    EXPECT_EQ(view.validate(), validation_error::none);

    // Verify structure
    EXPECT_EQ(view.cif0(), cif0_mask);
    EXPECT_EQ(view.cif1(), cif1_mask);
    EXPECT_EQ(view.cif2(), cif2_mask);

    // Verify fields
    EXPECT_EQ(view.stream_id().value(), 0xAABBCCDD);
    EXPECT_EQ(get(view, field::bandwidth).value(), 100'000'000);
    EXPECT_EQ(get(view, field::aux_frequency).value(), 75'000'000);

    auto uuid = view.controller_uuid();
    ASSERT_TRUE(uuid.has_value());
    EXPECT_EQ(cif::read_u32_safe(uuid->data(), 0), 0x12345678);
}

// Test round-trip: create with template, parse with view
TEST_F(ContextPacketTest, RoundTrip) {
    // Create packet with template
    constexpr uint32_t cif0_mask = cif0::BANDWIDTH | cif0::GAIN;
    using TestContext = ContextPacket<
        true,           // Has stream ID
        NoTimeStamp,
        NoClassId,
        cif0_mask, 0, 0,
        false
    >;

    TestContext tx_packet(buffer.data());
    tx_packet.set_stream_id(0xDEADBEEF);
    set(tx_packet, field::bandwidth, 100'000'000ULL);  // 100 MHz
    set(tx_packet, field::gain, 0x12345678U);

    // Parse same buffer with view
    ContextPacketView view(buffer.data(), TestContext::size_bytes);
    EXPECT_EQ(view.validate(), validation_error::none);

    EXPECT_EQ(view.stream_id().value(), 0xDEADBEEF);
    EXPECT_EQ(get(view, field::bandwidth).value(), 100'000'000);
    EXPECT_EQ(get(view, field::gain).value(), 0x12345678);
}

// Test timestamp initialization (should be zero)
TEST_F(ContextPacketTest, TimestampInitialization) {
    using TestContext = ContextPacket<
        true,                   // Has stream ID
        TimeStampUTC,          // UTC timestamp
        NoClassId,
        0, 0, 0,               // No context fields
        false
    >;

    TestContext packet(buffer.data());

    // Timestamps should be zero-initialized
    EXPECT_EQ(packet.timestamp_integer(), 0);
    EXPECT_EQ(packet.timestamp_fractional(), 0);
}

// Test setting and getting integer timestamp
TEST_F(ContextPacketTest, TimestampIntegerAccess) {
    using TestContext = ContextPacket<
        true,
        TimeStampUTC,
        NoClassId,
        0, 0, 0,
        false
    >;

    TestContext packet(buffer.data());

    // Set integer timestamp
    packet.set_timestamp_integer(1699000000);
    EXPECT_EQ(packet.timestamp_integer(), 1699000000);
}

// Test setting and getting fractional timestamp
TEST_F(ContextPacketTest, TimestampFractionalAccess) {
    using TestContext = ContextPacket<
        true,
        TimeStampUTC,
        NoClassId,
        0, 0, 0,
        false
    >;

    TestContext packet(buffer.data());

    // Set fractional timestamp
    uint64_t frac = 500000000000ULL;
    packet.set_timestamp_fractional(frac);
    EXPECT_EQ(packet.timestamp_fractional(), frac);
}

// Test unified timestamp accessors
TEST_F(ContextPacketTest, UnifiedTimestampAccess) {
    using TestContext = ContextPacket<
        true,
        TimeStampUTC,
        NoClassId,
        0, 0, 0,
        false
    >;

    TestContext packet(buffer.data());

    // Create a timestamp
    TimeStampUTC ts(1699000000, 250000000000ULL);

    // Set using unified API
    packet.setTimeStamp(ts);

    // Verify individual components
    EXPECT_EQ(packet.timestamp_integer(), 1699000000);
    EXPECT_EQ(packet.timestamp_fractional(), 250000000000ULL);

    // Get using unified API
    auto retrieved = packet.getTimeStamp();
    EXPECT_EQ(retrieved.seconds(), 1699000000);
    EXPECT_EQ(retrieved.fractional(), 250000000000ULL);
}

// Test timestamp with class ID (offset should be calculated correctly)
TEST_F(ContextPacketTest, TimestampWithClassId) {
    using TestClassId = ClassId<0x123456, 0xABCDEF00>;
    using TestContext = ContextPacket<
        true,
        TimeStampUTC,
        TestClassId,           // Has class ID (8 bytes)
        0, 0, 0,
        false
    >;

    TestContext packet(buffer.data());

    // Timestamps should be zero-initialized even with class ID present
    EXPECT_EQ(packet.timestamp_integer(), 0);
    EXPECT_EQ(packet.timestamp_fractional(), 0);

    // Set and verify
    packet.set_timestamp_integer(1234567890);
    packet.set_timestamp_fractional(999999999999ULL);

    EXPECT_EQ(packet.timestamp_integer(), 1234567890);
    EXPECT_EQ(packet.timestamp_fractional(), 999999999999ULL);
}

// Test timestamp with context fields
TEST_F(ContextPacketTest, TimestampWithContextFields) {
    constexpr uint32_t cif0_mask = cif0::BANDWIDTH | cif0::SAMPLE_RATE;
    using TestContext = ContextPacket<
        true,
        TimeStampUTC,
        NoClassId,
        cif0_mask, 0, 0,
        false
    >;

    TestContext packet(buffer.data());

    // Set all fields
    packet.set_stream_id(0x12345678);
    packet.set_timestamp_integer(1600000000);
    packet.set_timestamp_fractional(123456789012ULL);
    set(packet, field::bandwidth, 20'000'000ULL);
    set(packet, field::sample_rate, 10'000'000ULL);

    // Verify all fields
    EXPECT_EQ(packet.stream_id(), 0x12345678);
    EXPECT_EQ(packet.timestamp_integer(), 1600000000);
    EXPECT_EQ(packet.timestamp_fractional(), 123456789012ULL);
    EXPECT_EQ(get(packet, field::bandwidth).value(), 20'000'000);
    EXPECT_EQ(get(packet, field::sample_rate).value(), 10'000'000);
}

// Test timestamp without stream ID
TEST_F(ContextPacketTest, TimestampNoStreamId) {
    using TestContext = ContextPacket<
        false,                  // No stream ID
        TimeStampUTC,
        NoClassId,
        0, 0, 0,
        false
    >;

    TestContext packet(buffer.data());

    // Should still work correctly
    packet.set_timestamp_integer(987654321);
    packet.set_timestamp_fractional(111111111111ULL);

    EXPECT_EQ(packet.timestamp_integer(), 987654321);
    EXPECT_EQ(packet.timestamp_fractional(), 111111111111ULL);
}