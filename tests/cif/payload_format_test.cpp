#include "vrtigo/detail/payload_format.hpp"

#include "../context_test_fixture.hpp"

using namespace vrtigo;

TEST_F(ContextPacketTest, PayloadFormatBasicAccess) {
    // Test basic three-tier access pattern
    using namespace vrtigo::field;
    using TestContext = ContextPacket<NoTimestamp, NoClassId, data_payload_format>;

    TestContext packet(buffer);

    // Access via field proxy
    auto proxy = packet[field::data_payload_format];
    ASSERT_TRUE(proxy.has_value());

    // Tier 1: Raw bytes (8 bytes)
    auto raw = proxy.bytes();
    EXPECT_EQ(raw.size(), 8);

    // Tier 2: Encoded (generic FieldView<2>)
    auto encoded = proxy.encoded();
    // FieldView<2> represents 2 words (8 bytes)

    // Tier 3: Interpreted (structured PayloadFormat::View)
    auto format = proxy.value();
    EXPECT_EQ(PayloadFormat::View::size(), 8);
}

TEST_F(ContextPacketTest, PayloadFormatBitfieldAccess) {
    // Test individual bitfield access
    using namespace vrtigo::field;
    using TestContext = ContextPacket<NoTimestamp, NoClassId, data_payload_format>;

    TestContext packet(buffer);
    auto format = packet[field::data_payload_format].value();

    // Test packing method (bit 31)
    format.set(PayloadFormat::packing_method, true);
    EXPECT_TRUE(format.get(PayloadFormat::packing_method));
    format.set(PayloadFormat::packing_method, false);
    EXPECT_FALSE(format.get(PayloadFormat::packing_method));

    // Test real/complex type (bits 30-29)
    format.set(PayloadFormat::real_complex_type, 0b01);
    EXPECT_EQ(format.get(PayloadFormat::real_complex_type), 0b01);

    // Test data item format (bits 28-24)
    format.set(PayloadFormat::data_item_format, 0b01110); // IEEE single precision
    EXPECT_EQ(format.get(PayloadFormat::data_item_format), 0b01110);

    // Test data item size (bits 5-0)
    format.set(PayloadFormat::data_item_size, 16);
    EXPECT_EQ(format.get(PayloadFormat::data_item_size), 16);

    // Test vector size (word 1, bits 15-0)
    format.set(PayloadFormat::vector_size, 1024);
    EXPECT_EQ(format.get(PayloadFormat::vector_size), 1024);

    // Test repeat count (word 1, bits 31-16)
    format.set(PayloadFormat::repeat_count, 256);
    EXPECT_EQ(format.get(PayloadFormat::repeat_count), 256);
}

TEST_F(ContextPacketTest, PayloadFormatMultipleFields) {
    // Test get_multiple
    using namespace vrtigo::field;
    using TestContext = ContextPacket<NoTimestamp, NoClassId, data_payload_format>;

    TestContext packet(buffer);
    auto format = packet[field::data_payload_format].value();

    // Set values
    format.set(PayloadFormat::packing_method, true);
    format.set(PayloadFormat::data_item_size, 32);
    format.set(PayloadFormat::vector_size, 2048);

    // Get multiple values at once
    auto [packing, size, vec_size] = format.get_multiple(
        PayloadFormat::packing_method, PayloadFormat::data_item_size, PayloadFormat::vector_size);

    EXPECT_TRUE(packing);
    EXPECT_EQ(size, 32);
    EXPECT_EQ(vec_size, 2048);
}

TEST_F(ContextPacketTest, PayloadFormatEndianness) {
    // Test that fields are properly byte-swapped
    using namespace vrtigo::field;
    using TestContext = ContextPacket<NoTimestamp, NoClassId, data_payload_format>;

    TestContext packet(buffer);
    auto format = packet[field::data_payload_format].value();

    // Set a value with a known bit pattern
    format.set(PayloadFormat::vector_size, 0x1234);

    // Read the raw bytes
    auto raw = packet[field::data_payload_format].bytes();

    // In network byte order (big-endian), the vector_size is in word 1, bits 15-0
    // Word 1 starts at offset 4
    // Bytes 6-7 contain vector_size
    uint8_t high_byte = static_cast<uint8_t>(raw[6]); // High byte of vector_size
    uint8_t low_byte = static_cast<uint8_t>(raw[7]);  // Low byte of vector_size

    // In big-endian, high byte comes first
    EXPECT_EQ(high_byte, 0x12);
    EXPECT_EQ(low_byte, 0x34);

    // Verify we can read it back correctly
    EXPECT_EQ(format.get(PayloadFormat::vector_size), 0x1234);
}

TEST_F(ContextPacketTest, PayloadFormatClear) {
    // Test clear method
    using namespace vrtigo::field;
    using TestContext = ContextPacket<NoTimestamp, NoClassId, data_payload_format>;

    TestContext packet(buffer);
    auto format = packet[field::data_payload_format].value();

    // Set some values
    format.set(PayloadFormat::packing_method, true);
    format.set(PayloadFormat::data_item_size, 16);
    format.set(PayloadFormat::vector_size, 1024);

    // Clear all fields
    format.clear();

    // Verify all fields are zero
    EXPECT_FALSE(format.get(PayloadFormat::packing_method));
    EXPECT_EQ(format.get(PayloadFormat::data_item_size), 0);
    EXPECT_EQ(format.get(PayloadFormat::vector_size), 0);
}

TEST_F(ContextPacketTest, PayloadFormatEnumValues) {
    // Test using enum values with automatic enum return
    using namespace vrtigo::field;
    using TestContext = ContextPacket<NoTimestamp, NoClassId, data_payload_format>;

    TestContext packet(buffer);
    auto format = packet[field::data_payload_format].value();

    // Set and get real/complex type using enum (no cast needed!)
    format.set(PayloadFormat::real_complex_type, DataSampleType::complex_cartesian);
    EXPECT_EQ(format.get(PayloadFormat::real_complex_type), DataSampleType::complex_cartesian);

    // Set and get data item format using enum (no cast needed!)
    format.set(PayloadFormat::data_item_format, DataItemFormatCode::ieee_754_single_precision);
    EXPECT_EQ(format.get(PayloadFormat::data_item_format),
              DataItemFormatCode::ieee_754_single_precision);
}

TEST_F(ContextPacketTest, PayloadFormatAllFields) {
    // Comprehensive test of all fields
    using namespace vrtigo::field;
    using TestContext = ContextPacket<NoTimestamp, NoClassId, data_payload_format>;

    TestContext packet(buffer);
    auto format = packet[field::data_payload_format].value();

    // Set all word 0 fields
    format.set(PayloadFormat::packing_method, true);
    format.set(PayloadFormat::real_complex_type, 0b01);
    format.set(PayloadFormat::data_item_format, 0b01110);
    format.set(PayloadFormat::sample_component_repeat, true);
    format.set(PayloadFormat::event_tag_size, 3);
    format.set(PayloadFormat::channel_tag_size, 7);
    format.set(PayloadFormat::data_item_fraction_size, 12);
    format.set(PayloadFormat::item_packing_field_size, 31);
    format.set(PayloadFormat::data_item_size, 15);

    // Set all word 1 fields
    format.set(PayloadFormat::repeat_count, 511);
    format.set(PayloadFormat::vector_size, 4095);

    // Verify all fields
    EXPECT_TRUE(format.get(PayloadFormat::packing_method));
    EXPECT_EQ(format.get(PayloadFormat::real_complex_type), 0b01);
    EXPECT_EQ(format.get(PayloadFormat::data_item_format), 0b01110);
    EXPECT_TRUE(format.get(PayloadFormat::sample_component_repeat));
    EXPECT_EQ(format.get(PayloadFormat::event_tag_size), 3);
    EXPECT_EQ(format.get(PayloadFormat::channel_tag_size), 7);
    EXPECT_EQ(format.get(PayloadFormat::data_item_fraction_size), 12);
    EXPECT_EQ(format.get(PayloadFormat::item_packing_field_size), 31);
    EXPECT_EQ(format.get(PayloadFormat::data_item_size), 15);
    EXPECT_EQ(format.get(PayloadFormat::repeat_count), 511);
    EXPECT_EQ(format.get(PayloadFormat::vector_size), 4095);
}
