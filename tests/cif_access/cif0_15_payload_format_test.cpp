// Self-Documenting CIF Access Test
// Demonstrates accessing and manipenting the Data Packet Payload Format CIF field
//
// This file auto-generates: docs/cif_access/cif0_15_payload_format.md

#include "vrtigo/detail/payload_format.hpp"

#include "../context_test_fixture.hpp"

using namespace vrtigo;
using namespace vrtigo::field;

// [TITLE]
// Payload Format Field (CIF0 bit 15)
// [/TITLE]

TEST_F(ContextPacketTest, CIF0_15_RealSignal) {
    // [EXAMPLE]
    // Real-Valued Signal Configuration
    // [/EXAMPLE]

    // [DESCRIPTION]
    // Configure the Payload Format field for real-valued 16-bit signed fixed-point
    // samples in Q3.12 format with 1024 samples per vector.
    // [/DESCRIPTION]

    // [SNIPPET]
    using FormatContext = typed::ContextPacket<NoTimestamp, NoClassId, data_payload_format>;

    alignas(4) std::array<uint8_t, FormatContext::size_bytes()> buffer{};
    FormatContext packet(buffer);

    auto format = packet[data_payload_format].value();

    // Configure for real-valued 16-bit signed fixed-point samples
    format.set(PayloadFormat::packing_method, false); // Processing-efficient
    format.set(PayloadFormat::real_complex_type, DataSampleType::real);
    format.set(PayloadFormat::data_item_format, DataItemFormatCode::signed_fixed_point);
    format.set(PayloadFormat::data_item_size, 15);          // 16 bits (value is one less)
    format.set(PayloadFormat::data_item_fraction_size, 12); // Q3.12 format
    format.set(PayloadFormat::vector_size, 1023);           // 1024 samples (value is one less)
    // [/SNIPPET]

    // Assertions
    EXPECT_FALSE(format.get(PayloadFormat::packing_method));
    EXPECT_EQ(format.get(PayloadFormat::real_complex_type), DataSampleType::real);
    EXPECT_EQ(format.get(PayloadFormat::data_item_size), 15);
    EXPECT_EQ(format.get(PayloadFormat::data_item_fraction_size), 12);
    EXPECT_EQ(format.get(PayloadFormat::vector_size), 1023);
}

TEST_F(ContextPacketTest, CIF0_15_ComplexFloat) {
    // [EXAMPLE]
    // Complex Floating-Point Signal Configuration
    // [/EXAMPLE]

    // [DESCRIPTION]
    // Configure for complex-valued signals using IEEE 754 single-precision floating-point
    // format (32 bits per component) with 512 samples per vector and link-efficient packing.
    // [/DESCRIPTION]

    // [SNIPPET]
    using FormatContext = typed::ContextPacket<NoTimestamp, NoClassId, data_payload_format>;

    alignas(4) std::array<uint8_t, FormatContext::size_bytes()> buffer{};
    FormatContext packet(buffer);

    auto format = packet[data_payload_format].value();

    // Configure for complex IEEE single-precision floating-point
    format.set(PayloadFormat::packing_method, true); // Link-efficient
    format.set(PayloadFormat::real_complex_type, DataSampleType::complex_cartesian);
    format.set(PayloadFormat::data_item_format, DataItemFormatCode::ieee_754_single_precision);
    format.set(PayloadFormat::data_item_size, 31);         // 32 bits (value is one less)
    format.set(PayloadFormat::data_item_fraction_size, 0); // Not used for IEEE
    format.set(PayloadFormat::vector_size, 511);           // 512 samples
    format.set(PayloadFormat::sample_component_repeat, false);
    // [/SNIPPET]

    // Assertions
    EXPECT_TRUE(format.get(PayloadFormat::packing_method));
    EXPECT_EQ(format.get(PayloadFormat::real_complex_type), DataSampleType::complex_cartesian);
    EXPECT_EQ(format.get(PayloadFormat::data_item_format),
              DataItemFormatCode::ieee_754_single_precision);
    EXPECT_EQ(format.get(PayloadFormat::data_item_size), 31);
    EXPECT_EQ(format.get(PayloadFormat::vector_size), 511);
}

TEST_F(ContextPacketTest, CIF0_15_BatchAccess) {
    // [EXAMPLE]
    // Batch Field Access
    // [/EXAMPLE]

    // [DESCRIPTION]
    // The `get_multiple()` method allows reading multiple bitfield values in a single
    // call, returning a tuple. This is more efficient than individual get() calls.
    // [/DESCRIPTION]

    // [SNIPPET]
    using FormatContext = typed::ContextPacket<NoTimestamp, NoClassId, data_payload_format>;

    alignas(4) std::array<uint8_t, FormatContext::size_bytes()> buffer{};
    FormatContext packet(buffer);

    auto format = packet[data_payload_format].value();

    // Set up a configuration
    format.set(PayloadFormat::packing_method, false);
    format.set(PayloadFormat::real_complex_type, DataSampleType::real);
    format.set(PayloadFormat::data_item_size, 15);
    format.set(PayloadFormat::vector_size, 2047);

    // Get multiple fields at once
    auto [packing, type, size, vec_size] =
        format.get_multiple(PayloadFormat::packing_method, PayloadFormat::real_complex_type,
                            PayloadFormat::data_item_size, PayloadFormat::vector_size);
    // [/SNIPPET]

    // Assertions
    EXPECT_FALSE(packing);
    EXPECT_EQ(type, DataSampleType::real);
    EXPECT_EQ(size, 15);
    EXPECT_EQ(vec_size, 2047);
}

TEST_F(ContextPacketTest, CIF0_15_ParseFormat) {
    // [EXAMPLE]
    // Parsing Received Payload Format
    // [/EXAMPLE]

    // [DESCRIPTION]
    // When receiving VRT packets, `get()` automatically returns strongly-typed
    // enum values for fields that declare them, requiring no manual casting.
    // [/DESCRIPTION]

    // [SNIPPET]
    using FormatContext = typed::ContextPacket<NoTimestamp, NoClassId, data_payload_format>;

    alignas(4) std::array<uint8_t, FormatContext::size_bytes()> buffer{};
    FormatContext packet(buffer);

    // Simulate received data with specific format
    auto format = packet[data_payload_format].value();
    format.set(PayloadFormat::packing_method, true);
    format.set(PayloadFormat::real_complex_type, DataSampleType::complex_polar);
    format.set(PayloadFormat::data_item_format, DataItemFormatCode::unsigned_vrt_4bit);
    format.set(PayloadFormat::data_item_size, 3); // 4 bits
    format.set(PayloadFormat::vector_size, 255);  // 256 samples

    // Parse - get() returns enum types automatically
    auto sample_type = format.get(PayloadFormat::real_complex_type); // Returns DataSampleType
    auto item_format = format.get(PayloadFormat::data_item_format);  // Returns DataItemFormatCode
    auto bits_per_item = format.get(PayloadFormat::data_item_size) + 1;   // Returns uint8_t
    auto samples_per_vector = format.get(PayloadFormat::vector_size) + 1; // Returns uint16_t
    // [/SNIPPET]

    // Assertions
    EXPECT_EQ(sample_type, DataSampleType::complex_polar);
    EXPECT_EQ(item_format, DataItemFormatCode::unsigned_vrt_4bit);
    EXPECT_EQ(bits_per_item, 4);
    EXPECT_EQ(samples_per_vector, 256);
}
