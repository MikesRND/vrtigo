# Payload Format Field (CIF0 bit 15)

*Auto-generated from `tests/cif_access/cif0_15_payload_format_test.cpp`. All examples are tested.*

---

## Real-Valued Signal Configuration

Configure the Payload Format field for real-valued 16-bit signed fixed-point
samples in Q3.12 format with 1024 samples per vector.

```cpp
    using FormatContext = ContextPacket<NoTimestamp, NoClassId, data_payload_format>;

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
```

## Complex Floating-Point Signal Configuration

Configure for complex-valued signals using IEEE 754 single-precision floating-point
format (32 bits per component) with 512 samples per vector and link-efficient packing.

```cpp
    using FormatContext = ContextPacket<NoTimestamp, NoClassId, data_payload_format>;

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
```

## Batch Field Access

The `get_multiple()` method allows reading multiple bitfield values in a single
call, returning a tuple. This is more efficient than individual get() calls.

```cpp
    using FormatContext = ContextPacket<NoTimestamp, NoClassId, data_payload_format>;

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
```

## Parsing Received Payload Format

When receiving VRT packets, `get()` automatically returns strongly-typed
enum values for fields that declare them, requiring no manual casting.

```cpp
    using FormatContext = ContextPacket<NoTimestamp, NoClassId, data_payload_format>;

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
```

