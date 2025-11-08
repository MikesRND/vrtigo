// Context Packet Example for VRTIO
//
// Demonstrates creating and parsing VRT context packets using both
// compile-time templates and runtime parsing.

#include <vrtio.hpp>
#include <iostream>
#include <iomanip>
#include <array>
#include <cstring>

using namespace vrtio;

// Example 1: Creating a context packet at compile time
void example_compile_time_context() {
    std::cout << "\n=== Compile-Time Context Packet Example ===\n";

    // Define which CIF fields we want using named constants
    constexpr uint32_t cif0_mask =
        cif0::BANDWIDTH |
        cif0::SAMPLE_RATE |
        cif0::GAIN |
        cif0::DEVICE_ID;

    // Define a packet type with these fields
    using SignalContext = ContextPacket<
        true,           // Has stream ID
        NoTimeStamp,    // No timestamp
        NoClassId,      // No class ID
        cif0_mask,      // CIF0 configuration
        0,              // No CIF1
        0,              // No CIF2
        false           // No trailer
    >;

    // Create buffer and packet
    alignas(4) std::array<uint8_t, SignalContext::size_bytes> buffer{};
    SignalContext packet(buffer.data());

    // Set the stream ID
    packet.set_stream_id(0x12345678);

    // Set signal parameters
    set(packet, field::bandwidth, 20'000'000ULL);      // 20 MHz
    set(packet, field::sample_rate, 10'000'000ULL);    // 10 MSPS
    set(packet, field::gain, 0x00100000U);             // Gain value

    // Device ID is 64-bit (OUI + device code)
    // For now we'll just set a simple value
    // Note: Need to add device ID setter to ContextPacket

    std::cout << "Created context packet:\n";
    std::cout << "  Size: " << SignalContext::size_bytes << " bytes\n";
    std::cout << "  Stream ID: 0x" << std::hex << packet.stream_id() << std::dec << "\n";
    std::cout << "  Bandwidth: " << get(packet, field::bandwidth).value() / 1'000'000.0 << " MHz\n";
    std::cout << "  Sample Rate: " << get(packet, field::sample_rate).value() / 1'000'000.0 << " MSPS\n";
    std::cout << "  Gain: 0x" << std::hex << get(packet, field::gain).value() << std::dec << "\n";
}

// Example 2: Creating a context packet with Class ID
void example_with_class_id() {
    std::cout << "\n=== Context Packet with Class ID Example ===\n";

    // Define a Class ID (OUI and PCC)
    using MyClassId = ClassId<0x00FF00, 0xABCD1234>;  // Example OUI and PCC

    // Simple context with just bandwidth
    constexpr uint32_t cif0_mask = cif0::BANDWIDTH;

    using ClassifiedContext = ContextPacket<
        true,           // Has stream ID
        NoTimeStamp,
        MyClassId,      // Has class ID
        cif0_mask, 0, 0,
        false
    >;

    alignas(4) std::array<uint8_t, ClassifiedContext::size_bytes> buffer{};
    ClassifiedContext packet(buffer.data());

    packet.set_stream_id(0x87654321);
    set(packet, field::bandwidth, 40'000'000ULL);  // 40 MHz

    std::cout << "Created classified context packet:\n";
    std::cout << "  Size: " << ClassifiedContext::size_bytes << " bytes\n";
    std::cout << "  Has Class ID: Yes (OUI=0x00FF00, PCC=0xABCD1234)\n";
    std::cout << "  Stream ID: 0x" << std::hex << packet.stream_id() << std::dec << "\n";
    std::cout << "  Bandwidth: " << get(packet, field::bandwidth).value() / 1'000'000.0 << " MHz\n";
}

// Example 3: Parsing a context packet received from network
void example_runtime_parsing() {
    std::cout << "\n=== Runtime Context Packet Parsing Example ===\n";

    // Simulate receiving a context packet from network
    alignas(4) std::array<uint8_t, 256> rx_buffer{};

    // Manually construct a context packet (simulating network receipt)
    // In real use, this would come from a socket/file/etc.
    // Packet structure: header(1) + stream_id(1) + cif0(1) + bandwidth(2) + sample_rate(2) + temp(1) = 8 words

    // Header: type=4 (context), has stream ID, size=8 words
    uint32_t header =
        (static_cast<uint32_t>(packet_type::context) << header::PACKET_TYPE_SHIFT) |
        header::STREAM_ID_INDICATOR |
        8;
    cif::write_u32_safe(rx_buffer.data(), 0, header);

    // Stream ID
    cif::write_u32_safe(rx_buffer.data(), 4, 0xCAFEBABE);

    // CIF0: Enable bandwidth, sample rate, temperature
    uint32_t cif0_mask = cif0::BANDWIDTH | cif0::SAMPLE_RATE | cif0::TEMPERATURE;
    cif::write_u32_safe(rx_buffer.data(), 8, cif0_mask);

    // Bandwidth: 100 MHz
    cif::write_u64_safe(rx_buffer.data(), 12, 100'000'000);

    // Sample Rate: 50 MSPS
    cif::write_u64_safe(rx_buffer.data(), 20, 50'000'000);

    // Temperature: 25.5°C (stored as hundredths)
    cif::write_u32_safe(rx_buffer.data(), 28, 0x09FB0000);  // 2555 in upper 16 bits

    // Now parse it with ContextPacketView
    ContextPacketView view(rx_buffer.data(), 8 * 4);

    // Validate the packet
    auto error = view.validate();
    if (error != validation_error::none) {
        std::cout << "Validation failed: " << validation_error_string(error) << "\n";
        return;
    }

    std::cout << "Successfully parsed context packet:\n";

    // Access fields
    if (view.has_stream_id()) {
        std::cout << "  Stream ID: 0x" << std::hex << view.stream_id().value()
                  << std::dec << "\n";
    }

    std::cout << "  CIF0: 0x" << std::hex << view.cif0() << std::dec << "\n";

    if (auto bw = get(view, field::bandwidth)) {
        std::cout << "  Bandwidth: " << *bw / 1'000'000.0 << " MHz\n";
    }

    if (auto sr = get(view, field::sample_rate)) {
        std::cout << "  Sample Rate: " << *sr / 1'000'000.0 << " MSPS\n";
    }

    // Temperature field - use helper method
    std::cout << "  Temperature field present: "
              << (view.has_cif0_field<cif0::TEMPERATURE>() ? "Yes" : "No") << "\n";
}

// Example 4: Handling variable-length fields (GPS ASCII)
void example_variable_fields() {
    std::cout << "\n=== Variable-Length Field Example (GPS ASCII) ===\n";

    alignas(4) std::array<uint8_t, 512> buffer{};

    // Create packet with GPS ASCII field
    const char* nmea_sentence = "$GPGGA,123456.00,3723.456,N,12202.345,W,1,08,0.9,545.4,M,46.9,M,,*47";
    size_t nmea_len = std::strlen(nmea_sentence);
    size_t nmea_words = 1 + (nmea_len + 3) / 4;  // Count word + data words

    // Header
    uint32_t total_words = 1 + 1 + nmea_words;  // header + cif0 + gps_ascii
    uint32_t header =
        (static_cast<uint32_t>(packet_type::context) << header::PACKET_TYPE_SHIFT) |
        total_words;
    cif::write_u32_safe(buffer.data(), 0, header);

    // CIF0 with GPS ASCII bit
    uint32_t cif0_mask = cif0::GPS_ASCII;
    cif::write_u32_safe(buffer.data(), 4, cif0_mask);

    // GPS ASCII: character count + data
    cif::write_u32_safe(buffer.data(), 8, nmea_len);
    std::memcpy(buffer.data() + 12, nmea_sentence, nmea_len);

    // Parse with view
    ContextPacketView view(buffer.data(), total_words * 4);
    auto error = view.validate();
    if (error != validation_error::none) {
        std::cout << "Validation failed: " << validation_error_string(error) << "\n";
        return;
    }

    std::cout << "Successfully parsed packet with GPS ASCII:\n";

    if (view.has_gps_ascii()) {
        auto gps_data = view.gps_ascii_data();
        std::cout << "  GPS ASCII field size: " << gps_data.size() << " bytes\n";

        // Extract character count
        uint32_t char_count;
        std::memcpy(&char_count, gps_data.data(), 4);
        char_count = detail::network_to_host32(char_count);
        std::cout << "  Character count: " << char_count << "\n";

        // Extract and print NMEA sentence
        std::string nmea(reinterpret_cast<const char*>(gps_data.data() + 4), char_count);
        std::cout << "  NMEA sentence: " << nmea << "\n";
    }
}

// Example 5: Demonstrating unsupported field rejection
void example_unsupported_rejection() {
    std::cout << "\n=== Unsupported Field Rejection Example ===\n";

    // Try to parse a packet with unsupported fields
    alignas(4) std::array<uint8_t, 64> buffer{};

    // Header
    uint32_t header =
        (static_cast<uint32_t>(packet_type::context) << header::PACKET_TYPE_SHIFT) |
        3;
    cif::write_u32_safe(buffer.data(), 0, header);

    std::cout << "Testing various unsupported CIF bits:\n";

    // Test 1: Reserved bit
    uint32_t cif0_reserved = (1U << 4);  // Bit 4 is reserved
    cif::write_u32_safe(buffer.data(), 4, cif0_reserved);

    ContextPacketView view1(buffer.data(), 3 * 4);
    auto error1 = view1.validate();
    std::cout << "  Reserved bit 4: "
              << (error1 == validation_error::unsupported_field ? "Correctly rejected" : "ERROR")
              << "\n";

    // Test 2: CIF3 enable bit
    uint32_t cif0_cif3 = (1U << 3);  // Bit 3 enables CIF3 (unsupported)
    cif::write_u32_safe(buffer.data(), 4, cif0_cif3);

    ContextPacketView view2(buffer.data(), 3 * 4);
    auto error2 = view2.validate();
    std::cout << "  CIF3 enable bit: "
              << (error2 == validation_error::unsupported_field ? "Correctly rejected" : "ERROR")
              << "\n";

    // Test 3: Field Attributes enable
    uint32_t cif0_attr = (1U << 7);  // Bit 7 is Field Attributes (unsupported)
    cif::write_u32_safe(buffer.data(), 4, cif0_attr);

    ContextPacketView view3(buffer.data(), 3 * 4);
    auto error3 = view3.validate();
    std::cout << "  Field Attributes bit: "
              << (error3 == validation_error::unsupported_field ? "Correctly rejected" : "ERROR")
              << "\n";
}

int main() {
    std::cout << "VRTIO Context Packet Examples\n";
    std::cout << "=============================\n";

    example_compile_time_context();
    example_with_class_id();
    example_runtime_parsing();
    example_variable_fields();
    example_unsupported_rejection();

    std::cout << "\nAll examples completed successfully!\n";
    return 0;
}