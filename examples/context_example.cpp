// Context Packet Example for VRTIO
//
// Demonstrates creating and parsing VRT context packets using both
// compile-time templates and runtime parsing.

#include <array>
#include <iomanip>
#include <iostream>

#include <cstring>
#include <vrtio.hpp>

using namespace vrtio;
using namespace vrtio::field;

// Example 1: Creating a context packet at compile time
void example_compile_time_context() {
    std::cout << "\n=== Compile-Time Context Packet Example ===\n";

    // Define a packet type with fields
    // Note: Per VITA 49.2 spec:
    //   - Context packets ALWAYS have Stream ID (no template parameter needed)
    //   - Context packets NEVER have Trailer (bit 26 reserved, must be 0)
    using SignalContext =
        ContextPacket<NoTimeStamp, NoClassId, bandwidth, sample_rate, gain, device_id>;

    // Create buffer and packet
    alignas(4) std::array<uint8_t, SignalContext::size_bytes> buffer{};
    SignalContext packet(buffer.data());

    // Set the stream ID
    packet.set_stream_id(0x12345678);

    // Set signal parameters using interpreted values (Hz)
    packet[bandwidth].set_value(20'000'000.0);   // 20 MHz
    packet[sample_rate].set_value(10'000'000.0); // 10 MSPS
    packet[gain].set_raw_value(0x00100000U);     // Gain value (raw format)

    // Device ID is 64-bit (OUI + device code)
    // For now we'll just set a simple value
    // Note: Need to add device ID setter to ContextPacket

    std::cout << "Created context packet:\n";
    std::cout << "  Size: " << SignalContext::size_bytes << " bytes\n";
    std::cout << "  Stream ID: 0x" << std::hex << packet.stream_id() << std::dec << "\n";
    std::cout << "  Bandwidth: " << packet[bandwidth].value() / 1'000'000.0 << " MHz\n";
    std::cout << "  Sample Rate: " << packet[sample_rate].value() / 1'000'000.0 << " MSPS\n";
    std::cout << "  Gain: 0x" << std::hex << packet[gain].raw_value() << std::dec << "\n";
}

// Example 2: Creating a context packet with Class ID
void example_with_class_id() {
    std::cout << "\n=== Context Packet with Class ID Example ===\n";

    // Note: Context packets always have Stream ID per spec
    using ClassifiedContext = ContextPacket<NoTimeStamp, ClassId, bandwidth>;

    alignas(4) std::array<uint8_t, ClassifiedContext::size_bytes> buffer{};
    ClassifiedContext packet(buffer.data());

    packet.set_stream_id(0x87654321);

    // Set Class ID: OUI=0x00FF00, ICC=0xABCD, PCC=0x1234
    ClassIdValue cid(0x00FF00, 0xABCD, 0x1234);
    packet.set_class_id(cid);

    packet[bandwidth].set_value(40'000'000.0); // 40 MHz

    std::cout << "Created classified context packet:\n";
    std::cout << "  Size: " << ClassifiedContext::size_bytes << " bytes\n";
    std::cout << "  Has Class ID: Yes (OUI=0x" << std::hex << cid.oui() << ", ICC=0x" << cid.icc()
              << ", PCC=0x" << cid.pcc() << std::dec << ")\n";
    std::cout << "  Stream ID: 0x" << std::hex << packet.stream_id() << std::dec << "\n";
    std::cout << "  Bandwidth: " << packet[bandwidth].value() / 1'000'000.0 << " MHz\n";
}

// Example 3: Parsing a context packet received from network
void example_runtime_parsing() {
    std::cout << "\n=== Runtime Context Packet Parsing Example ===\n";

    // Simulate receiving a context packet from network
    // In real use, this would come from a socket/file/etc.
    alignas(4) std::array<uint8_t, 256> rx_buffer{};

    // Create a context packet using the compile-time API
    // This simulates what a transmitter would send
    using TxPacket = ContextPacket<NoTimeStamp, NoClassId, bandwidth, sample_rate, temperature>;
    TxPacket tx_packet(rx_buffer.data());

    // Set stream ID and field values
    tx_packet.set_stream_id(0xCAFEBABE);
    tx_packet[bandwidth].set_value(100'000'000.0);  // 100 MHz
    tx_packet[sample_rate].set_value(50'000'000.0); // 50 MSPS
    // Temperature would be set here if we had a setter for it

    // Now parse it with ContextPacketView (simulating receiver)
    ContextPacketView view(rx_buffer.data(), TxPacket::size_bytes);

    // Validate the packet
    auto error = view.error();
    if (error != ValidationError::none) {
        std::cout << "Validation failed: " << validation_error_string(error) << "\n";
        return;
    }

    std::cout << "Successfully parsed context packet:\n";

    // Access fields
    if (auto sid = view.stream_id()) {
        std::cout << "  Stream ID: 0x" << std::hex << sid.value() << std::dec << "\n";
    }

    std::cout << "  CIF0: 0x" << std::hex << view.cif0() << std::dec << "\n";

    if (auto bw = view[bandwidth]) {
        std::cout << "  Bandwidth: " << bw.value() / 1'000'000.0 << " MHz\n";
    }

    if (auto sr = view[sample_rate]) {
        std::cout << "  Sample Rate: " << sr.value() / 1'000'000.0 << " MSPS\n";
    }

    // Temperature field - use operator[] for presence check
    std::cout << "  Temperature field present: " << (view[temperature] ? "Yes" : "No") << "\n";
}

// Example 4: Handling variable-length fields (GPS ASCII)
void example_variable_fields() {
    std::cout << "\n=== Variable-Length Field Example (GPS ASCII) ===\n";

    alignas(4) std::array<uint8_t, 512> buffer{};

    // NOTE: Variable-length fields like GPS ASCII cannot use the compile-time
    // ContextPacket template (which requires fixed sizes). For such fields,
    // you must construct packets manually using the low-level primitives.

    // Create packet with GPS ASCII field
    const char* nmea_sentence =
        "$GPGGA,123456.00,3723.456,N,12202.345,W,1,08,0.9,545.4,M,46.9,M,,*47";
    size_t nmea_len = std::strlen(nmea_sentence);
    size_t nmea_words = 1 + (nmea_len + 3) / 4; // Count word + data words

    // Header
    // Note: Context packets ALWAYS have Stream ID per spec
    uint32_t total_words = 1 + 1 + 1 + nmea_words; // header + stream_id + cif0 + gps_ascii
    uint32_t header =
        (static_cast<uint32_t>(PacketType::Context) << header::packet_type_shift) | total_words;
    cif::write_u32_safe(buffer.data(), 0, header);

    // Stream ID (always present for context packets)
    cif::write_u32_safe(buffer.data(), 4, 0x12345678);

    // CIF0 with GPS ASCII bit (bit 8 of CIF0)
    uint32_t cif0_mask = (1U << 8);
    cif::write_u32_safe(buffer.data(), 8, cif0_mask);

    // GPS ASCII: character count + data (at offset 12 after header + stream_id + cif0)
    cif::write_u32_safe(buffer.data(), 12, nmea_len);
    std::memcpy(buffer.data() + 16, nmea_sentence, nmea_len);

    // Parse with view
    ContextPacketView view(buffer.data(), total_words * 4);
    auto error = view.error();
    if (error != ValidationError::none) {
        std::cout << "Validation failed: " << validation_error_string(error) << "\n";
        return;
    }

    std::cout << "Successfully parsed packet with GPS ASCII:\n";

    // Use operator[] for GPS ASCII field
    if (auto gps_proxy = view[field::gps_ascii]) {
        auto gps_data = gps_proxy.raw_bytes();
        std::cout << "  GPS ASCII field size: " << gps_data.size() << " bytes\n";

        // Extract character count
        uint32_t char_count;
        std::memcpy(&char_count, gps_data.data(), 4);
        char_count = cif::read_u32_safe(gps_data.data(), 0);
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

    // Header (context packets always have stream_id + cif0)
    uint32_t header = (static_cast<uint32_t>(PacketType::Context) << header::packet_type_shift) | 3;
    cif::write_u32_safe(buffer.data(), 0, header);

    // Stream ID (always present)
    cif::write_u32_safe(buffer.data(), 4, 0);

    std::cout << "Testing various unsupported CIF bits:\n";

    // Test 1: Reserved bit
    uint32_t cif0_reserved = (1U << 4); // Bit 4 is reserved
    cif::write_u32_safe(buffer.data(), 8, cif0_reserved);

    ContextPacketView view1(buffer.data(), 3 * 4);
    auto error1 = view1.error();
    std::cout << "  Reserved bit 4: "
              << (error1 == ValidationError::unsupported_field ? "Correctly rejected" : "ERROR")
              << "\n";

    // Test 2: CIF3 enable bit
    uint32_t cif0_cif3 = (1U << 3); // Bit 3 enables CIF3 (unsupported)
    cif::write_u32_safe(buffer.data(), 8, cif0_cif3);

    ContextPacketView view2(buffer.data(), 3 * 4);
    auto error2 = view2.error();
    std::cout << "  CIF3 enable bit: "
              << (error2 == ValidationError::unsupported_field ? "Correctly rejected" : "ERROR")
              << "\n";

    // Test 3: Field Attributes enable
    uint32_t cif0_attr = (1U << 7); // Bit 7 is Field Attributes (unsupported)
    cif::write_u32_safe(buffer.data(), 8, cif0_attr);

    ContextPacketView view3(buffer.data(), 3 * 4);
    auto error3 = view3.error();
    std::cout << "  Field Attributes bit: "
              << (error3 == ValidationError::unsupported_field ? "Correctly rejected" : "ERROR")
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