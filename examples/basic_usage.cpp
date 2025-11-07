// Basic usage example for VRTIO Phase 1

#include <vrtio.hpp>
#include <iostream>
#include <iomanip>
#include <array>
#include <ctime>

int main() {
    std::cout << "VRTIO Phase 1 - Basic Usage Example\n";
    std::cout << "===================================\n\n";

    // Example 1: Simple packet without stream ID
    {
        std::cout << "Example 1: Signal packet without stream ID\n";

        using SimplePacket = vrtio::signal_packet<
            vrtio::packet_type::signal_data_no_stream,
            vrtio::tsi_type::utc,
            vrtio::tsf_type::none,
            false,  // No trailer
            256     // 1024 bytes payload
        >;

        std::cout << "  Packet size: " << SimplePacket::size_bytes << " bytes\n";
        std::cout << "  Payload size: " << SimplePacket::payload_size_bytes << " bytes\n";

        // User provides buffer
        alignas(4) std::array<uint8_t, SimplePacket::size_bytes> buffer;

        // Create packet view
        SimplePacket packet(buffer.data());
        packet.set_timestamp_integer(static_cast<uint32_t>(std::time(nullptr)));
        packet.set_packet_count(1);

        // Fill payload with test data
        auto payload = packet.payload();
        for (size_t i = 0; i < payload.size(); ++i) {
            payload[i] = static_cast<uint8_t>(i & 0xFF);
        }

        std::cout << "  Timestamp: " << packet.timestamp_integer() << "\n";
        std::cout << "  Packet count: " << static_cast<int>(packet.packet_count()) << "\n";
        std::cout << "\n";
    }

    // Example 2: Packet with stream ID using builder
    {
        std::cout << "Example 2: Signal packet with stream ID (builder pattern)\n";

        using StreamPacket = vrtio::signal_packet<
            vrtio::packet_type::signal_data_with_stream,
            vrtio::tsi_type::gps,
            vrtio::tsf_type::real_time,
            true,   // Has trailer
            512     // 2048 bytes payload
        >;

        std::cout << "  Packet size: " << StreamPacket::size_bytes << " bytes\n";
        std::cout << "  Has stream ID: " << (StreamPacket::has_stream_id ? "yes" : "no") << "\n";

        // User provides buffer
        alignas(4) std::array<uint8_t, StreamPacket::size_bytes> tx_buffer;

        // Use builder pattern
        std::array<uint8_t, 2048> sensor_data{};

        auto builder = vrtio::packet_builder<StreamPacket>(tx_buffer.data());
        auto packet = builder
            .stream_id(0x12345678)
            .timestamp_integer(1699000000)
            .timestamp_fractional(500000)  // 500ns in picoseconds
            .trailer(0x00000001)
            .packet_count(42)
            .payload(sensor_data.data(), sensor_data.size())
            .build();

        std::cout << "  Stream ID: 0x" << std::hex << packet.stream_id() << std::dec << "\n";
        std::cout << "  Timestamp (integer): " << packet.timestamp_integer() << "\n";
        std::cout << "  Timestamp (fractional): " << packet.timestamp_fractional() << " ps\n";
        std::cout << "  Trailer: 0x" << std::hex << packet.trailer() << std::dec << "\n";
        std::cout << "  Packet count: " << static_cast<int>(packet.packet_count()) << "\n";
        std::cout << "\n";
    }

    // Example 3: Parsing untrusted data (CRITICAL: requires validation!)
    {
        std::cout << "Example 3: Parsing untrusted packet data\n";

        using RxPacket = vrtio::signal_packet<
            vrtio::packet_type::signal_data_with_stream,
            vrtio::tsi_type::utc,
            vrtio::tsf_type::none,
            false,
            256
        >;

        // Simulate received data from network/file/etc.
        alignas(4) std::array<uint8_t, RxPacket::size_bytes> rx_buffer;

        // Build a packet to simulate received data
        vrtio::packet_builder<RxPacket>(rx_buffer.data())
            .stream_id(0xABCDEF00)
            .timestamp_integer(1234567890)
            .packet_count(99)
            .build();

        // SAFE parsing pattern: Parse WITHOUT init, then validate
        RxPacket received(rx_buffer.data(), false);  // init=false for parsing

        // CRITICAL: MUST validate before accessing any fields!
        auto validation_result = received.validate(rx_buffer.size());
        if (validation_result != vrtio::validation_error::none) {
            std::cerr << "  ERROR: Packet validation failed: "
                      << vrtio::validation_error_string(validation_result) << "\n";
            return 1;
        }

        // Now safe to access fields after successful validation
        std::cout << "  Validation: PASSED\n";
        std::cout << "  Received stream ID: 0x" << std::hex << received.stream_id() << std::dec << "\n";
        std::cout << "  Received timestamp: " << received.timestamp_integer() << "\n";
        std::cout << "  Received count: " << static_cast<int>(received.packet_count()) << "\n";
        std::cout << "  Packet size: " << received.packet_size_words() << " words\n";
        std::cout << "\n";
    }

    // Example 4: Compile-time validation
    {
        std::cout << "Example 4: Compile-time type safety\n";

        using NoStreamPacket = vrtio::signal_packet<
            vrtio::packet_type::signal_data_no_stream,
            vrtio::tsi_type::none,
            vrtio::tsf_type::none,
            false,
            128
        >;

        alignas(4) std::array<uint8_t, NoStreamPacket::size_bytes> buffer;
        NoStreamPacket packet(buffer.data());

        // This works:
        auto payload = packet.payload();
        std::cout << "  Payload size: " << payload.size() << " bytes\n";

        // These would cause compile errors (uncomment to test):
        // packet.set_stream_id(0x123);       // Error: no stream ID for type 0
        // packet.set_timestamp_integer(123); // Error: TSI is none
        // packet.set_trailer(0);             // Error: no trailer

        std::cout << "  Type safety verified at compile time!\n";
        std::cout << "\n";
    }

    std::cout << "All examples completed successfully!\n";
    return 0;
}
