// Basic usage example for VRTIO

#include <array>
#include <iostream>

#include <ctime>
#include <vrtio.hpp>

int main() {
    std::cout << "VRTIO - Basic Usage Example\n";
    std::cout << "===================================\n\n";

    // Example 1: Creating a signal packet with builder
    {
        std::cout << "Example 1: Creating a signal packet\n";

        using Packet = vrtio::SignalDataPacket<vrtio::TimeStampUTC, vrtio::Trailer::None, 128>;
        alignas(4) std::array<uint8_t, Packet::size_bytes> buffer;

        // Prepare payload data
        std::array<uint8_t, 512> payload_data{};
        for (size_t i = 0; i < payload_data.size(); ++i) {
            payload_data[i] = static_cast<uint8_t>(i);
        }

        // Use builder pattern for convenient construction
        auto ts = vrtio::TimeStampUTC::fromComponents(static_cast<uint32_t>(std::time(nullptr)), 0);
        auto packet = vrtio::PacketBuilder<Packet>(buffer.data())
                          .stream_id(0x12345678)
                          .timestamp(ts)
                          .packet_count(1)
                          .payload(payload_data.data(), payload_data.size())
                          .build();

        std::cout << "  Stream ID: 0x" << std::hex << packet.stream_id() << std::dec << "\n";
        std::cout << "  Timestamp: " << packet.getTimeStamp().seconds() << "s\n";
        std::cout << "  Payload: " << packet.payload().size() << " bytes\n\n";
    }

    // Example 2: Parsing data (requires validation)
    {
        std::cout << "Example 2: Parsing data\n";

        using Packet = vrtio::SignalDataPacket<vrtio::NoTimeStamp, vrtio::Trailer::None, 64>;
        alignas(4) std::array<uint8_t, Packet::size_bytes> buffer;

        // Create test data
        Packet(buffer.data()).set_stream_id(0xABCDEF00);

        // Parse without initialization
        Packet received(buffer.data(), false);

        // MUST validate before accessing fields
        auto result = received.validate(buffer.size());
        if (result != vrtio::ValidationError::none) {
            std::cerr << "  Validation failed: " << vrtio::validation_error_string(result) << "\n";
            return 1;
        }

        std::cout << "  Validation: PASSED\n";
        std::cout << "  Stream ID: 0x" << std::hex << received.stream_id() << std::dec << "\n\n";
    }

    std::cout << "All examples completed!\n";
    return 0;
}
