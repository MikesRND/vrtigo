// Basic usage example for VRTIGO

#include <array>
#include <iostream>

#include <vrtigo.hpp>

int main() {
    std::cout << "VRTIGO - Basic Usage Example\n";
    std::cout << "===================================\n\n";

    // Example 1: Creating a signal packet with builder
    {
        std::cout << "Example 1: Creating a signal packet\n";

        using Packet = vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::TimeStampUTC,
                                               vrtigo::Trailer::none, 128>;
        alignas(4) std::array<uint8_t, Packet::size_bytes> buffer;

        // Prepare payload data
        std::array<uint8_t, 512> payload_data{};
        for (size_t i = 0; i < payload_data.size(); ++i) {
            payload_data[i] = static_cast<uint8_t>(i);
        }

        // Use builder pattern for convenient construction
        auto ts = vrtigo::TimeStampUTC::now();
        auto packet = vrtigo::PacketBuilder<Packet>(buffer.data())
                          .stream_id(0x12345678)
                          .timestamp(ts)
                          .packet_count(1)
                          .payload(payload_data.data(), payload_data.size())
                          .build();

        std::cout << "  Stream ID: 0x" << std::hex << packet.stream_id() << std::dec << "\n";
        std::cout << "  Timestamp: " << packet.timestamp().seconds() << "s\n";
        std::cout << "  Payload: " << packet.payload().size() << " bytes\n\n";
    }

    // Example 2: Parsing data (requires validation)
    {
        std::cout << "Example 2: Parsing data\n";

        using Packet =
            vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::NoTimeStamp, vrtigo::Trailer::none, 64>;
        alignas(4) std::array<uint8_t, Packet::size_bytes> buffer;

        // Create test data
        Packet(buffer.data()).set_stream_id(0xABCDEF00);

        // Parse without initialization
        Packet received(buffer.data(), false);

        // MUST validate before accessing fields
        auto result = received.validate(buffer.size());
        if (result != vrtigo::ValidationError::none) {
            std::cerr << "  Validation failed: " << vrtigo::validation_error_string(result) << "\n";
            return 1;
        }

        std::cout << "  Validation: PASSED\n";
        std::cout << "  Stream ID: 0x" << std::hex << received.stream_id() << std::dec << "\n\n";
    }

    std::cout << "All examples completed!\n";
    return 0;
}
