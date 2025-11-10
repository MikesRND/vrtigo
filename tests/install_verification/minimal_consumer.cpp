// Minimal consumer to verify vrtio install/package works correctly
// This is built out-of-tree using find_package(vrtio) to ensure
// downstream consumers can use the library as a header-only dependency

#include <array>
#include <iostream>

#include <vrtio.hpp>
#include <vrtio/version.hpp>

int main() {
    std::cout << "VRTIO Install Verification\n";
    std::cout << "===========================\n";
    std::cout << "vrtio version: " << vrtio::version_string << "\n\n";

    // Create a minimal packet to verify headers compile and link correctly
    using MinimalPacket = vrtio::SignalDataPacketNoId<vrtio::NoTimeStamp, vrtio::Trailer::None,
                                                      64 // 256 bytes payload
                                                      >;

    std::cout << "Creating minimal packet...\n";
    std::cout << "  Packet size: " << MinimalPacket::size_bytes << " bytes\n";
    std::cout << "  Payload size: " << MinimalPacket::payload_size_bytes << " bytes\n";

    // Allocate buffer and create packet
    alignas(4) std::array<uint8_t, MinimalPacket::size_bytes> buffer;
    MinimalPacket packet(buffer.data());

    // Set and verify a field
    packet.set_packet_count(5);
    if (packet.packet_count() != 5) {
        std::cerr << "ERROR: Packet count mismatch!\n";
        return 1;
    }

    std::cout << "  Packet count: " << static_cast<int>(packet.packet_count()) << "\n";
    std::cout << "\nInstall verification PASSED!\n";
    std::cout << "vrtio headers are correctly installed and functional.\n";

    return 0;
}
