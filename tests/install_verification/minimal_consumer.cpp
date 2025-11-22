// Minimal consumer to verify vrtigo install/package works correctly
// This is built out-of-tree using find_package(vrtigo) to ensure
// downstream consumers can use the library as a header-only dependency

#include <array>
#include <iostream>

#include <vrtigo.hpp>
#include <vrtigo/version.hpp>

int main() {
    std::cout << "VRTIGO Install Verification\n";
    std::cout << "===========================\n";
    std::cout << "vrtigo version: " << vrtigo::version_string << "\n\n";

    // Create a minimal packet to verify headers compile and link correctly
    using MinimalPacket =
        vrtigo::SignalDataPacketNoId<vrtigo::NoClassId, vrtigo::NoTimestamp, vrtigo::Trailer::none,
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
    std::cout << "vrtigo headers are correctly installed and functional.\n";

    return 0;
}
