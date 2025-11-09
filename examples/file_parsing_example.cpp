#include <vrtio/vrtio_io.hpp>
#include <iostream>
#include <iomanip>

using namespace vrtio;

/**
 * @brief Packet processor functor with internal state
 *
 * Processes VRT packets and maintains an internal counter for numbering.
 * No external variable capture needed - all state is encapsulated.
 */
class PacketProcessor {
private:
    size_t packet_num_ = 0;  // Internal packet counter

public:
    /**
     * @brief Process a single packet
     *
     * @param packet Raw packet data (unused in this example)
     * @param info Packet metadata (type, size, offset, header)
     * @return true to continue processing, false to stop
     */
    bool operator()(std::span<const uint8_t> /*packet*/,
                    const VRTFileReader<>::ReadResult& info) {
        packet_num_++;

        std::cout << "Packet " << packet_num_ << ":\n";
        std::cout << "  Type: " << static_cast<int>(info.type) << "\n";
        std::cout << "  Size: " << info.packet_size_bytes << " bytes\n";
        std::cout << "  Offset: 0x" << std::hex << info.file_offset << std::dec << "\n";

        // Decode header for more details
        auto decoded = detail::decode_header(info.header);
        std::cout << "  Stream ID: " << (detail::has_stream_id_field(decoded.type) ? "Yes" : "No") << "\n";
        std::cout << "  Class ID: " << (decoded.has_class_id ? "Yes" : "No") << "\n";
        // Trailer only applies to Signal/Extension Data packets
        std::cout << "  Trailer: " << (decoded.trailer_included ? "Yes" : "No") << "\n";
        std::cout << "  TSI: " << static_cast<int>(decoded.tsi) << "\n";
        std::cout << "  TSF: " << static_cast<int>(decoded.tsf) << "\n";
        std::cout << "  Count: " << static_cast<int>(decoded.packet_count) << "\n";
        std::cout << "\n";

        // Process up to 30 packets (could also return true to process all)
        return packet_num_ < 30; 
    }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <vrt_file>\n";
        return 1;
    }

    const char* filepath = argv[1];

    try {
        // Open VRT file
        VRTFileReader reader(filepath);

        std::cout << "File: " << filepath << "\n";
        std::cout << "Size: " << reader.size() << " bytes\n\n";

        // Parse all packets using functor with internal state
        PacketProcessor processor;
        reader.for_each_packet(processor);

        std::cout << "Total packets read: " << reader.packets_read() << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
