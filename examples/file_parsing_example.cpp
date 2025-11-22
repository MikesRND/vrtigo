#include <iomanip>
#include <iostream>

#include <vrtigo/vrtigo_io.hpp>

using namespace vrtigo;

/**
 * @brief Packet processor functor with internal state
 *
 * Processes VRT packets and maintains an internal counter for numbering.
 * No external variable capture needed - all state is encapsulated.
 */
class PacketProcessor {
private:
    size_t packet_num_ = 0; // Internal packet counter

public:
    /**
     * @brief Process a single packet
     *
     * @param pkt Type-safe packet variant (RuntimeDataPacket or RuntimeContextPacket)
     * @return true to continue processing, false to stop
     */
    bool operator()(const PacketVariant& pkt) {
        packet_num_++;

        std::cout << "Packet " << packet_num_ << ":\n";
        std::cout << "  Type: " << static_cast<int>(packet_type(pkt)) << "\n";

        // Use appropriate packet view based on packet type
        if (is_data_packet(pkt)) {
            const auto& view = std::get<RuntimeDataPacket>(pkt);
            std::cout << "  Stream ID: " << (view.has_stream_id() ? "Yes" : "No") << "\n";
            std::cout << "  Class ID: " << (view.has_class_id() ? "Yes" : "No") << "\n";
            std::cout << "  Trailer: " << (view.has_trailer() ? "Yes" : "No") << "\n";
            std::cout << "  TSI: " << static_cast<int>(view.tsi_kind()) << "\n";
            std::cout << "  TSF: " << static_cast<int>(view.tsf_kind()) << "\n";
            std::cout << "  Count: " << static_cast<int>(view.packet_count()) << "\n";
        } else if (is_context_packet(pkt)) {
            const auto& view = std::get<RuntimeContextPacket>(pkt);
            std::cout << "  Stream ID: " << (view.stream_id().has_value() ? "Yes" : "No") << "\n";
            std::cout << "  Class ID: " << (view.class_id().has_value() ? "Yes" : "No") << "\n";
            std::cout << "  Trailer: No\n"; // Context packets don't have trailers
            // Context packets don't have TSI/TSF/Count in the same way as data packets
            std::cout << "  TSI: N/A\n";
            std::cout << "  TSF: N/A\n";
            std::cout << "  Count: N/A\n";
        } else {
            // Invalid packet
            std::cout << "  (Invalid packet)\n";
        }
        std::cout << "\n";

        // Process up to 30 packets (could also return true to process all)
        return packet_num_ < 30;
    }

    size_t count() const { return packet_num_; }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <vrt_file>\n";
        return 1;
    }

    const char* filepath = argv[1];

    // Open VRT file
    VRTFileReader<> reader(filepath);

    if (!reader.is_open()) {
        std::cerr << "Failed to open file: " << filepath << "\n";
        return 1;
    }

    std::cout << "File: " << filepath << "\n";
    std::cout << "Size: " << reader.size() << " bytes\n\n";

    // Parse all packets using functor with internal state
    PacketProcessor processor;
    reader.for_each_validated_packet(processor);

    std::cout << "Total packets read: " << processor.count() << "\n";

    return 0;
}
