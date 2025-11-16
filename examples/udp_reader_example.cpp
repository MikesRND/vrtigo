#include <atomic>
#include <iomanip>
#include <iostream>

#include <csignal>
#include <vrtio/utils/netio/udp_vrt_reader.hpp>

using namespace vrtio;
using namespace vrtio::utils::netio;

// Global flag for graceful shutdown
std::atomic<bool> keep_running{true};

void signal_handler(int) {
    keep_running.store(false);
}

/**
 * @brief UDP packet processor with internal state
 *
 * Processes VRT packets received via UDP and maintains statistics.
 */
class UDPPacketProcessor {
private:
    size_t packet_count_ = 0;
    size_t data_packet_count_ = 0;
    size_t context_packet_count_ = 0;
    size_t invalid_packet_count_ = 0;

public:
    /**
     * @brief Process a single packet
     *
     * @param pkt Type-safe packet variant (DataPacketView, ContextPacketView, or InvalidPacket)
     * @return true to continue processing, false to stop
     */
    bool operator()(const vrtio::PacketVariant& pkt) {
        packet_count_++;

        // Check global flag for graceful shutdown
        if (!keep_running.load()) {
            return false;
        }

        std::cout << "\n=== Packet " << packet_count_ << " ===\n";

        // Process based on packet type
        if (vrtio::is_data_packet(pkt)) {
            const auto& view = std::get<DataPacketView>(pkt);
            data_packet_count_++;

            std::cout << "Type: Data Packet\n";
            std::cout << "  Packet Type: " << static_cast<int>(view.type()) << "\n";
            std::cout << "  Stream ID: " << (view.has_stream_id() ? "Yes" : "No");
            if (view.has_stream_id()) {
                auto sid = view.stream_id();
                if (sid) {
                    std::cout << " (0x" << std::hex << *sid << std::dec << ")";
                }
            }
            std::cout << "\n";
            std::cout << "  Class ID: " << (view.has_class_id() ? "Yes" : "No") << "\n";
            std::cout << "  Trailer: " << (view.has_trailer() ? "Yes" : "No") << "\n";
            std::cout << "  Packet Count: " << static_cast<int>(view.packet_count()) << "\n";
            std::cout << "  Payload Size: " << view.payload().size() << " bytes\n";

        } else if (vrtio::is_context_packet(pkt)) {
            const auto& view = std::get<ContextPacketView>(pkt);
            context_packet_count_++;

            std::cout << "Type: Context Packet\n";

            auto stream_id = view.stream_id();
            std::cout << "  Stream ID: " << (stream_id.has_value() ? "Yes" : "No");
            if (stream_id) {
                std::cout << " (0x" << std::hex << *stream_id << std::dec << ")";
            }
            std::cout << "\n";

            auto class_id = view.class_id();
            std::cout << "  Class ID: " << (class_id.has_value() ? "Yes" : "No") << "\n";

            // Context packets contain various fields that can be queried

        } else {
            // Invalid packet
            const auto& invalid = std::get<vrtio::InvalidPacket>(pkt);
            invalid_packet_count_++;

            std::cout << "Type: INVALID PACKET\n";
            std::cout << "  Error: " << static_cast<int>(invalid.error) << " ("
                      << validation_error_string(invalid.error) << ")\n";
            std::cout << "  Attempted Type: " << static_cast<int>(invalid.attempted_type) << "\n";
        }

        return true; // Continue processing
    }

    void print_summary() const {
        std::cout << "\n========== Summary ==========\n";
        std::cout << "Total packets received: " << packet_count_ << "\n";
        std::cout << "  Data packets: " << data_packet_count_ << "\n";
        std::cout << "  Context packets: " << context_packet_count_ << "\n";
        std::cout << "  Invalid packets: " << invalid_packet_count_ << "\n";
    }

    size_t count() const { return packet_count_; }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <udp_port> [max_packets]\n";
        std::cerr << "\n";
        std::cerr << "Example: " << argv[0] << " 12345 100\n";
        std::cerr << "  Listens on UDP port 12345 and processes up to 100 packets\n";
        std::cerr << "  (Press Ctrl+C to stop early)\n";
        return 1;
    }

    // Parse command line arguments
    uint16_t port = static_cast<uint16_t>(std::atoi(argv[1]));
    size_t max_packets = (argc >= 3) ? std::atoi(argv[2]) : SIZE_MAX;

    // Set up signal handler for graceful shutdown
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << "UDP VRT Reader Example\n";
    std::cout << "======================\n";
    std::cout << "Listening on UDP port: " << port << "\n";
    if (max_packets != SIZE_MAX) {
        std::cout << "Max packets: " << max_packets << "\n";
    }
    std::cout << "Press Ctrl+C to stop\n";
    std::cout << "\nWaiting for packets...\n";

    try {
        // Create UDP reader
        UDPVRTReader<> reader(port);

        // Optional: Set timeout to allow checking keep_running flag
        reader.try_set_timeout(std::chrono::seconds(1));

        // Process packets using visitor pattern
        UDPPacketProcessor processor;

        size_t count = 0;
        while (keep_running.load() && count < max_packets) {
            auto pkt = reader.read_next_packet();

            if (!pkt) {
                // Check transport status to distinguish errors
                const auto& status = reader.transport_status();

                if (status.is_truncated()) {
                    std::cerr << "\nERROR: Datagram truncated!\n";
                    std::cerr << "  Received: " << status.bytes_received << " bytes\n";
                    std::cerr << "  Actual size: " << status.actual_size << " bytes\n";
                    std::cerr << "  Increase MaxPacketWords template parameter to "
                              << ((status.actual_size + 3) / 4) << " or larger\n";
                    break;
                }

                if (status.is_terminal()) {
                    std::cerr << "\nSocket closed or error (errno: " << status.errno_value << ")\n";
                    break;
                }

                // Timeout - continue waiting
                continue;
            }

            // Process the packet
            if (!processor(*pkt)) {
                break;
            }
            count++;
        }

        // Print summary
        processor.print_summary();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\nExiting...\n";
    return 0;
}
