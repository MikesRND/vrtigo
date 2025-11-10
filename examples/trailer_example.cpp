#include <iomanip>
#include <iostream>
#include <vector>

#include <vrtio/core/trailer_view.hpp>
#include <vrtio/packet/builder.hpp>
#include <vrtio/packet/data_packet.hpp>

// Example demonstrating trailer field usage

int main() {
    // Define a packet type with trailer enabled
    using PacketType = vrtio::SignalDataPacket<vrtio::TimeStampUTC,      // Using UTC timestamps
                                               vrtio::Trailer::Included, // Trailer included
                                               128                       // Payload words
                                               >;

    std::cout << "=== VRTIO Trailer Fields Example ===\n\n";

    // Allocate buffer for packet
    std::vector<uint8_t> buffer(PacketType::size_bytes);

    // Example 1: Build a packet with good status
    std::cout << "Example 1: Building a packet with good status\n";
    std::cout << "----------------------------------------------\n";

    auto good_status = vrtio::TrailerBuilder{}.valid_data(true).calibrated_time(true);

    auto ts1 = vrtio::TimeStampUTC::fromComponents(1000000, 0);
    auto packet = vrtio::PacketBuilder<PacketType>(buffer.data())
                      .stream_id(0x12345678)
                      .timestamp(ts1)
                      .trailer(good_status) // Apply reusable trailer configuration
                      .packet_count(0)
                      .build();

    std::cout << "Stream ID: 0x" << std::hex << std::setw(8) << std::setfill('0')
              << packet.stream_id() << std::dec << "\n";
    std::cout << "Timestamp: " << packet.getTimeStamp().seconds() << "\n";
    std::cout << "Trailer raw value: 0x" << std::hex << std::setw(8) << std::setfill('0')
              << packet.trailer().raw() << std::dec << "\n";
    std::cout << "Valid data: " << (packet.trailer().valid_data() ? "Yes" : "No") << "\n";
    std::cout << "Calibrated time: " << (packet.trailer().calibrated_time() ? "Yes" : "No") << "\n";
    std::cout << "Has errors: " << (packet.trailer().has_errors() ? "Yes" : "No") << "\n";
    std::cout << "\n";

    // Example 2: Set individual trailer fields
    std::cout << "Example 2: Setting individual trailer fields\n";
    std::cout << "----------------------------------------------\n";

    auto inline_trailer =
        vrtio::TrailerBuilder{}.clear().reference_lock(true).context_packets(5).valid_data(true);

    auto ts2 = vrtio::TimeStampUTC::fromComponents(1500000, 0);
    auto inline_packet = vrtio::PacketBuilder<PacketType>(buffer.data())
                             .stream_id(0x12345678)
                             .timestamp(ts2)
                             .trailer(inline_trailer)
                             .build();

    std::cout << "Reference locked: " << (inline_packet.trailer().reference_lock() ? "Yes" : "No")
              << "\n";
    std::cout << "Context packets: " << (int)inline_packet.trailer().context_packets() << "\n";
    std::cout << "Trailer raw value: 0x" << std::hex << std::setw(8) << std::setfill('0')
              << inline_packet.trailer().raw() << std::dec << "\n";
    std::cout << "\n";

    // Example 3: Build a packet with error conditions
    std::cout << "Example 3: Indicating error conditions\n";
    std::cout << "---------------------------------------\n";

    auto ts3 = vrtio::TimeStampUTC::fromComponents(2000000, 0);
    auto error_packet = vrtio::PacketBuilder<PacketType>(buffer.data())
                            .stream_id(0xAABBCCDD)
                            .timestamp(ts3)
                            .trailer_valid_data(false) // Data not valid
                            .trailer_over_range(true)  // Over-range error
                            .trailer_sample_loss(true) // Sample loss error
                            .packet_count(1)
                            .build();

    std::cout << "Stream ID: 0x" << std::hex << std::setw(8) << std::setfill('0')
              << error_packet.stream_id() << std::dec << "\n";
    std::cout << "Valid data: " << (error_packet.trailer().valid_data() ? "Yes" : "No") << "\n";
    std::cout << "Over-range: " << (error_packet.trailer().over_range() ? "Yes" : "No") << "\n";
    std::cout << "Sample loss: " << (error_packet.trailer().sample_loss() ? "Yes" : "No") << "\n";
    std::cout << "Has errors: " << (error_packet.trailer().has_errors() ? "Yes" : "No") << "\n";
    std::cout << "Trailer raw value: 0x" << std::hex << std::setw(8) << std::setfill('0')
              << error_packet.trailer().raw() << std::dec << "\n";
    std::cout << "\n";

    // Example 4: Bulk status configuration
    std::cout << "Example 4: Bulk status configuration\n";
    std::cout << "-------------------------------------\n";

    auto status_trailer = vrtio::TrailerBuilder{}
                              .valid_data(true)
                              .calibrated_time(true)
                              .reference_lock(true)
                              .context_packets(10);

    auto ts4 = vrtio::TimeStampUTC::fromComponents(3000000, 0);
    auto status_packet = vrtio::PacketBuilder<PacketType>(buffer.data())
                             .stream_id(0x11111111)
                             .timestamp(ts4)
                             .trailer(status_trailer)
                             .packet_count(2)
                             .build();

    std::cout << "Stream ID: 0x" << std::hex << std::setw(8) << std::setfill('0')
              << status_packet.stream_id() << std::dec << "\n";
    std::cout << "Valid data: " << (status_packet.trailer().valid_data() ? "Yes" : "No") << "\n";
    std::cout << "Calibrated time: " << (status_packet.trailer().calibrated_time() ? "Yes" : "No")
              << "\n";
    std::cout << "Reference locked: " << (status_packet.trailer().reference_lock() ? "Yes" : "No")
              << "\n";
    std::cout << "Context packets: " << (int)status_packet.trailer().context_packets() << "\n";
    std::cout << "Has errors: " << (status_packet.trailer().has_errors() ? "Yes" : "No") << "\n";
    std::cout << "\n";

    // Example 5: Processing received packets
    std::cout << "Example 5: Processing received packets\n";
    std::cout << "---------------------------------------\n";

    // Simulate receiving a packet
    auto ts5 = vrtio::TimeStampUTC::fromComponents(4000000, 0);
    auto rx_packet = vrtio::PacketBuilder<PacketType>(buffer.data())
                         .stream_id(0xFEDCBA98)
                         .timestamp(ts5)
                         .trailer_valid_data(true)
                         .trailer_calibrated_time(true)
                         .trailer_signal_detected(true)
                         .trailer_context_packets(3)
                         .packet_count(3)
                         .build();

    // Process the packet
    std::cout << "Received packet from stream 0x" << std::hex << std::setw(8) << std::setfill('0')
              << rx_packet.stream_id() << std::dec << "\n";

    if (rx_packet.trailer().valid_data()) {
        std::cout << "✓ Data is valid\n";
    } else {
        std::cout << "✗ Data is invalid\n";
    }

    if (rx_packet.trailer().calibrated_time()) {
        std::cout << "✓ Timestamp is calibrated\n";
    } else {
        std::cout << "✗ Timestamp is not calibrated\n";
    }

    if (rx_packet.trailer().signal_detected()) {
        std::cout << "✓ Signal detected\n";
    }

    if (rx_packet.trailer().has_errors()) {
        std::cout << "✗ ERROR: Packet has error conditions\n";
        if (rx_packet.trailer().over_range()) {
            std::cout << "  - Over-range detected\n";
        }
        if (rx_packet.trailer().sample_loss()) {
            std::cout << "  - Sample loss detected\n";
        }
    } else {
        std::cout << "✓ No errors detected\n";
    }

    uint8_t ctx_count = rx_packet.trailer().context_packets();
    if (ctx_count > 0) {
        std::cout << "Note: " << (int)ctx_count << " context packet(s) associated\n";
    }

    std::cout << "\n=== All examples completed successfully ===\n";

    return 0;
}
