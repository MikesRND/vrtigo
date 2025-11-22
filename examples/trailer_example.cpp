#include <iomanip>
#include <iostream>
#include <vector>

#include <vrtigo.hpp>
#include <vrtigo/detail/trailer_view.hpp>

// Example demonstrating trailer field usage

int main() {
    // Define a packet type with trailer enabled
    using PacketType = vrtigo::SignalDataPacket<vrtigo::NoClassId,
                                                vrtigo::UtcRealTimestamp,  // Using UTC timestamps
                                                vrtigo::Trailer::included, // Trailer included
                                                128                        // Payload words
                                                >;

    std::cout << "=== VRTIGO Trailer Fields Example ===\n\n";

    // Allocate buffer for packet
    std::vector<uint8_t> buffer(PacketType::size_bytes);

    // Example 1: Build a packet with good status
    std::cout << "Example 1: Building a packet with good status\n";
    std::cout << "----------------------------------------------\n";

    auto good_status = vrtigo::TrailerBuilder{}.valid_data(true).calibrated_time(true);

    auto ts1 = vrtigo::UtcRealTimestamp(1000000, 0);
    auto packet = vrtigo::PacketBuilder<PacketType>(buffer.data())
                      .stream_id(0x12345678)
                      .timestamp(ts1)
                      .trailer(good_status) // Apply reusable trailer configuration
                      .packet_count(0)
                      .build();

    std::cout << "Stream ID: 0x" << std::hex << std::setw(8) << std::setfill('0')
              << packet.stream_id() << std::dec << "\n";
    std::cout << "Timestamp: " << packet.timestamp().tsi() << "\n";
    std::cout << "Trailer raw value: 0x" << std::hex << std::setw(8) << std::setfill('0')
              << packet.trailer().raw() << std::dec << "\n";
    std::cout << "Valid data: " << (packet.trailer().valid_data() ? "Yes" : "No") << "\n";
    std::cout << "Calibrated time: " << (packet.trailer().calibrated_time() ? "Yes" : "No") << "\n";
    std::cout << "\n";

    // Example 2: Set individual trailer fields
    std::cout << "Example 2: Setting individual trailer fields\n";
    std::cout << "----------------------------------------------\n";

    auto inline_trailer =
        vrtigo::TrailerBuilder{}.clear().reference_lock(true).context_packet_count(5).valid_data(
            true);

    auto ts2 = vrtigo::UtcRealTimestamp(1500000, 0);
    auto inline_packet = vrtigo::PacketBuilder<PacketType>(buffer.data())
                             .stream_id(0x12345678)
                             .timestamp(ts2)
                             .trailer(inline_trailer)
                             .build();

    std::cout << "Reference locked: " << (inline_packet.trailer().reference_lock() ? "Yes" : "No")
              << "\n";
    std::cout << "Context packets: " << inline_packet.trailer().context_packet_count().value_or(0)
              << "\n";
    std::cout << "Trailer raw value: 0x" << std::hex << std::setw(8) << std::setfill('0')
              << inline_packet.trailer().raw() << std::dec << "\n";
    std::cout << "\n";

    // Example 3: Build a packet with error conditions
    std::cout << "Example 3: Indicating error conditions\n";
    std::cout << "---------------------------------------\n";

    auto ts3 = vrtigo::UtcRealTimestamp(2000000, 0);
    auto error_packet = vrtigo::PacketBuilder<PacketType>(buffer.data())
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
    std::cout << "Trailer raw value: 0x" << std::hex << std::setw(8) << std::setfill('0')
              << error_packet.trailer().raw() << std::dec << "\n";
    std::cout << "\n";

    // Example 4: Bulk status configuration
    std::cout << "Example 4: Bulk status configuration\n";
    std::cout << "-------------------------------------\n";

    auto status_trailer = vrtigo::TrailerBuilder{}
                              .valid_data(true)
                              .calibrated_time(true)
                              .reference_lock(true)
                              .context_packet_count(10);

    auto ts4 = vrtigo::UtcRealTimestamp(3000000, 0);
    auto status_packet = vrtigo::PacketBuilder<PacketType>(buffer.data())
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
    std::cout << "Context packets: " << status_packet.trailer().context_packet_count().value_or(0)
              << "\n";
    std::cout << "\n";

    // Example 5: Processing received packets
    std::cout << "Example 5: Processing received packets\n";
    std::cout << "---------------------------------------\n";

    // Simulate receiving a packet
    auto ts5 = vrtigo::UtcRealTimestamp(4000000, 0);
    auto rx_packet = vrtigo::PacketBuilder<PacketType>(buffer.data())
                         .stream_id(0xFEDCBA98)
                         .timestamp(ts5)
                         .trailer_valid_data(true)
                         .trailer_calibrated_time(true)
                         .trailer_detected_signal(true)
                         .trailer_context_packet_count(3)
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

    if (rx_packet.trailer().detected_signal()) {
        std::cout << "✓ Signal detected\n";
    }

    // Check for errors
    bool has_over_range = rx_packet.trailer().over_range().value_or(false);
    bool has_sample_loss = rx_packet.trailer().sample_loss().value_or(false);

    if (has_over_range || has_sample_loss) {
        std::cout << "✗ ERROR: Packet has error conditions\n";
        if (has_over_range) {
            std::cout << "  - Over-range detected\n";
        }
        if (has_sample_loss) {
            std::cout << "  - Sample loss detected\n";
        }
    } else {
        std::cout << "✓ No errors detected\n";
    }

    auto ctx_count_opt = rx_packet.trailer().context_packet_count();
    if (ctx_count_opt.has_value() && *ctx_count_opt > 0) {
        std::cout << "Note: " << *ctx_count_opt << " context packet(s) associated\n";
    }

    std::cout << "\n=== All examples completed successfully ===\n";

    return 0;
}
