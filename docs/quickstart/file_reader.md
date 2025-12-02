# Reading VRT Files

*Auto-generated from `tests/quickstart/file_reader_test.cpp`. All examples are tested.*

---

## High-Level File Reading

This example demonstrates reading a VRT file with the high-level reader:
- Automatic packet validation
- Type-safe variant access (dynamic::DataPacketView, dynamic::ContextPacketView)
- Elegant iteration with for_each helpers
- Zero-copy access to packet data

```cpp
    using namespace vrtigo::dynamic; // File reader returns dynamic packet views
    using namespace vrtigo::field;

    // Open VRT file - that's it!
    vrtigo::VRTFileReader<> reader(sine_wave_file.c_str());
    ASSERT_TRUE(reader.is_open());

    // Count packets and samples
    size_t data_packets = 0;
    size_t context_packets = 0;
    size_t total_samples = 0;

    // Iterate through all valid packets
    reader.for_each_validated_packet([&](const vrtigo::PacketVariant& pkt) {
        if (vrtigo::is_data_packet(pkt)) {
            data_packets++;

            // Access type-safe data packet view
            const auto& data = std::get<DataPacketView>(pkt);

            // Get payload - zero-copy span into file buffer!
            auto payload = data.payload();
            total_samples += payload.size() / 4; // 4 bytes per I/Q sample

        } else if (vrtigo::is_context_packet(pkt)) {
            context_packets++;

            // Access context packet view
            const auto& ctx = std::get<ContextPacketView>(pkt);
            if (auto sr = ctx[sample_rate]) {
                std::cout << "Context packet sample rate: " << sr.value() << " Hz\n";
            }
        }

        return true; // Continue processing
    });
```

## Manual Packet Iteration

This example shows manual packet iteration for more control.
Use this when you need to handle invalid packets or implement
custom processing logic.

```cpp
    using namespace vrtigo::dynamic; // File reader returns dynamic packet views

    // Manual packet iteration with full control
    vrtigo::VRTFileReader<> reader(sine_wave_file.c_str());
    ASSERT_TRUE(reader.is_open());

    size_t valid_packets = 0;
    size_t invalid_packets = 0;

    // Read packets one at a time
    while (auto pkt = reader.read_next_packet()) {
        if (pkt.has_value()) {
            valid_packets++;

            // Access packet type
            auto type = vrtigo::packet_type(pkt.value());
            std::cout << "Type: " << static_cast<int>(type) << "\n";

            // Process based on type
            if (vrtigo::is_data_packet(pkt.value())) {
                const auto& data = std::get<DataPacketView>(pkt.value());
                auto payload = data.payload();
                // Process payload...
                (void)payload;
            }

        } else {
            invalid_packets++;

            // Get error details from parse result
            auto error = pkt.error();
            // Handle error...
            (void)error;
        }
    }
```

