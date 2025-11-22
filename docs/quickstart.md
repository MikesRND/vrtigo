# VRTIGO Quickstart Code Examples

This file contains executable code snippets automatically extracted from test files in `tests/quickstart/`.
All code here is tested and guaranteed to compile.

---

## Data Packet (Signal Samples)

This example demonstrates creating a simple VRT signal data packet with:
- UTC timestamp (current time)
- 8-byte payload
- Stream identifier

The builder pattern provides a fluent API for packet creation.

```cpp
    // Create a VRT Signal Data Packet with timestamp and payload

    // Define packet type with UTC timestamp
    using PacketType = vrtigo::SignalDataPacket<vrtigo::NoClassId,        // No class ID
                                                vrtigo::UtcRealTimestamp, // Include UTC timestamp
                                                vrtigo::Trailer::none,    // No trailer
                                                2 // Max payload words (8 bytes)
                                                >;

    // Allocate aligned buffer for the packet
    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};

    // Create simple payload
    std::array<uint8_t, 8> payload{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};

    // Build packet with current timestamp and payload
    auto packet = vrtigo::PacketBuilder<PacketType>(buffer.data())
                      .stream_id(0x12345678)                      // Set stream identifier
                      .timestamp(vrtigo::UtcRealTimestamp::now()) // Set current time
                      .packet_count(1)                            // First packet in stream
                      .payload(payload)                           // Attach payload
                      .build();

    // The packet is now ready to transmit
    // You can access fields: packet.stream_id(), packet.timestamp(), etc.
```

## Context Packet (Signal Metadata)

This example demonstrates creating a VRT context packet to describe
signal characteristics.


```cpp
    // Create a VRT Context Packet with signal parameters
    using namespace vrtigo::field; // Enable short field syntax

    // Define context packet type with sample rate and bandwidth fields
    using PacketType = vrtigo::ContextPacket<vrtigo::NoTimestamp, // No timestamp for this example
                                             vrtigo::NoClassId,   // No class ID
                                             sample_rate,         // Include sample rate field
                                             bandwidth            // Include bandwidth field
                                             >;

    // Allocate aligned buffer for the packet
    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};

    // Create context packet
    PacketType packet(buffer.data());

    packet.set_stream_id(0x12345678);
    packet[sample_rate].set_value(100'000'000.0); // 100 MHz sample rate
    packet[bandwidth].set_value(40'000'000.0);    // 40 MHz bandwidth

    // The packet is ready to transmit
    // You can read back values:
    auto fs = packet[sample_rate].value(); // Returns 100'000'000.0 Hz
    auto bw = packet[bandwidth].value();   // Returns 40'000'000.0 Hz
```

## Reading VRT Files

This example demonstrates reading a VRT file with the high-level reader:
- Automatic packet validation
- Type-safe variant access (RuntimeDataPacket, RuntimeContextPacket)
- Elegant iteration with for_each helpers
- Zero-copy access to packet data

```cpp
    // Read VRT packets from a file

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
            const auto& data = std::get<vrtigo::RuntimeDataPacket>(pkt);

            // Get payload - zero-copy span into file buffer!
            auto payload = data.payload();
            total_samples += payload.size() / 4; // 4 bytes per I/Q sample

        } else if (vrtigo::is_context_packet(pkt)) {
            context_packets++;

            // Access context packet view
            const auto& ctx = std::get<vrtigo::RuntimeContextPacket>(pkt);
            if (auto sr = ctx[sample_rate]) {
                std::cout << "Context packet sample rate: " << sr.value() << " Hz\n";
            }
        }

        return true; // Continue processing
    });
```

