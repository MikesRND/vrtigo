# Data Packet (Signal Samples)

*Auto-generated from `tests/quickstart/data_packet_test.cpp`. All examples are tested.*

---

## Creating a Signal Data Packet

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
    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    // Create simple payload
    std::array<uint8_t, 8> payload{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};

    // Build packet with current timestamp and payload
    auto packet = vrtigo::PacketBuilder<PacketType>(buffer)
                      .stream_id(0x12345678)                      // Set stream identifier
                      .timestamp(vrtigo::UtcRealTimestamp::now()) // Set current time
                      .packet_count(1)                            // First packet in stream
                      .payload(payload)                           // Attach payload
                      .build();

    // The packet is now ready to transmit
    // You can access fields: packet.stream_id(), packet.timestamp(), etc.
```

