# Data Packet (Signal Samples)

*Auto-generated from `tests/quickstart/data_packet_test.cpp`. All examples are tested.*

---

## Creating a Signal Data Packet

This example demonstrates creating a simple VRT signal data packet with:
- UTC timestamp (current time)
- 8-byte payload
- Stream identifier

The builder API allows direct packet construction in user-provided buffers.

```cpp
    // Create a VRT Signal Data Packet with timestamp and payload

    // Define packet builder type with UTC timestamp
    using PacketType =
        vrtigo::typed::SignalDataPacketBuilder<2,                       // Payload words (8 bytes)
                                               vrtigo::UtcRealTimestamp // Include UTC timestamp
                                               >;

    // Allocate aligned buffer for the packet
    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    // Create simple payload
    std::array<uint8_t, 8> payload{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};

    // Build packet with current timestamp and payload
    PacketType packet(buffer);
    packet.set_stream_id(0x12345678);                      // Set stream identifier
    packet.set_timestamp(vrtigo::UtcRealTimestamp::now()); // Set current time
    packet.set_packet_count(1);                            // First packet in stream
    packet.set_payload(payload.data(), payload.size());    // Attach payload

    // The packet is now ready to transmit
    // You can access fields: packet.stream_id(), packet.timestamp(), etc.
```

