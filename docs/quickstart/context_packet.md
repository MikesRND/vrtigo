# Context Packet (Signal Metadata)

*Auto-generated from `tests/quickstart/context_packet_test.cpp`. All examples are tested.*

---

## Creating a Context Packet

This example demonstrates creating a VRT context packet to describe
signal characteristics.

```cpp
    // Create a VRT Context Packet with signal parameters
    using namespace vrtigo::field; // Enable short field syntax

    // Define context packet type with sample rate and bandwidth fields
    using PacketType =
        vrtigo::typed::ContextPacket<vrtigo::NoTimestamp, // No timestamp for this example
                                     vrtigo::NoClassId,   // No class ID
                                     sample_rate,         // Include sample rate field
                                     bandwidth            // Include bandwidth field
                                     >;

    // Allocate aligned buffer for the packet
    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    // Create context packet
    PacketType packet(buffer);

    packet.set_stream_id(0x12345678);
    packet[sample_rate].set_value(100'000'000.0); // 100 MHz sample rate
    packet[bandwidth].set_value(40'000'000.0);    // 40 MHz bandwidth

    // The packet is ready to transmit
    // You can read back values:
    auto fs = packet[sample_rate].value(); // Returns 100'000'000.0 Hz
    auto bw = packet[bandwidth].value();   // Returns 40'000'000.0 Hz
```

