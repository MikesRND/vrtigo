# Data Packet (Signal Samples)

*Auto-generated from `tests/quickstart/data_packet_test.cpp`. All examples are tested.*

---

## Creating a Signal Data Packet

This example demonstrates creating a simple VRT signal data packet with:
- UTC timestamp (current time)
- 8 real int16_t samples using typed sample access
- Stream identifier

The builder API allows direct packet construction in user-provided buffers.

```cpp
    // Create a VRT Signal Data Packet with 8 real int16_t samples

    // Define packet builder type: 4 words = 16 bytes = 8 int16_t samples
    using PacketType =
        vrtigo::typed::SignalDataPacketBuilder<4,                       // Payload words (16 bytes)
                                               vrtigo::UtcRealTimestamp // Include UTC timestamp
                                               >;

    // Allocate aligned buffer for the packet
    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};

    // Build packet structure
    PacketType packet(buffer);
    packet.set_stream_id(0x12345678);                      // Set stream identifier
    packet.set_timestamp(vrtigo::UtcRealTimestamp::now()); // Set current time
    packet.set_packet_count(1);                            // First packet in stream

    // Write samples using typed access - handles endian conversion automatically
    auto samples = vrtigo::SampleSpan<int16_t>(packet.payload());
    for (size_t i = 0; i < samples.count(); ++i) {
        samples.set(i, static_cast<int16_t>(i * 100)); // 0, 100, 200, ...
    }

    // The packet is now ready to transmit
    // Read back: samples[0] returns 0, samples[1] returns 100, etc.
```

