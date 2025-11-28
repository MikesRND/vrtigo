# Bandwidth Field (CIF0 bit 29)

*Auto-generated from `tests/cif_access/cif0_29_bandwidth_test.cpp`. All examples are tested.*

---

## Setting and Reading Bandwidth

The Bandwidth field is a 64-bit Q52.12 fixed-point value representing
the signal bandwidth in Hz.

```cpp
    using BandwidthContext = ContextPacket<NoTimestamp, NoClassId, bandwidth>;

    alignas(4) std::array<uint8_t, BandwidthContext::size_bytes()> buffer{};
    BandwidthContext packet(buffer);

    // Set bandwidth to 20 MHz
    packet[bandwidth].set_value(20'000'000.0);

    // Read back the value in Hz
    double bw_hz = packet[bandwidth].value();

    // Can also access the encoded Q52.12 value directly
    uint64_t encoded = packet[bandwidth].encoded();
```

