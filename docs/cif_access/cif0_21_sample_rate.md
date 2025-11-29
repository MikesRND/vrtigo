# Sample Rate Field (CIF0 bit 21)

*Auto-generated from `tests/cif_access/cif0_21_sample_rate_test.cpp`. All examples are tested.*

---

## Setting and Reading Sample Rate

The Sample Rate field is a 64-bit Q52.12 fixed-point value representing
the sample rate in Hz. Set and read the encoded value directly.

```cpp
    using SampleRateContext = typed::ContextPacket<NoTimestamp, NoClassId, sample_rate>;

    alignas(4) std::array<uint8_t, SampleRateContext::size_bytes()> buffer{};
    SampleRateContext packet(buffer);

    // Set sample rate to 10 MHz
    packet[sample_rate].set_value(10'000'000.0);

    // Read back the value in Hz
    double rate_hz = packet[sample_rate].value();

    // Can also access the encoded Q52.12 value directly
    uint64_t encoded = packet[sample_rate].encoded();
```

