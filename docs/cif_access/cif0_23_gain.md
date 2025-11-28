# Gain/Attenuation Field (CIF0 bit 23)

*Auto-generated from `tests/cif_access/cif0_23_gain_test.cpp`. All examples are tested.*

---

## Setting and Reading Gain Values

The Gain field describes the signal gain or attenuation from the Reference Point
to the Described Signal. It contains two 16-bit Q9.7 fixed-point subfields:
- Stage 1 Gain (lower 16 bits): Front-end or RF gain
- Stage 2 Gain (upper 16 bits): Back-end or IF gain

Each subfield provides ±256 dB range with 1/128 dB resolution (0.0078125 dB).
Single-stage equipment uses Stage 1 only with Stage 2 set to zero.

```cpp
    using GainContext = ContextPacket<NoTimestamp, NoClassId, gain>;

    alignas(4) std::array<uint8_t, GainContext::size_bytes()> buffer{};
    GainContext packet(buffer);

    // Set single-stage gain of +10 dB (Stage 2 = 0)
    packet[gain].set_value({10.0, 0.0});

    // Read back the gain value
    auto gain_value = packet[gain].value();
    double stage1 = gain_value.stage1_db;
    double stage2 = gain_value.stage2_db;
    double total = gain_value.total_db();
```

