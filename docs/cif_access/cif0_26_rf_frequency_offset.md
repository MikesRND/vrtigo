# RF Frequency Offset Field (CIF0 bit 26)

*Auto-generated from `tests/cif_access/cif0_26_rf_frequency_offset_test.cpp`. All examples are tested.*

---

## Setting and Reading RF Frequency Offset

The RF Frequency Offset field works with RF Reference Frequency to describe
channelized signals. When present, the original frequency is RF Reference Frequency
+ RF Frequency Offset. Uses 64-bit two's complement Q44.20 fixed-point format.

```cpp
    using RFOffsetContext = ContextPacket<NoTimestamp, NoClassId, rf_frequency_offset>;

    alignas(4) std::array<uint8_t, RFOffsetContext::size_bytes> buffer{};
    RFOffsetContext packet(buffer.data());

    // Set RF frequency offset to 1 MHz (channelizer offset)
    packet[rf_frequency_offset].set_value(1.0e6);

    // Read back the value in Hz
    double offset_hz = packet[rf_frequency_offset].value();
```

