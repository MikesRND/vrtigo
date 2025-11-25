# IF Band Offset Field (CIF0 bit 25)

*Auto-generated from `tests/cif_access/cif0_25_if_band_offset_test.cpp`. All examples are tested.*

---

## Setting and Reading IF Band Offset

The IF Band Offset field specifies the frequency offset from IF Reference Frequency
to the center of the band. Band Center = IF Reference Frequency + IF Band Offset.
Uses 64-bit two's complement Q44.20 fixed-point format.

```cpp
    using IFOffsetContext = ContextPacket<NoTimestamp, NoClassId, if_band_offset>;

    alignas(4) std::array<uint8_t, IFOffsetContext::size_bytes> buffer{};
    IFOffsetContext packet(buffer.data());

    // Set IF band offset to 500 kHz
    packet[if_band_offset].set_value(500.0e3);

    // Read back the value in Hz
    double offset_hz = packet[if_band_offset].value();
```

