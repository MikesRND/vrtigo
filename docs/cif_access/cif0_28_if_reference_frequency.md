# IF Reference Frequency Field (CIF0 bit 28)

*Auto-generated from `tests/cif_access/cif0_28_if_reference_frequency_test.cpp`. All examples are tested.*

---

## Setting and Reading IF Reference Frequency

The IF Reference Frequency field indicates a frequency within the usable spectrum
of the described signal. It uses 64-bit two's complement Q44.20 fixed-point format
(radix point right of bit 20), providing ±8.79 THz range with 0.95 µHz resolution.

```cpp
    using IFRefContext = ContextPacket<NoTimestamp, NoClassId, if_reference_frequency>;

    alignas(4) std::array<uint8_t, IFRefContext::size_bytes()> buffer{};
    IFRefContext packet(buffer);

    // Set IF reference frequency to 10.7 MHz (typical AM/FM IF)
    packet[if_reference_frequency].set_value(10.7e6);

    // Read back the value in Hz
    double freq_hz = packet[if_reference_frequency].value();
```

