# RF Reference Frequency Field (CIF0 bit 27)

*Auto-generated from `tests/cif_access/cif0_27_rf_reference_frequency_test.cpp`. All examples are tested.*

---

## Setting and Reading RF Reference Frequency

The RF Reference Frequency field specifies the original RF frequency that was
translated to the IF Reference Frequency. It uses 64-bit two's complement Q44.20
fixed-point format, providing ±8.79 THz range with 0.95 µHz resolution.

```cpp
    using RFRefContext =
        typed::ContextPacketBuilder<NoTimestamp, NoClassId, rf_reference_frequency>;

    alignas(4) std::array<uint8_t, RFRefContext::size_bytes()> buffer{};
    RFRefContext packet(buffer);

    // Set RF reference frequency to 2.4 GHz (WiFi band)
    packet[rf_reference_frequency].set_value(2.4e9);

    // Read back the value in Hz
    double freq_hz = packet[rf_reference_frequency].value();
```

