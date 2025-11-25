# Reference Level Field (CIF0 bit 24)

*Auto-generated from `tests/cif_access/cif0_24_reference_level_test.cpp`. All examples are tested.*

---

## Setting and Reading Reference Level

The Reference Level field specifies the reference level at the Reference Point
in dBm. It uses a 16-bit two's complement Q9.7 fixed-point format stored in
the lower 16 bits of a 32-bit word, providing ±256 dBm range with 1/128 dBm
resolution (approximately 0.0078 dBm).

```cpp
    using RefLevelContext = ContextPacket<NoTimestamp, NoClassId, reference_level>;

    alignas(4) std::array<uint8_t, RefLevelContext::size_bytes> buffer{};
    RefLevelContext packet(buffer.data());

    // Set reference level to -10 dBm (typical receiver reference)
    packet[reference_level].set_value(-10.0);

    // Read back the value in dBm
    double level_dbm = packet[reference_level].value();
```

