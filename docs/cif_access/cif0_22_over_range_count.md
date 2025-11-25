# Over-Range Count Field (CIF0 bit 22)

*Auto-generated from `tests/cif_access/cif0_22_over_range_count_test.cpp`. All examples are tested.*

---

## Setting and Reading Over-Range Count

The Over-Range Count field contains the number of Data Samples in the paired
Data Packet whose amplitudes were beyond the range of the Data Item format.
Per VITA 49.2 Rule 9.10.6-1, this count applies only to the paired Data Packet
with the corresponding Timestamp and does not accumulate over multiple packets.
For complex Cartesian samples, the count includes samples where either the real
or imaginary component was beyond range (Rule 9.10.6-2).

```cpp
    using OverRangeContext = ContextPacket<NoTimestamp, NoClassId, over_range_count>;

    alignas(4) std::array<uint8_t, OverRangeContext::size_bytes> buffer{};
    OverRangeContext packet(buffer.data());

    // Set Over-Range Count to indicate 42 over-range samples
    uint32_t count = 42;
    packet[over_range_count].set_value(count);

    // Read back the count
    uint32_t read_count = packet[over_range_count].value();
```

