# Reference Point Identifier Field (CIF0 bit 30)

*Auto-generated from `tests/cif_access/cif0_30_reference_point_id_test.cpp`. All examples are tested.*

---

## Setting and Reading Reference Point ID

The Reference Point Identifier field contains the Stream ID of the Reference
Point where VRT timestamps are measured. Per VITA 49.2, this identifies the
point in the signal processing architecture (often an analog signal like an
air interface) where the timing reference applies.

```cpp
    using RefPointContext = typed::ContextPacket<NoTimestamp, NoClassId, reference_point_id>;

    alignas(4) std::array<uint8_t, RefPointContext::size_bytes()> buffer{};
    RefPointContext packet(buffer);

    // Set Reference Point ID to a Stream ID
    uint32_t ref_point_sid = 0x12345678;
    packet[reference_point_id].set_value(ref_point_sid);

    // Read back the Stream ID
    uint32_t read_sid = packet[reference_point_id].value();
```

