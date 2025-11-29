# Context Field Change Indicator (CIF0 bit 31)

*Auto-generated from `tests/cif_access/cif0_31_change_indicator_test.cpp`. All examples are tested.*

---

## Setting and Reading the Change Indicator

The Context Field Change Indicator (CIF0 bit 31) indicates whether any
context field values have changed since the previous context packet.
Per VITA 49.2: false = no changes, true = at least one field changed.

> **Special Case:** Accessed via packet-level methods, not the field proxy pattern.

```cpp
    using TestContext = typed::ContextPacket<NoTimestamp, NoClassId, bandwidth>;

    alignas(4) std::array<uint8_t, TestContext::size_bytes()> buffer{};
    TestContext packet(buffer);

    // Indicate that context fields have changed
    packet.set_change_indicator(true);

    // Read the indicator
    bool has_changes = packet.change_indicator();
```

