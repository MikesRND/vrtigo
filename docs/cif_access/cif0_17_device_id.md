# Device Identifier Field (CIF0 bit 17)

*Auto-generated from `tests/cif_access/cif0_17_device_id_test.cpp`. All examples are tested.*

---

## Setting and Reading Device Identifier

The Device Identifier field identifies the manufacturer and model of the
device generating the VRT packet stream. Per VITA 49.2 Section 9.10.1:
- Manufacturer OUI: 24-bit IEEE-registered Organizationally Unique Identifier
- Device Code: 16-bit manufacturer-assigned device model identifier

```cpp
    using DeviceIdContext = ContextPacket<NoTimestamp, NoClassId, device_id>;

    alignas(4) std::array<uint8_t, DeviceIdContext::size_bytes> buffer{};
    DeviceIdContext packet(buffer.data());

    // Access via interpreted view
    auto view = packet[device_id].value();

    // Set manufacturer OUI (e.g., 0x00001A = Cisco)
    view.set(DeviceIdentifier::manufacturer_oui, 0x00001A);
    // Set device code
    view.set(DeviceIdentifier::device_code, 0x1234);

    // Read back values
    uint32_t oui = view.get(DeviceIdentifier::manufacturer_oui);
    uint16_t code = view.get(DeviceIdentifier::device_code);
```

