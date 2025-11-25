// Self-Documenting CIF Access Test
// Demonstrates accessing and manipulating the Device Identifier field
//
// This file auto-generates: docs/cif_access/cif0_17_device_id.md

#include "../context_test_fixture.hpp"

using namespace vrtigo;
using namespace vrtigo::field;

// [TITLE]
// Device Identifier Field (CIF0 bit 17)
// [/TITLE]

TEST_F(ContextPacketTest, CIF0_17_BasicAccess) {
    // [EXAMPLE]
    // Setting and Reading Device Identifier
    // [/EXAMPLE]

    // [DESCRIPTION]
    // The Device Identifier field identifies the manufacturer and model of the
    // device generating the VRT packet stream. Per VITA 49.2 Section 9.10.1:
    // - Manufacturer OUI: 24-bit IEEE-registered Organizationally Unique Identifier
    // - Device Code: 16-bit manufacturer-assigned device model identifier
    // [/DESCRIPTION]

    // [SNIPPET]
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
    // [/SNIPPET]

    // Assertions
    EXPECT_EQ(oui, 0x00001A);
    EXPECT_EQ(code, 0x1234);
}

// Additional tests (not included in documentation)

TEST_F(ContextPacketTest, CIF0_17_ZeroValues) {
    using DeviceIdContext = ContextPacket<NoTimestamp, NoClassId, device_id>;
    alignas(4) std::array<uint8_t, DeviceIdContext::size_bytes> buffer{};
    DeviceIdContext packet(buffer.data());

    auto view = packet[device_id].value();

    // Set zero values
    view.set(DeviceIdentifier::manufacturer_oui, 0x000000);
    view.set(DeviceIdentifier::device_code, 0x0000);

    EXPECT_EQ(view.get(DeviceIdentifier::manufacturer_oui), 0x000000);
    EXPECT_EQ(view.get(DeviceIdentifier::device_code), 0x0000);
}

TEST_F(ContextPacketTest, CIF0_17_MaxValues) {
    using DeviceIdContext = ContextPacket<NoTimestamp, NoClassId, device_id>;
    alignas(4) std::array<uint8_t, DeviceIdContext::size_bytes> buffer{};
    DeviceIdContext packet(buffer.data());

    auto view = packet[device_id].value();

    // Set maximum values per VITA 49.2 rules
    // Rule 9.10.1-2: OUI range 00-00-00 to FF-FE-FF, plus FF-FF-FF for unknown
    view.set(DeviceIdentifier::manufacturer_oui, 0xFFFFFF);
    view.set(DeviceIdentifier::device_code, 0xFFFF);

    EXPECT_EQ(view.get(DeviceIdentifier::manufacturer_oui), 0xFFFFFF);
    EXPECT_EQ(view.get(DeviceIdentifier::device_code), 0xFFFF);
}

TEST_F(ContextPacketTest, CIF0_17_TypicalOUIValues) {
    using DeviceIdContext = ContextPacket<NoTimestamp, NoClassId, device_id>;
    alignas(4) std::array<uint8_t, DeviceIdContext::size_bytes> buffer{};
    DeviceIdContext packet(buffer.data());

    auto view = packet[device_id].value();

    // Test some well-known IEEE OUI values
    struct TestCase {
        uint32_t oui;
        const char* description;
    };

    TestCase test_cases[] = {
        {0x00001A, "Cisco"},
        {0x000C29, "VMware"},
        {0x001122, "Cimsys"},
        {0xFFFFFF, "Unknown manufacturer (Permission 9.10.1-3)"},
    };

    for (const auto& tc : test_cases) {
        view.set(DeviceIdentifier::manufacturer_oui, tc.oui);
        EXPECT_EQ(view.get(DeviceIdentifier::manufacturer_oui), tc.oui)
            << "Failed for: " << tc.description;
    }
}

TEST_F(ContextPacketTest, CIF0_17_ReservedBitsIsolation) {
    using DeviceIdContext = ContextPacket<NoTimestamp, NoClassId, device_id>;
    alignas(4) std::array<uint8_t, DeviceIdContext::size_bytes> buffer{};
    DeviceIdContext packet(buffer.data());

    auto view = packet[device_id].value();

    // Set known values
    view.set(DeviceIdentifier::manufacturer_oui, 0x123456);
    view.set(DeviceIdentifier::device_code, 0xABCD);

    // Verify that OUI only uses 24 bits (reserved bits 31-24 of word 0 should not affect it)
    EXPECT_EQ(view.get(DeviceIdentifier::manufacturer_oui), 0x123456);
    // Verify that device code only uses 16 bits (reserved bits 31-16 of word 1 should not affect
    // it)
    EXPECT_EQ(view.get(DeviceIdentifier::device_code), 0xABCD);
}

TEST_F(ContextPacketTest, CIF0_17_RuntimeAccess) {
    // Create a compile-time packet with Device ID
    using DeviceIdContext = ContextPacket<NoTimestamp, NoClassId, device_id>;
    alignas(4) std::array<uint8_t, DeviceIdContext::size_bytes> buffer{};
    DeviceIdContext packet(buffer.data());

    auto view = packet[device_id].value();
    view.set(DeviceIdentifier::manufacturer_oui, 0xABCDEF);
    view.set(DeviceIdentifier::device_code, 0x5678);

    // Parse with runtime packet
    RuntimeContextPacket runtime_packet(buffer.data(), buffer.size());

    // Runtime packet can access the Device ID field
    EXPECT_TRUE(runtime_packet.is_valid());
    if (runtime_packet[device_id]) {
        auto rt_view = runtime_packet[device_id].value();
        EXPECT_EQ(rt_view.get(DeviceIdentifier::manufacturer_oui), 0xABCDEF);
        EXPECT_EQ(rt_view.get(DeviceIdentifier::device_code), 0x5678);
    }
}

TEST_F(ContextPacketTest, CIF0_17_EncodedAccess) {
    using DeviceIdContext = ContextPacket<NoTimestamp, NoClassId, device_id>;
    alignas(4) std::array<uint8_t, DeviceIdContext::size_bytes> buffer{};
    DeviceIdContext packet(buffer.data());

    auto view = packet[device_id].value();

    // Set values and verify via raw FieldView access
    view.set(DeviceIdentifier::manufacturer_oui, 0x112233);
    view.set(DeviceIdentifier::device_code, 0x4455);

    // Access via encoded() returns FieldView<2>
    auto field_view = packet[device_id].encoded();

    // Word 0 should have OUI in lower 24 bits
    EXPECT_EQ(field_view.word(0) & 0xFFFFFF, 0x112233);

    // Word 1 should have device code in lower 16 bits
    EXPECT_EQ(field_view.word(1) & 0xFFFF, 0x4455);
}

TEST_F(ContextPacketTest, CIF0_17_WithOtherFields) {
    // Device ID combined with other fields
    using CombinedContext = ContextPacket<NoTimestamp, NoClassId, device_id, reference_point_id>;

    alignas(4) std::array<uint8_t, CombinedContext::size_bytes> buffer{};
    CombinedContext packet(buffer.data());

    // Set device ID
    auto dev_view = packet[device_id].value();
    dev_view.set(DeviceIdentifier::manufacturer_oui, 0xAABBCC);
    dev_view.set(DeviceIdentifier::device_code, 0x1234);

    // Set reference point ID
    packet[reference_point_id].set_value(0xDEADBEEF);

    // Verify both fields maintained correctly
    EXPECT_EQ(dev_view.get(DeviceIdentifier::manufacturer_oui), 0xAABBCC);
    EXPECT_EQ(dev_view.get(DeviceIdentifier::device_code), 0x1234);
    EXPECT_EQ(packet[reference_point_id].value(), 0xDEADBEEF);
}
