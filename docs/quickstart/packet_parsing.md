# Parsing Packets

*Auto-generated from `tests/quickstart/packet_parsing_test.cpp`. All examples are tested.*

---

## Parsing Unknown Packet Types

When you receive VRT packet bytes from a network or file, you don't
know the packet type in advance. Use `parse_packet()` to parse the
buffer and get a `ParseResult<PacketVariant>` - which either holds
a valid packet variant or parse error information.

```cpp
    // Parse a packet when you don't know its type
    std::span<const uint8_t> received_bytes = buffer;

    // parse_packet() returns a ParseResult<PacketVariant>:
    // - On success: contains dynamic::DataPacketView or dynamic::ContextPacketView
    // - On failure: contains ParseError with error details
    auto result = vrtigo::parse_packet(received_bytes);

    // Check if parsing succeeded
    if (!result.ok()) {
        // Get error details from the ParseError
        std::cerr << "Parse failed: " << result.error().message() << "\n";
        return;
    }

    // Get the packet variant
    const auto& packet = result.value();

    // Determine packet type and process accordingly
    if (vrtigo::is_data_packet(packet)) {
        std::cout << "Received data packet\n";
    } else if (vrtigo::is_context_packet(packet)) {
        std::cout << "Received context packet\n";
    }

    // Helper functions work on the variant directly
    std::cout << "Packet type: " << static_cast<int>(vrtigo::packet_type(packet)) << "\n";

    if (auto sid = vrtigo::stream_id(packet)) {
        std::cout << "Stream ID: 0x" << std::hex << *sid << std::dec << "\n";
    }
```

Once you know the packet type, use `std::get<>` to access the
typed view and its specific methods.

## Accessing Data Packet Contents

After confirming you have a data packet, extract the typed view
to access payload data and packet-specific fields.

```cpp
    // Extract the data packet view from the variant
    const auto& data = std::get<vrtigo::dynamic::DataPacketView>(packet);

    // Access packet metadata
    std::cout << "Size: " << data.size_bytes() << " bytes\n";
    std::cout << "Packet count: " << static_cast<int>(data.packet_count()) << "\n";

    // Access optional fields
    if (auto stream_id = data.stream_id()) {
        std::cout << "Stream ID: 0x" << std::hex << *stream_id << std::dec << "\n";
    }

    // Access timestamp if present
    if (data.has_timestamp()) {
        if (auto ts = data.timestamp()) {
            std::cout << "TSI: " << ts->tsi() << ", TSF: " << ts->tsf() << "\n";
        }
    }

    // Access payload - zero-copy span into the buffer
    std::span<const uint8_t> payload = data.payload();
    std::cout << "Payload: " << payload.size() << " bytes\n";

    // Process payload data
    for (size_t i = 0; i < std::min(payload.size(), size_t{4}); ++i) {
        std::cout << "  [" << i << "] = 0x" << std::hex << static_cast<int>(payload[i]) << std::dec
                  << "\n";
    }
```

Context packets provide access to CIF-encoded metadata fields
like sample rate, bandwidth, and gain using the field tag syntax.

## Accessing Context Packet Fields

Context packets carry signal metadata. Access fields using
`operator[]` with field tags. Each field provides `.value()`
for interpreted units and `.encoded()` for the raw wire format.

```cpp
    // Extract the context packet view from the variant
    const auto& ctx = std::get<vrtigo::dynamic::ContextPacketView>(packet);
    using namespace vrtigo::field;

    // Access stream ID (always present in context packets)
    if (auto sid = ctx.stream_id()) {
        std::cout << "Stream ID: 0x" << std::hex << *sid << std::dec << "\n";
    }

    // Access CIF fields using operator[] with field tags
    // The proxy is falsy if the field isn't present
    if (auto sr = ctx[sample_rate]) {
        // .value() returns interpreted units (Hz)
        std::cout << "Sample rate: " << sr.value() / 1e6 << " MHz\n";

        // .encoded() returns the on-wire format (Q52.12)
        std::cout << "  (encoded: 0x" << std::hex << sr.encoded() << std::dec << ")\n";
    }

    if (auto bw = ctx[bandwidth]) {
        std::cout << "Bandwidth: " << bw.value() / 1e6 << " MHz\n";
    }

    if (auto ref = ctx[reference_level]) {
        std::cout << "Reference level: " << ref.value() << " dBm\n";
    }

    // Check for fields that aren't present
    if (!ctx[gain]) {
        std::cout << "Gain field not present in this packet\n";
    }
```

## Handling Invalid Packets

When a packet fails validation, `parse_packet()` returns a
`ParseResult` in error state with details for debugging.

```cpp
    // Handling packets that fail validation
    std::span<const uint8_t> received_bytes = bad_buffer;

    auto result = vrtigo::parse_packet(received_bytes);

    if (!result.ok()) {
        const auto& error = result.error();

        // Get the validation error message
        std::cout << "Validation error: " << error.message() << "\n";

        // Access additional debug info
        std::cout << "Attempted type: " << static_cast<int>(error.attempted_type) << "\n";
        std::cout << "Raw bytes available: " << error.raw_bytes.size() << "\n";
    }
```

