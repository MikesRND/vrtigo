// [TITLE]
// Parsing Packets
// [/TITLE]
//
// This test demonstrates parsing VRT packets from raw buffers using
// the parse_packet() function, which handles unknown packet types
// and returns a type-safe variant.

#include <array>
#include <iostream>
#include <span>

#include <cstdint>
#include <gtest/gtest.h>
#include <vrtigo.hpp>
#include <vrtigo/vrtigo_io.hpp>

// Helper: Create a valid data packet buffer for testing
// In real code, this would come from a network socket or file
static auto create_test_data_packet() {
    using PacketType =
        vrtigo::typed::SignalDataPacketBuilder<2, vrtigo::UtcRealTimestamp>; // 8 bytes payload

    alignas(4) static std::array<uint8_t, PacketType::size_bytes()> buffer{};
    std::array<uint8_t, 8> payload{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};

    PacketType packet(buffer);
    packet.set_stream_id(0xABCD1234);
    packet.set_timestamp(vrtigo::UtcRealTimestamp{0x12345678, 0x9ABCDEF0'12345678ULL});
    packet.set_packet_count(7);
    packet.set_payload(payload.data(), payload.size());

    return buffer;
}

// Helper: Create a valid context packet buffer for testing
static auto create_test_context_packet() {
    using namespace vrtigo::field;
    using PacketType = vrtigo::typed::ContextPacketBuilder<vrtigo::NoTimestamp, vrtigo::NoClassId,
                                                           sample_rate, bandwidth, reference_level>;

    alignas(4) static std::array<uint8_t, PacketType::size_bytes()> buffer{};
    PacketType packet(buffer);

    packet.set_stream_id(0xDEADBEEF);
    packet[sample_rate].set_value(100'000'000.0); // 100 MHz
    packet[bandwidth].set_value(20'000'000.0);    // 20 MHz
    packet[reference_level].set_value(-10.0);     // -10 dBm

    return buffer;
}

// [EXAMPLE]
// Parsing Unknown Packet Types
// [/EXAMPLE]

// [DESCRIPTION]
// When you receive VRT packet bytes from a network or file, you don't
// know the packet type in advance. Use `parse_packet()` to parse the
// buffer and get a `ParseResult<PacketVariant>` - which either holds
// a valid packet variant or parse error information.
// [/DESCRIPTION]

TEST(PacketParsing, ParseUnknownPacket) {
    auto buffer = create_test_data_packet();

    // [SNIPPET]
    // Parse a packet when you don't know its type
    std::span<const uint8_t> received_bytes = buffer;

    // parse_packet() returns a ParseResult<PacketVariant>:
    // - On success: contains dynamic::DataPacketView or dynamic::ContextPacketView
    // - On failure: contains ParseError with error details
    auto result = vrtigo::parse_packet(received_bytes);

    if (!result.ok()) {
        std::cerr << "Parse failed: " << result.error().message() << "\n";
        return;
    }
    const auto& packet = result.value();

    // Determine packet type and process accordingly
    if (vrtigo::is_data_packet(packet)) {
        std::cout << "Received data packet\n";
    } else if (vrtigo::is_context_packet(packet)) {
        std::cout << "Received context packet\n";
    }

    // Helper functions work on the variant directly
    std::cout << "Packet type: " << static_cast<int>(vrtigo::packet_type(packet)) << "\n";
    std::cout << "Stream ID: 0x" << std::hex << *vrtigo::stream_id(packet) << std::dec << "\n";
    // [/SNIPPET]

    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(vrtigo::is_data_packet(result.value()));
    EXPECT_EQ(vrtigo::stream_id(result.value()), 0xABCD1234);
}

// [TEXT]
// Once you know the packet type, use `std::get<>` to access the
// typed view and its specific methods.
// [/TEXT]

// [EXAMPLE]
// Accessing Data Packet Contents
// [/EXAMPLE]

// [DESCRIPTION]
// After confirming you have a data packet, extract the typed view
// to access payload data and packet-specific fields.
// [/DESCRIPTION]

TEST(PacketParsing, AccessDataPacket) {
    auto buffer = create_test_data_packet();
    auto result = vrtigo::parse_packet(buffer);
    ASSERT_TRUE(result.ok());
    const auto& packet = result.value();
    ASSERT_TRUE(vrtigo::is_data_packet(packet));

    // [SNIPPET]
    using namespace vrtigo::dynamic; // Parsing returns dynamic packet views

    // Extract the data packet view from the variant
    const auto& data = std::get<DataPacketView>(packet);

    // Access packet metadata
    std::cout << "Size: " << data.size_bytes() << " bytes\n";
    std::cout << "Packet count: " << +data.packet_count() << "\n";

    // Optional fields return std::optional - dereference when you know they exist
    std::cout << "Stream ID: 0x" << std::hex << *data.stream_id() << std::dec << "\n";
    std::cout << "TSI: " << data.timestamp()->tsi() << ", TSF: " << data.timestamp()->tsf() << "\n";

    // Access payload - zero-copy span into the buffer
    std::cout << "Payload: " << data.payload().size() << " bytes\n";
    // [/SNIPPET]

    EXPECT_EQ(data.size_bytes(), 28);
    EXPECT_EQ(data.payload().size(), 8);
    EXPECT_EQ(data.payload()[0], 0x01);
}

// [TEXT]
// Context packets provide access to CIF-encoded metadata fields
// like sample rate, bandwidth, and gain using the field tag syntax.
// [/TEXT]

// [EXAMPLE]
// Accessing Context Packet Fields
// [/EXAMPLE]

// [DESCRIPTION]
// Context packets carry signal metadata. Access fields using
// `operator[]` with field tags. Each field provides `.value()`
// for interpreted units and `.encoded()` for the raw wire format.
// [/DESCRIPTION]

TEST(PacketParsing, AccessContextPacket) {
    auto buffer = create_test_context_packet();
    auto result = vrtigo::parse_packet(buffer);
    ASSERT_TRUE(result.ok());
    const auto& packet = result.value();
    ASSERT_TRUE(vrtigo::is_context_packet(packet));

    // [SNIPPET]
    using namespace vrtigo::dynamic; // Parsing returns dynamic packet views
    using namespace vrtigo::field;

    // Extract the context packet view from the variant
    const auto& ctx = std::get<ContextPacketView>(packet);

    // Stream ID is always present in context packets
    std::cout << "Stream ID: 0x" << std::hex << *ctx.stream_id() << std::dec << "\n";

    // Access CIF fields directly with operator[] - chain .value() for interpreted units
    std::cout << "Sample rate: " << ctx[sample_rate].value() / 1e6 << " MHz\n";
    std::cout << "Bandwidth: " << ctx[bandwidth].value() / 1e6 << " MHz\n";
    std::cout << "Reference level: " << ctx[reference_level].value() << " dBm\n";

    // Use .encoded() for the raw on-wire format
    std::cout << "  (sample_rate encoded: 0x" << std::hex << ctx[sample_rate].encoded() << std::dec
              << ")\n";

    // Check for fields that might not be present
    if (!ctx[gain]) {
        std::cout << "Gain field not present\n";
    }
    // [/SNIPPET]

    EXPECT_DOUBLE_EQ(ctx[sample_rate].value(), 100'000'000.0);
    EXPECT_DOUBLE_EQ(ctx[bandwidth].value(), 20'000'000.0);
    EXPECT_DOUBLE_EQ(ctx[reference_level].value(), -10.0);
    EXPECT_FALSE(ctx[gain].has_value());
}

// [EXAMPLE]
// Handling Invalid Packets
// [/EXAMPLE]

// [DESCRIPTION]
// When a packet fails validation, `parse_packet()` returns a
// `ParseResult` in error state with details for debugging.
// [/DESCRIPTION]

TEST(PacketParsing, HandleInvalidPacket) {
    // Create a buffer that's too small to be valid
    std::array<uint8_t, 2> bad_buffer{0x00, 0x01};

    // [SNIPPET]
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
    // [/SNIPPET]

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, vrtigo::ValidationError::buffer_too_small);
}
