#include <array>

#include <cstring>
#include <gtest/gtest.h>
#include <vrtigo.hpp>

using namespace vrtigo;

// =============================================================================
// Basic Functionality Tests
// =============================================================================

// Test: Default initialization creates max-size packet
TEST(VariablePacketSizeTest, DefaultInitCreatesMaxSizePacket) {
    using PacketType = typed::SignalDataPacketBuilder<64>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};
    PacketType packet(buffer);

    EXPECT_EQ(packet.size_words(), PacketType::max_size_words());
    EXPECT_EQ(packet.payload_words(), 64);
    EXPECT_EQ(packet.payload_size_bytes(), 64 * 4);
}

// Test: set_payload_size() updates header correctly
TEST(VariablePacketSizeTest, SetPayloadSizeUpdatesHeader) {
    using PacketType = typed::SignalDataPacketBuilder<64>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};
    PacketType packet(buffer);

    // Initial state
    EXPECT_EQ(packet.payload_words(), 64);

    // Resize to smaller
    EXPECT_TRUE(packet.set_payload_size(32));
    EXPECT_EQ(packet.payload_words(), 32);
    EXPECT_EQ(packet.payload_size_bytes(), 32 * 4);

    // Verify total packet size updated
    // stream_id header (2 words) + 32 payload words = 34 words
    EXPECT_EQ(packet.size_words(), 2 + 32);
}

// Test: set_payload(span) auto-resizes packet to fit data
TEST(VariablePacketSizeTest, SetPayloadAutoResizes) {
    using PacketType = typed::SignalDataPacketBuilder<64>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};
    PacketType packet(buffer);

    // Create test data smaller than max
    std::array<uint8_t, 48> data{};
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(i);
    }

    EXPECT_TRUE(packet.set_payload(data));

    // 48 bytes = 12 words
    EXPECT_EQ(packet.payload_words(), 12);

    // Verify data copied correctly
    auto payload = packet.payload();
    EXPECT_EQ(payload.size(), 48);
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(payload[i], data[i]);
    }
}

// Test: payload() returns span with correct size after resize
TEST(VariablePacketSizeTest, PayloadSpanCorrectAfterResize) {
    using PacketType = typed::SignalDataPacketBuilder<128>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};
    PacketType packet(buffer);

    // Initial size
    EXPECT_EQ(packet.payload().size(), 128 * 4);

    // Resize down
    packet.set_payload_size(32);
    EXPECT_EQ(packet.payload().size(), 32 * 4);

    // Resize back up
    packet.set_payload_size(100);
    EXPECT_EQ(packet.payload().size(), 100 * 4);
}

// Test: as_bytes() returns span with correct size after resize
TEST(VariablePacketSizeTest, AsBytesCorrectAfterResize) {
    using PacketType = typed::SignalDataPacketBuilder<64>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};
    PacketType packet(buffer);

    // Resize to 32 words payload
    packet.set_payload_size(32);

    // Total: 2 (prologue) + 32 (payload) = 34 words = 136 bytes
    EXPECT_EQ(packet.as_bytes().size(), 34 * 4);
}

// Test: Trailer position moves with payload size
TEST(VariablePacketSizeTest, TrailerMovesWithPayload) {
    using PacketType = typed::SignalDataPacketBuilder<64, NoTimestamp, NoClassId, WithTrailer>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};
    PacketType packet(buffer);

    // Set distinctive trailer value
    TrailerBuilder{0xDEADBEEF}.apply(packet.trailer());
    EXPECT_EQ(packet.trailer().raw(), 0xDEADBEEF);

    // Resize payload
    packet.set_payload_size(32);

    // Re-set trailer after resize
    TrailerBuilder{0xCAFEBABE}.apply(packet.trailer());

    // Verify trailer accessible at new position
    EXPECT_EQ(packet.trailer().raw(), 0xCAFEBABE);

    // Total size: 2 (prologue) + 32 (payload) + 1 (trailer) = 35 words
    EXPECT_EQ(packet.size_words(), 35);
}

// Test: Round-trip with variable size packet
TEST(VariablePacketSizeTest, RoundTripVariableSize) {
    using PacketType = typed::SignalDataPacketBuilder<256, UtcRealTimestamp>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};
    PacketType packet(buffer);

    // Build packet with smaller payload
    packet.set_stream_id(0x12345678);
    auto ts = UtcRealTimestamp(1699000000, 500000000000ULL);
    packet.set_timestamp(ts);
    packet.set_packet_count(7);

    // Set smaller payload
    std::array<uint8_t, 64> data{};
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>((i * 3) & 0xFF);
    }
    packet.set_payload(data);

    // Parse with dynamic view
    auto result = dynamic::DataPacketView::parse(buffer);
    ASSERT_TRUE(result.has_value()) << result.error().message();

    const auto& view = result.value();
    EXPECT_EQ(*view.stream_id(), 0x12345678);
    EXPECT_EQ(view.packet_count(), 7);
    EXPECT_EQ(view.payload_size_bytes(), 64);

    auto view_ts = view.timestamp();
    ASSERT_TRUE(view_ts.has_value());
    EXPECT_EQ(view_ts->tsi(), 1699000000);
    EXPECT_EQ(view_ts->tsf(), 500000000000ULL);
}

// Test: Multiple resize operations work correctly
TEST(VariablePacketSizeTest, MultipleResizeOperations) {
    using PacketType = typed::SignalDataPacketBuilder<128>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};
    PacketType packet(buffer);

    // Resize sequence
    packet.set_payload_size(64);
    EXPECT_EQ(packet.payload_words(), 64);

    packet.set_payload_size(32);
    EXPECT_EQ(packet.payload_words(), 32);

    packet.set_payload_size(100);
    EXPECT_EQ(packet.payload_words(), 100);

    packet.set_payload_size(0);
    EXPECT_EQ(packet.payload_words(), 0);

    packet.set_payload_size(128);
    EXPECT_EQ(packet.payload_words(), 128);
}

// Test: Resize to zero payload
TEST(VariablePacketSizeTest, ResizeToZeroPayload) {
    using PacketType = typed::SignalDataPacketBuilder<64>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};
    PacketType packet(buffer);

    EXPECT_TRUE(packet.set_payload_size(0));
    EXPECT_EQ(packet.payload_words(), 0);
    EXPECT_EQ(packet.payload_size_bytes(), 0);
    EXPECT_EQ(packet.payload().size(), 0);

    // Total size is just prologue: 2 words
    EXPECT_EQ(packet.size_words(), 2);
}

// Test: Resize to max payload
TEST(VariablePacketSizeTest, ResizeToMaxPayload) {
    using PacketType = typed::SignalDataPacketBuilder<64>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};
    PacketType packet(buffer);

    // Resize down then back to max
    packet.set_payload_size(10);
    EXPECT_EQ(packet.payload_words(), 10);

    packet.set_payload_size(64);
    EXPECT_EQ(packet.payload_words(), 64);
    EXPECT_EQ(packet.payload_size_bytes(), 64 * 4);
}

// =============================================================================
// Constructor Initialization Tests
// =============================================================================

// Test: Constructor with init=true seeds header with max_size_words()
TEST(VariablePacketSizeTest, ConstructorInitializesMaxSize) {
    using PacketType = typed::SignalDataPacketBuilder<100>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};
    PacketType packet(buffer); // init=true by default

    // Should be initialized to max size
    EXPECT_EQ(packet.size_words(), PacketType::max_size_words());
    EXPECT_EQ(packet.payload_words(), 100);
}

// Test: size_words() returns correct value immediately after construction
TEST(VariablePacketSizeTest, SizeWordsCorrectAfterConstruction) {
    using PacketType = typed::SignalDataPacketBuilder<50, UtcRealTimestamp, NoClassId, WithTrailer>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};
    PacketType packet(buffer);

    // Prologue (4 words for timestamp) + stream_id (1) + payload (50) + trailer (1) = 56 words
    // Actually: prologue for type-1 with UTC timestamp is 4 words (header + stream_id + tsi + tsf)
    EXPECT_EQ(packet.size_words(), PacketType::max_size_words());
}

// Test: payload_words() returns PayloadWords after fresh construction
TEST(VariablePacketSizeTest, PayloadWordsCorrectAfterConstruction) {
    using PacketType = typed::SignalDataPacketBuilder<77>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};
    PacketType packet(buffer);

    EXPECT_EQ(packet.payload_words(), 77);
}

// =============================================================================
// Bounds Protection Tests (set_payload_size)
// Note: Tests with invalid inputs trigger asserts in debug builds.
// In debug builds, we verify the assert fires via death tests.
// In release builds (NDEBUG defined), we verify the clamping behavior.
// =============================================================================

#ifdef NDEBUG
// Release build: verify clamping behavior

// Test: set_payload_size with overflow clamps to PayloadWords
TEST(VariablePacketSizeTest, SetPayloadSizeOverflowClamps) {
    using PacketType = typed::SignalDataPacketBuilder<64>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};
    PacketType packet(buffer);

    // Request more than capacity - should clamp and return false
    EXPECT_FALSE(packet.set_payload_size(65));
    EXPECT_EQ(packet.payload_words(), 64); // Clamped to max

    EXPECT_FALSE(packet.set_payload_size(1000));
    EXPECT_EQ(packet.payload_words(), 64);
}

// Test: set_payload_size with SIZE_MAX clamps to PayloadWords
TEST(VariablePacketSizeTest, SetPayloadSizeSizeMaxClamps) {
    using PacketType = typed::SignalDataPacketBuilder<32>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};
    PacketType packet(buffer);

    EXPECT_FALSE(packet.set_payload_size(SIZE_MAX));
    EXPECT_EQ(packet.payload_words(), 32);
}

// Test: Spans don't exceed buffer after oversize request
TEST(VariablePacketSizeTest, SpansWithinBufferAfterOversize) {
    using PacketType = typed::SignalDataPacketBuilder<64>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};
    PacketType packet(buffer);

    // Try to set oversize - will clamp
    packet.set_payload_size(100);

    // Spans should still be within buffer
    EXPECT_LE(packet.payload().size(), 64 * 4);
    EXPECT_LE(packet.as_bytes().size(), PacketType::max_size_bytes());
}

#else
// Debug build: verify asserts fire for invalid inputs

TEST(VariablePacketSizeDeathTest, SetPayloadSizeOverflowAsserts) {
    using PacketType = typed::SignalDataPacketBuilder<64>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};
    PacketType packet(buffer);

    // In debug mode, this should trigger an assertion
    EXPECT_DEATH(packet.set_payload_size(65), "Payload size exceeds maximum capacity");
}

TEST(VariablePacketSizeDeathTest, SetPayloadSizeSizeMaxAsserts) {
    using PacketType = typed::SignalDataPacketBuilder<32>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};
    PacketType packet(buffer);

    EXPECT_DEATH(packet.set_payload_size(SIZE_MAX), "Payload size exceeds maximum capacity");
}

// Note: set_payload() does NOT assert - it clamps internally before calling set_payload_size()
// So there's no death test for it. It gracefully truncates and returns false.

#endif // NDEBUG

// Test that set_payload with oversized data truncates (works in both debug and release)
TEST(VariablePacketSizeTest, SetPayloadOversizedTruncates) {
    using PacketType = typed::SignalDataPacketBuilder<16>; // 64 bytes max payload

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};
    PacketType packet(buffer);

    // Create data larger than capacity
    std::array<uint8_t, 100> data{};
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(i);
    }

    // set_payload clamps internally, doesn't assert
    EXPECT_FALSE(packet.set_payload(data)); // Returns false for truncation
    EXPECT_EQ(packet.payload_words(), 16);  // Clamped to max
    EXPECT_EQ(packet.payload_size_bytes(), 64);

    // Verify copied data is correct (first 64 bytes)
    auto payload = packet.payload();
    for (size_t i = 0; i < 64; ++i) {
        EXPECT_EQ(payload[i], data[i]);
    }
}

// =============================================================================
// Bounds Protection Tests (malformed headers, init=false)
// =============================================================================

// Test: Header size < min_size_words() - payload_words returns 0
TEST(VariablePacketSizeTest, MalformedHeaderTooSmall) {
    using PacketType = typed::SignalDataPacketBuilder<64>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};

    // Create a malformed header with size = 1 (less than min)
    uint32_t header = (1U << 28) | 1; // type=1, size=1
    uint32_t header_be = detail::host_to_network32(header);
    std::memcpy(buffer.data(), &header_be, sizeof(header_be));

    PacketType packet(buffer, false); // Don't reinitialize

    // size_words() returns raw value
    EXPECT_EQ(packet.size_words(), 1);

    // payload_words() protected from underflow
    EXPECT_EQ(packet.payload_words(), 0);
}

// Test: Header size > max_size_words() - size_words returns raw value
TEST(VariablePacketSizeTest, MalformedHeaderTooLarge) {
    using PacketType = typed::SignalDataPacketBuilder<64>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};

    // Create a malformed header with size = 1000 (more than max)
    uint32_t header = (1U << 28) | 1000; // type=1, size=1000
    uint32_t header_be = detail::host_to_network32(header);
    std::memcpy(buffer.data(), &header_be, sizeof(header_be));

    PacketType packet(buffer, false);

    // size_words() returns raw (unclamped) value
    EXPECT_EQ(packet.size_words(), 1000);

    // payload_words() returns raw derived value (no upper clamping)
    EXPECT_EQ(packet.payload_words(), 1000 - 2); // minus prologue
}

// Test: Header size = 0 - payload_words returns 0
TEST(VariablePacketSizeTest, MalformedHeaderZeroSize) {
    using PacketType = typed::SignalDataPacketBuilder<64>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};

    // Create a malformed header with size = 0
    uint32_t header = (1U << 28) | 0; // type=1, size=0
    uint32_t header_be = detail::host_to_network32(header);
    std::memcpy(buffer.data(), &header_be, sizeof(header_be));

    PacketType packet(buffer, false);

    EXPECT_EQ(packet.size_words(), 0);
    EXPECT_EQ(packet.payload_words(), 0); // Protected from underflow
}

// Test: Spans clamped even with malformed oversized header
TEST(VariablePacketSizeTest, SpansClampedOnMalformedHeader) {
    using PacketType = typed::SignalDataPacketBuilder<32>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};

    // Create malformed header claiming size = 5000 words
    uint32_t header = (1U << 28) | 5000;
    uint32_t header_be = detail::host_to_network32(header);
    std::memcpy(buffer.data(), &header_be, sizeof(header_be));

    PacketType packet(buffer, false);

    // Raw values are unclamped
    EXPECT_EQ(packet.size_words(), 5000);

    // But spans are clamped for safety
    EXPECT_LE(packet.payload().size(), 32 * 4);
    EXPECT_LE(packet.as_bytes().size(), PacketType::max_size_bytes());
}

// =============================================================================
// Validation API Tests
// =============================================================================

// Test: validate_size() returns none for valid headers
TEST(VariablePacketSizeTest, ValidateSizeValidHeader) {
    using PacketType = typed::SignalDataPacketBuilder<64>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};
    PacketType packet(buffer); // Properly initialized

    EXPECT_EQ(packet.validate_size(), ValidationError::none);
}

// Test: validate_size() returns error when header < min_size_words()
TEST(VariablePacketSizeTest, ValidateSizeHeaderTooSmall) {
    using PacketType = typed::SignalDataPacketBuilder<64>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};

    // Malformed header with size = 1
    uint32_t header = (1U << 28) | 1;
    uint32_t header_be = detail::host_to_network32(header);
    std::memcpy(buffer.data(), &header_be, sizeof(header_be));

    PacketType packet(buffer, false);

    EXPECT_EQ(packet.validate_size(), ValidationError::size_field_mismatch);
}

// Test: validate_size() returns error when header > max_size_words()
TEST(VariablePacketSizeTest, ValidateSizeHeaderTooLarge) {
    using PacketType = typed::SignalDataPacketBuilder<64>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};

    // Malformed header with size = 1000 (exceeds max)
    uint32_t header = (1U << 28) | 1000;
    uint32_t header_be = detail::host_to_network32(header);
    std::memcpy(buffer.data(), &header_be, sizeof(header_be));

    PacketType packet(buffer, false);

    EXPECT_EQ(packet.validate_size(), ValidationError::size_field_mismatch);
}

// Test: validate_size(small_buffer) returns buffer_too_small
TEST(VariablePacketSizeTest, ValidateSizeBufferTooSmall) {
    using PacketType = typed::SignalDataPacketBuilder<64>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};
    PacketType packet(buffer); // Initialized with max size

    // Validate against smaller buffer
    EXPECT_EQ(packet.validate_size(100), ValidationError::buffer_too_small);
}

// Test: Fresh packet always has validate_size() == none
TEST(VariablePacketSizeTest, FreshPacketValidates) {
    using PacketType1 = typed::SignalDataPacketBuilder<128>;
    using PacketType2 = typed::SignalDataPacketBuilderNoId<64, UtcRealTimestamp>;
    using PacketType3 = typed::SignalDataPacketBuilder<32, NoTimestamp, NoClassId, WithTrailer>;

    alignas(4) std::array<uint8_t, PacketType1::max_size_bytes()> buffer1{};
    alignas(4) std::array<uint8_t, PacketType2::max_size_bytes()> buffer2{};
    alignas(4) std::array<uint8_t, PacketType3::max_size_bytes()> buffer3{};

    PacketType1 packet1(buffer1);
    PacketType2 packet2(buffer2);
    PacketType3 packet3(buffer3);

    EXPECT_EQ(packet1.validate_size(), ValidationError::none);
    EXPECT_EQ(packet2.validate_size(), ValidationError::none);
    EXPECT_EQ(packet3.validate_size(), ValidationError::none);
}

// Test: validate_size after resize
TEST(VariablePacketSizeTest, ValidateSizeAfterResize) {
    using PacketType = typed::SignalDataPacketBuilder<64>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};
    PacketType packet(buffer);

    // Resize to various sizes
    packet.set_payload_size(32);
    EXPECT_EQ(packet.validate_size(), ValidationError::none);

    packet.set_payload_size(0);
    EXPECT_EQ(packet.validate_size(), ValidationError::none);

    packet.set_payload_size(64);
    EXPECT_EQ(packet.validate_size(), ValidationError::none);
}

// =============================================================================
// Padding Safety Tests
// =============================================================================

// Test: set_payload with non-word-aligned size zeros padding bytes
TEST(VariablePacketSizeTest, SetPayloadZerosPadding) {
    using PacketType = typed::SignalDataPacketBuilder<32>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};

    // Fill buffer with recognizable pattern
    std::memset(buffer.data(), 0xAA, buffer.size());

    PacketType packet(buffer);

    // Set payload of 5 bytes (not word-aligned)
    std::array<uint8_t, 5> data = {0x01, 0x02, 0x03, 0x04, 0x05};
    packet.set_payload(data);

    // Should be 2 words (8 bytes) of payload
    EXPECT_EQ(packet.payload_words(), 2);
    EXPECT_EQ(packet.payload().size(), 8);

    auto payload = packet.payload();

    // First 5 bytes should be data
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(payload[i], data[i]);
    }

    // Padding bytes (6, 7, 8) should be zero, not 0xAA
    for (size_t i = 5; i < 8; ++i) {
        EXPECT_EQ(payload[i], 0) << "Padding byte " << i << " not zeroed";
    }
}

// Test: No stale data in padding after resize
TEST(VariablePacketSizeTest, NoStaleDataAfterResize) {
    using PacketType = typed::SignalDataPacketBuilder<32>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};
    PacketType packet(buffer);

    // First, fill payload with pattern
    auto payload1 = packet.payload();
    std::memset(payload1.data(), 0xFF, payload1.size());

    // Resize to smaller with non-aligned data
    std::array<uint8_t, 3> small_data = {0x01, 0x02, 0x03};
    packet.set_payload(small_data);

    auto payload2 = packet.payload();
    EXPECT_EQ(payload2.size(), 4); // 1 word

    // Verify padding is zeroed
    EXPECT_EQ(payload2[0], 0x01);
    EXPECT_EQ(payload2[1], 0x02);
    EXPECT_EQ(payload2[2], 0x03);
    EXPECT_EQ(payload2[3], 0); // Padding byte
}

// =============================================================================
// Static Method Tests
// =============================================================================

// Test: Static methods are available at compile time
TEST(VariablePacketSizeTest, StaticMethodsCompileTime) {
    using PacketType = typed::SignalDataPacketBuilder<64, UtcRealTimestamp, NoClassId, WithTrailer>;

    // These should all be compile-time constants
    constexpr size_t max_words = PacketType::max_size_words();
    constexpr size_t max_bytes = PacketType::max_size_bytes();
    constexpr size_t min_words = PacketType::min_size_words();

    // Verify values
    // Prologue: 5 words (header=1 + stream_id=1 + tsi=1 + tsf=2)
    // Payload: 64 words
    // Trailer: 1 word
    EXPECT_EQ(max_words, 5 + 64 + 1);
    EXPECT_EQ(max_bytes, max_words * 4);
    EXPECT_EQ(min_words, 5 + 1); // Prologue + trailer, no payload
}

// Test: Type without trailer
TEST(VariablePacketSizeTest, StaticMethodsNoTrailer) {
    using PacketType = typed::SignalDataPacketBuilder<32>;

    constexpr size_t max_words = PacketType::max_size_words();
    constexpr size_t min_words = PacketType::min_size_words();

    // Prologue: 2 words (header + stream_id)
    EXPECT_EQ(max_words, 2 + 32);
    EXPECT_EQ(min_words, 2); // Just prologue
}

// Test: Type without stream ID
TEST(VariablePacketSizeTest, StaticMethodsNoStreamId) {
    using PacketType = typed::SignalDataPacketBuilderNoId<16>;

    constexpr size_t max_words = PacketType::max_size_words();
    constexpr size_t min_words = PacketType::min_size_words();

    // Prologue: 1 word (header only)
    EXPECT_EQ(max_words, 1 + 16);
    EXPECT_EQ(min_words, 1);
}

// =============================================================================
// Edge Case Tests
// =============================================================================

// Test: Trailer with zero payload
TEST(VariablePacketSizeTest, TrailerWithZeroPayload) {
    using PacketType = typed::SignalDataPacketBuilder<64, NoTimestamp, NoClassId, WithTrailer>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};
    PacketType packet(buffer);

    // Set zero payload
    packet.set_payload_size(0);
    EXPECT_EQ(packet.payload_words(), 0);

    // Trailer should still be accessible
    TrailerBuilder{0x12345678}.apply(packet.trailer());
    EXPECT_EQ(packet.trailer().raw(), 0x12345678);

    // Total: 2 (prologue) + 0 (payload) + 1 (trailer) = 3 words
    EXPECT_EQ(packet.size_words(), 3);
}

// Test: Exact boundary conditions (valid inputs only)
TEST(VariablePacketSizeTest, ExactBoundaries) {
    using PacketType = typed::SignalDataPacketBuilder<100>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};
    PacketType packet(buffer);

    // Set to exact max
    EXPECT_TRUE(packet.set_payload_size(100));
    EXPECT_EQ(packet.payload_words(), 100);

    // Set to exact zero
    EXPECT_TRUE(packet.set_payload_size(0));
    EXPECT_EQ(packet.payload_words(), 0);

    // Set back to max
    EXPECT_TRUE(packet.set_payload_size(100));
    EXPECT_EQ(packet.payload_words(), 100);
}

#ifdef NDEBUG
// Test: Exact boundary overflow (release only - triggers assert in debug)
TEST(VariablePacketSizeTest, ExactBoundaryOverflow) {
    using PacketType = typed::SignalDataPacketBuilder<100>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};
    PacketType packet(buffer);

    // Set to one over max - should clamp
    EXPECT_FALSE(packet.set_payload_size(101));
    EXPECT_EQ(packet.payload_words(), 100); // Clamped
}
#endif

// Test: set_payload with exact word-aligned size
TEST(VariablePacketSizeTest, SetPayloadWordAligned) {
    using PacketType = typed::SignalDataPacketBuilder<32>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};

    // Fill with pattern
    std::memset(buffer.data(), 0xBB, buffer.size());

    PacketType packet(buffer);

    // 16 bytes = 4 words exactly
    std::array<uint8_t, 16> data{};
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(i + 1);
    }

    EXPECT_TRUE(packet.set_payload(data));
    EXPECT_EQ(packet.payload_words(), 4);

    // All payload bytes should be data (no padding needed)
    auto payload = packet.payload();
    EXPECT_EQ(payload.size(), 16);
    for (size_t i = 0; i < 16; ++i) {
        EXPECT_EQ(payload[i], data[i]);
    }
}

// Test: set_payload with empty span
TEST(VariablePacketSizeTest, SetPayloadEmpty) {
    using PacketType = typed::SignalDataPacketBuilder<32>;

    alignas(4) std::array<uint8_t, PacketType::max_size_bytes()> buffer{};
    PacketType packet(buffer);

    EXPECT_TRUE(packet.set_payload(std::span<uint8_t>{}));
    EXPECT_EQ(packet.payload_words(), 0);
    EXPECT_EQ(packet.payload().size(), 0);
}
