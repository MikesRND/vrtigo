#include <array>

#include <gtest/gtest.h>
#include <vrtigo.hpp>

using namespace vrtigo;

// Test basic signal packet without stream ID
TEST(RtDataPacketTest, BasicPacketNoStream) {
    using PacketType = typed::SignalDataPacketBuilderNoId<vrtigo::NoClassId, NoTimestamp,
                                                          vrtigo::Trailer::none, // no trailer
                                                          64                     // 64 words payload
                                                          >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};
    PacketType packet(buffer);

    // Parse with runtime view
    auto result = dynamic::DataPacketView::parse(buffer);

    ASSERT_TRUE(result.ok()) << result.error().message();
    const auto& view = result.value();
    EXPECT_EQ(view.type(), vrtigo::PacketType::signal_data_no_id);
    EXPECT_FALSE(view.has_stream_id());
    EXPECT_FALSE(view.has_trailer());
    EXPECT_FALSE(view.has_timestamp());
    EXPECT_EQ(view.size_words(), PacketType::size_bytes() / 4);
    EXPECT_EQ(view.payload_size_bytes(), 64 * 4);
}

// Test signal packet with stream ID
TEST(RtDataPacketTest, PacketWithStreamID) {
    using PacketType =
        typed::SignalDataPacketBuilder<vrtigo::NoClassId, NoTimestamp, vrtigo::Trailer::none, 64>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    PacketType packet(buffer);
    packet.set_stream_id(0x12345678);

    // Parse with runtime view
    auto result = dynamic::DataPacketView::parse(buffer);

    ASSERT_TRUE(result.ok()) << result.error().message();
    const auto& view = result.value();
    EXPECT_EQ(view.type(), vrtigo::PacketType::signal_data);
    EXPECT_TRUE(view.has_stream_id());

    auto id = view.stream_id();
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(*id, 0x12345678);
}

// Test signal packet with timestamps
TEST(RtDataPacketTest, PacketWithTimestamps) {
    using PacketType = typed::SignalDataPacketBuilder<vrtigo::NoClassId, UtcRealTimestamp,
                                                      vrtigo::Trailer::none, 64>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    PacketType packet(buffer);
    packet.set_stream_id(0xABCDEF00);
    auto build_ts = UtcRealTimestamp(1234567890, 500000000000ULL);
    packet.set_timestamp(build_ts);

    // Parse with runtime view
    auto result = dynamic::DataPacketView::parse(buffer);

    ASSERT_TRUE(result.ok()) << result.error().message();
    const auto& view = result.value();
    EXPECT_TRUE(view.has_timestamp());

    auto ts = view.timestamp();
    ASSERT_TRUE(ts.has_value());
    EXPECT_EQ(view.header().tsi_kind(), TsiType::utc);
    EXPECT_EQ(view.header().tsf_kind(), TsfType::real_time);
    EXPECT_TRUE(ts->has_tsi());
    EXPECT_TRUE(ts->has_tsf());
    EXPECT_EQ(ts->tsi(), 1234567890);
    EXPECT_EQ(ts->tsf(), 500000000000ULL);
}

// Test signal packet with trailer
TEST(RtDataPacketTest, PacketWithTrailer) {
    using PacketType = typed::SignalDataPacketBuilder<vrtigo::NoClassId, NoTimestamp,
                                                      vrtigo::Trailer::included, // has trailer
                                                      64>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    PacketType packet(buffer);
    packet.set_stream_id(0x11111111);
    auto trailer_cfg = vrtigo::TrailerBuilder{0xDEADBEEF};
    trailer_cfg.apply(packet.trailer());

    // Parse with runtime view
    auto result = dynamic::DataPacketView::parse(buffer);

    ASSERT_TRUE(result.ok()) << result.error().message();
    const auto& view = result.value();
    EXPECT_TRUE(view.has_trailer());

    auto trailer = view.trailer();
    ASSERT_TRUE(trailer.has_value());
    EXPECT_EQ(trailer->raw(), 0xDEADBEEF);
}

// Test full-featured packet (stream ID + timestamps + trailer)
TEST(RtDataPacketTest, FullFeaturedPacket) {
    using PacketType = typed::SignalDataPacketBuilder<vrtigo::NoClassId, UtcRealTimestamp,
                                                      vrtigo::Trailer::included, // has trailer
                                                      128                        // larger payload
                                                      >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    PacketType packet(buffer);
    packet.set_stream_id(0xCAFEBABE);
    auto build_ts = UtcRealTimestamp(9999999, 123456789012ULL);
    packet.set_timestamp(build_ts);
    auto trailer_cfg = vrtigo::TrailerBuilder{}.valid_data(true).calibrated_time(true);
    trailer_cfg.apply(packet.trailer());
    packet.set_packet_count(7);

    // Parse with runtime view
    auto result = dynamic::DataPacketView::parse(buffer);

    ASSERT_TRUE(result.ok()) << result.error().message();
    const auto& view = result.value();
    EXPECT_EQ(view.type(), vrtigo::PacketType::signal_data);
    EXPECT_TRUE(view.has_stream_id());
    EXPECT_TRUE(view.has_timestamp());
    EXPECT_TRUE(view.has_trailer());
    EXPECT_EQ(view.packet_count(), 7);

    EXPECT_EQ(*view.stream_id(), 0xCAFEBABE);
    auto ts = view.timestamp();
    ASSERT_TRUE(ts.has_value());
    EXPECT_EQ(ts->tsi(), 9999999);
    EXPECT_EQ(ts->tsf(), 123456789012ULL);
    EXPECT_TRUE(view.trailer().has_value());
}

// Test payload access
TEST(RtDataPacketTest, PayloadAccess) {
    using PacketType =
        typed::SignalDataPacketBuilderNoId<vrtigo::NoClassId, NoTimestamp, vrtigo::Trailer::none,
                                           16 // 16 words = 64 bytes
                                           >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};
    PacketType packet(buffer);

    // Fill payload with test pattern
    auto payload = packet.payload();
    for (size_t i = 0; i < payload.size(); i++) {
        payload[i] = static_cast<uint8_t>(i & 0xFF);
    }

    // Parse with runtime view
    auto result = dynamic::DataPacketView::parse(buffer);

    ASSERT_TRUE(result.ok()) << result.error().message();
    const auto& view = result.value();
    EXPECT_EQ(view.payload_size_bytes(), 64);
    EXPECT_EQ(view.payload_size_words(), 16);

    auto view_payload = view.payload();
    ASSERT_EQ(view_payload.size(), 64);

    // Verify payload contents
    for (size_t i = 0; i < view_payload.size(); i++) {
        EXPECT_EQ(view_payload[i], static_cast<uint8_t>(i & 0xFF));
    }
}

// Test validation: buffer too small
TEST(RtDataPacketTest, ValidationBufferTooSmall) {
    using PacketType = typed::SignalDataPacketBuilderNoId<vrtigo::NoClassId, NoTimestamp,
                                                          vrtigo::Trailer::none, 64>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};
    PacketType packet(buffer);

    // Parse with smaller buffer size
    auto result = dynamic::DataPacketView::parse(
        std::span<const uint8_t>(buffer.data(), 10)); // Only 10 bytes

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, ValidationError::buffer_too_small);
}

// Test validation: wrong packet type
TEST(RtDataPacketTest, ValidationWrongPacketType) {
    alignas(4) std::array<uint8_t, 64> buffer{};

    // Manually create a context packet header (type = 4)
    uint32_t header = (4U << 28) | 10; // type=4 (context), size=10
    uint32_t header_be = detail::host_to_network32(header);
    std::memcpy(buffer.data(), &header_be, sizeof(header_be));

    // Try to parse as signal packet
    auto result = dynamic::DataPacketView::parse(buffer);

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, ValidationError::packet_type_mismatch);
}

// Test validation: empty buffer
TEST(RtDataPacketTest, ValidationEmptyBuffer) {
    auto result = dynamic::DataPacketView::parse(std::span<const uint8_t>{});

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.error().code, ValidationError::buffer_too_small);
}

// Test round-trip: build → parse → verify
TEST(RtDataPacketTest, RoundTripBuildParse) {
    using PacketType = typed::SignalDataPacketBuilder<vrtigo::NoClassId,
                                                      Timestamp<TsiType::gps, TsfType::real_time>,
                                                      vrtigo::Trailer::included, 256>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    // Build packet
    std::array<uint8_t, 256 * 4> payload_data{};
    for (size_t i = 0; i < payload_data.size(); i++) {
        payload_data[i] = static_cast<uint8_t>((i * 7) & 0xFF);
    }

    PacketType packet(buffer);
    packet.set_stream_id(0x87654321);
    using GPSTimestamp = Timestamp<TsiType::gps, TsfType::real_time>;
    auto ts = GPSTimestamp(2000000000, 999999999999ULL);
    packet.set_timestamp(ts);
    auto trailer_cfg = vrtigo::TrailerBuilder{0x12345678};
    trailer_cfg.apply(packet.trailer());
    packet.set_packet_count(15);
    packet.set_payload(payload_data.data(), payload_data.size());

    // Parse with view
    auto result = dynamic::DataPacketView::parse(buffer);

    // Verify all fields
    ASSERT_TRUE(result.ok()) << result.error().message();
    const auto& view = result.value();
    EXPECT_EQ(view.type(), vrtigo::PacketType::signal_data);
    EXPECT_TRUE(view.has_stream_id());
    EXPECT_TRUE(view.has_timestamp());
    EXPECT_TRUE(view.has_trailer());
    EXPECT_EQ(view.packet_count(), 15);

    EXPECT_EQ(*view.stream_id(), 0x87654321);
    auto view_ts = view.timestamp();
    ASSERT_TRUE(view_ts.has_value());
    EXPECT_EQ(view.header().tsi_kind(), TsiType::gps);
    EXPECT_EQ(view.header().tsf_kind(), TsfType::real_time);
    EXPECT_EQ(view_ts->tsi(), 2000000000);
    EXPECT_EQ(view_ts->tsf(), 999999999999ULL);
    EXPECT_EQ(view.trailer()->raw(), 0x12345678);

    // Verify payload
    auto parsed_payload = view.payload();
    ASSERT_EQ(parsed_payload.size(), payload_data.size());
    for (size_t i = 0; i < parsed_payload.size(); i++) {
        EXPECT_EQ(parsed_payload[i], payload_data[i]);
    }
}

// Test that signal packets can now have class IDs
// Data packets (signal and extension) now support optional class ID fields,
// and the parser correctly handles the offset calculations for all subsequent fields
TEST(RtDataPacketTest, ValidationAcceptsClassIdBit) {
    alignas(4) std::array<uint8_t, 64> buffer{};

    // Create a signal packet header with class ID bit set (bit 27)
    uint32_t header = (1U << 28) | // type=1 (signal_data_with_stream)
                      (1U << 27) | // class ID bit SET (now supported!)
                      12;          // size=12 words (header + stream_id + class_id(2) + payload)
    uint32_t header_be = detail::host_to_network32(header);
    std::memcpy(buffer.data(), &header_be, sizeof(header_be));

    // Add a stream ID (required for type-1 packets)
    uint32_t stream_id = 0x12345678;
    uint32_t stream_id_be = detail::host_to_network32(stream_id);
    std::memcpy(buffer.data() + 4, &stream_id_be, sizeof(stream_id_be));

    // Add class ID (2 words = 8 bytes)
    // Word 1: [31:27] PBC=0 | [26:24] Reserved=0 | [23:0] OUI=0x123456
    // Word 2: [31:16] ICC=0x5678 | [15:0] PCC=0xABCD
    uint32_t class_id_word0 = 0x123456U;                 // PBC=0, Reserved=0, OUI=0x123456
    uint32_t class_id_word1 = (0x5678U << 16) | 0xABCDU; // ICC=0x5678, PCC=0xABCD
    uint32_t word0_be = detail::host_to_network32(class_id_word0);
    uint32_t word1_be = detail::host_to_network32(class_id_word1);
    std::memcpy(buffer.data() + 8, &word0_be, sizeof(word0_be));
    std::memcpy(buffer.data() + 12, &word1_be, sizeof(word1_be));

    // Parse packet - should be valid
    auto result = dynamic::DataPacketView::parse(buffer);

    ASSERT_TRUE(result.ok()) << result.error().message();
    const auto& view = result.value();

    // Verify class ID is accessible
    EXPECT_TRUE(view.has_class_id());
    auto class_id = view.class_id();
    ASSERT_TRUE(class_id.has_value());
    EXPECT_EQ(class_id->oui(), 0x123456U);
    EXPECT_EQ(class_id->icc(), 0x5678U);
    EXPECT_EQ(class_id->pcc(), 0xABCDU);
    EXPECT_EQ(class_id->pbc(), 0U);

    // Verify stream ID is accessible
    auto sid = view.stream_id();
    ASSERT_TRUE(sid.has_value());
    EXPECT_EQ(*sid, 0x12345678U);
}

// Test that bit 25 (Nd0) is independent of packet type
// Per VITA 49.2 Table 5.1.1.1-1, bit 25 is "Not a V49.0 Packet Indicator",
// NOT the stream ID indicator. Stream ID presence is determined by packet type only.
TEST(RtDataPacketTest, Bit25IsIndependentOfPacketType) {
    alignas(4) std::array<uint8_t, 64> buffer{};

    // Case 1: Type-0 packet (no stream ID) with bit 25 set (Nd0=1, V49.2 mode)
    // This is VALID - bit 25 just indicates V49.2 features, doesn't affect stream ID
    uint32_t header_type0_with_nd0 = (0U << 28) | // type=0 (no stream ID)
                                     (1U << 25) | // Nd0=1 (V49.2 mode)
                                     10;          // size=10 words
    uint32_t header_be = detail::host_to_network32(header_type0_with_nd0);
    std::memcpy(buffer.data(), &header_be, sizeof(header_be));

    auto result1 = dynamic::DataPacketView::parse(buffer);
    ASSERT_TRUE(result1.ok()) << result1.error().message();
    EXPECT_FALSE(result1.value().has_stream_id()); // No stream ID (type 0)

    // Case 2: Type-1 packet (with stream ID) with bit 25 clear (Nd0=0, V49.0 compatible)
    // This is also VALID - type-1 packets can be V49.0 compatible
    uint32_t header_type1_without_nd0 = (1U << 28) | // type=1 (with stream ID)
                                        (0U << 25) | // Nd0=0 (V49.0 mode)
                                        10;          // size=10 words
    header_be = detail::host_to_network32(header_type1_without_nd0);
    std::memcpy(buffer.data(), &header_be, sizeof(header_be));

    auto result2 = dynamic::DataPacketView::parse(buffer);
    ASSERT_TRUE(result2.ok()) << result2.error().message();
    EXPECT_TRUE(result2.value().has_stream_id()); // Has stream ID (type 1)
}
