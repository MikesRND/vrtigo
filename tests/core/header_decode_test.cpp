#include <gtest/gtest.h>
#include <vrtigo/detail/header_decode.hpp>

using namespace vrtigo;
using namespace vrtigo::detail;

// Test 1: Decode signal packet type 0 (no stream ID)
TEST(HeaderDecodeTest, DecodeSignalPacketType0) {
    // Header: Type=0, no class ID, no trailer, no stream ID, TSI=none, TSF=none, count=5, size=10
    uint32_t header = (0U << 28) | // type = 0
                      (0U << 27) | // no class ID
                      (0U << 26) | // no trailer
                      (0U << 25) | // no stream ID (must be 0 for type 0)
                      (0U << 22) | // TSI = none
                      (0U << 20) | // TSF = none
                      (5U << 16) | // packet count = 5
                      10;          // size = 10 words

    auto decoded = decode_header(header);

    EXPECT_EQ(decoded.type, PacketType::signal_data_no_id);
    EXPECT_EQ(decoded.size_words, 10);
    EXPECT_FALSE(has_stream_id_field(decoded.type)); // Type 0 has no stream ID
    EXPECT_FALSE(decoded.has_class_id);
    EXPECT_FALSE(decoded.trailer_included); // Use type-aware field
    EXPECT_EQ(decoded.tsi, TsiType::none);
    EXPECT_EQ(decoded.tsf, TsfType::none);
    EXPECT_EQ(decoded.packet_count, 5);
}

// Test 2: Decode signal packet type 1 (with stream ID)
TEST(HeaderDecodeTest, DecodeSignalPacketType1) {
    // Header: Type=1, no class ID, no trailer, has stream ID, TSI=GPS, TSF=realtime, count=12,
    // size=512
    uint32_t header = (1U << 28) |  // type = 1
                      (0U << 27) |  // no class ID
                      (0U << 26) |  // no trailer
                      (1U << 25) |  // has stream ID
                      (2U << 22) |  // TSI = GPS
                      (2U << 20) |  // TSF = real time
                      (12U << 16) | // packet count = 12
                      512;          // size = 512 words

    auto decoded = decode_header(header);

    EXPECT_EQ(decoded.type, PacketType::signal_data);
    EXPECT_EQ(decoded.size_words, 512);
    EXPECT_TRUE(has_stream_id_field(decoded.type)); // Type 1 has stream ID
    EXPECT_FALSE(decoded.has_class_id);
    EXPECT_FALSE(decoded.trailer_included); // Use type-aware field
    EXPECT_EQ(decoded.tsi, TsiType::gps);
    EXPECT_EQ(decoded.tsf, TsfType::real_time);
    EXPECT_EQ(decoded.packet_count, 12);
}

// Test 3: Decode context packet
TEST(HeaderDecodeTest, DecodeContextPacket) {
    // Header: Type=5 (context with stream), class ID, no trailer, has stream, TSI=UTC, TSF=sample,
    // count=0, size=20
    uint32_t header = (5U << 28) | // type = 5 (context with stream)
                      (1U << 27) | // has class ID
                      (0U << 26) | // no trailer
                      (1U << 25) | // has stream ID
                      (1U << 22) | // TSI = UTC
                      (1U << 20) | // TSF = sample count
                      (0U << 16) | // packet count = 0
                      20;          // size = 20 words

    auto decoded = decode_header(header);

    EXPECT_EQ(decoded.type, PacketType::extension_context);
    EXPECT_EQ(decoded.size_words, 20);
    EXPECT_TRUE(has_stream_id_field(decoded.type)); // Type 5 has stream ID
    EXPECT_TRUE(decoded.has_class_id);
    // Context packets don't have trailer field - bit 26 is reserved
    EXPECT_FALSE(decoded.trailer_included); // Not applicable to context packets
    EXPECT_EQ(decoded.tsi, TsiType::utc);
    EXPECT_EQ(decoded.tsf, TsfType::sample_count);
    EXPECT_EQ(decoded.packet_count, 0);
}

// Test 4: Decode with class ID
TEST(HeaderDecodeTest, DecodeWithClassID) {
    // Header: Type=1, WITH class ID, no trailer, has stream, TSI=none, TSF=none, count=3, size=100
    uint32_t header = (1U << 28) | // type = 1
                      (1U << 27) | // HAS class ID
                      (0U << 26) | // no trailer
                      (1U << 25) | // has stream ID
                      (0U << 22) | // TSI = none
                      (0U << 20) | // TSF = none
                      (3U << 16) | // packet count = 3
                      100;         // size = 100 words

    auto decoded = decode_header(header);

    EXPECT_EQ(decoded.type, PacketType::signal_data);
    EXPECT_TRUE(decoded.has_class_id);
    EXPECT_FALSE(decoded.trailer_included); // Use type-aware field
    EXPECT_EQ(decoded.size_words, 100);
    EXPECT_EQ(decoded.packet_count, 3);
}

// Test 5: Decode with trailer
TEST(HeaderDecodeTest, DecodeWithTrailer) {
    // Header: Type=1, no class ID, WITH trailer, has stream, TSI=other, TSF=free_run, count=7,
    // size=256
    uint32_t header = (1U << 28) | // type = 1
                      (0U << 27) | // no class ID
                      (1U << 26) | // HAS trailer
                      (1U << 25) | // has stream ID
                      (3U << 22) | // TSI = other
                      (3U << 20) | // TSF = free running
                      (7U << 16) | // packet count = 7
                      256;         // size = 256 words

    auto decoded = decode_header(header);

    EXPECT_EQ(decoded.type, PacketType::signal_data);
    EXPECT_FALSE(decoded.has_class_id);
    EXPECT_TRUE(decoded.trailer_included); // Use type-aware field for signal packets
    EXPECT_EQ(decoded.tsi, TsiType::other);
    EXPECT_EQ(decoded.tsf, TsfType::free_running);
    EXPECT_EQ(decoded.size_words, 256);
    EXPECT_EQ(decoded.packet_count, 7);
}

// Test 6: Decode all timestamp types
TEST(HeaderDecodeTest, DecodeTimestamps) {
    // Test TSI=GPS, TSF=real_time
    uint32_t header1 = (1U << 28) | (1U << 25) | (2U << 22) | (2U << 20) | 10;
    auto decoded1 = decode_header(header1);
    EXPECT_EQ(decoded1.tsi, TsiType::gps);
    EXPECT_EQ(decoded1.tsf, TsfType::real_time);

    // Test TSI=UTC, TSF=sample_count
    uint32_t header2 = (1U << 28) | (1U << 25) | (1U << 22) | (1U << 20) | 10;
    auto decoded2 = decode_header(header2);
    EXPECT_EQ(decoded2.tsi, TsiType::utc);
    EXPECT_EQ(decoded2.tsf, TsfType::sample_count);

    // Test TSI=other, TSF=free_running
    uint32_t header3 = (1U << 28) | (1U << 25) | (3U << 22) | (3U << 20) | 10;
    auto decoded3 = decode_header(header3);
    EXPECT_EQ(decoded3.tsi, TsiType::other);
    EXPECT_EQ(decoded3.tsf, TsfType::free_running);

    // Test TSI=none, TSF=none
    uint32_t header4 = (1U << 28) | (1U << 25) | (0U << 22) | (0U << 20) | 10;
    auto decoded4 = decode_header(header4);
    EXPECT_EQ(decoded4.tsi, TsiType::none);
    EXPECT_EQ(decoded4.tsf, TsfType::none);
}

// Test 7: Decode packet count range
TEST(HeaderDecodeTest, DecodePacketCount) {
    // Test packet count = 0
    uint32_t header0 = (1U << 28) | (1U << 25) | (0U << 16) | 10;
    EXPECT_EQ(decode_header(header0).packet_count, 0);

    // Test packet count = 7
    uint32_t header7 = (1U << 28) | (1U << 25) | (7U << 16) | 10;
    EXPECT_EQ(decode_header(header7).packet_count, 7);

    // Test packet count = 15 (max)
    uint32_t header15 = (1U << 28) | (1U << 25) | (15U << 16) | 10;
    EXPECT_EQ(decode_header(header15).packet_count, 15);
}

// Test 8: Validate packet types
TEST(HeaderDecodeTest, ValidatePacketType) {
    // Valid packet types (0-7)
    EXPECT_TRUE(is_valid_packet_type(PacketType::signal_data_no_id));
    EXPECT_TRUE(is_valid_packet_type(PacketType::signal_data));
    EXPECT_TRUE(is_valid_packet_type(PacketType::extension_data_no_id));
    EXPECT_TRUE(is_valid_packet_type(PacketType::extension_data));
    EXPECT_TRUE(is_valid_packet_type(PacketType::context));
    EXPECT_TRUE(is_valid_packet_type(PacketType::extension_context));
    EXPECT_TRUE(is_valid_packet_type(PacketType::command));
    EXPECT_TRUE(is_valid_packet_type(PacketType::extension_command));

    // Invalid packet types (8-15)
    EXPECT_FALSE(is_valid_packet_type(static_cast<PacketType>(8)));
    EXPECT_FALSE(is_valid_packet_type(static_cast<PacketType>(9)));
    EXPECT_FALSE(is_valid_packet_type(static_cast<PacketType>(15)));

    // Test TSI/TSF validators (all 2-bit values are valid)
    EXPECT_TRUE(is_valid_tsi_type(TsiType::none));
    EXPECT_TRUE(is_valid_tsi_type(TsiType::utc));
    EXPECT_TRUE(is_valid_tsi_type(TsiType::gps));
    EXPECT_TRUE(is_valid_tsi_type(TsiType::other));

    EXPECT_TRUE(is_valid_tsf_type(TsfType::none));
    EXPECT_TRUE(is_valid_tsf_type(TsfType::sample_count));
    EXPECT_TRUE(is_valid_tsf_type(TsfType::real_time));
    EXPECT_TRUE(is_valid_tsf_type(TsfType::free_running));
}
