#include <array>
#include <filesystem>
#include <vector>

#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <vrtigo/vrtigo_utils.hpp>

#include "pcap_test_helpers.hpp"

using namespace vrtigo::utils::pcapio;
using namespace vrtigo::utils::pcapio::test;
using vrtigo::PacketType;
using vrtigo::dynamic::DataPacketView;
using DataPacket = vrtigo::dynamic::DataPacketView;

// =============================================================================
// Basic Functionality Tests
// =============================================================================

TEST(PCAPWriterTest, CreateAndClose) {
    std::filesystem::path path = "test_pcap_write_create.pcap";

    {
        PCAPVRTWriter writer(path.c_str());
        EXPECT_TRUE(writer.is_open());
        EXPECT_EQ(writer.packets_written(), 0);
    }

    // Verify file exists and has global header
    EXPECT_TRUE(std::filesystem::exists(path));
    EXPECT_GE(std::filesystem::file_size(path), 24); // At least global header

    // Cleanup
    std::filesystem::remove(path);
}

TEST(PCAPWriterTest, WriteSinglePacket) {
    std::filesystem::path path = "test_pcap_write_single.pcap";

    // Create and write packet
    {
        PCAPVRTWriter writer(path.c_str());

        auto vrt_pkt = create_simple_data_packet(0x12345678, 10);
        auto pkt_result = parse_test_packet(vrt_pkt);
        ASSERT_TRUE(pkt_result.ok());

        bool result = writer.write_packet(pkt_result.value());
        EXPECT_TRUE(result);
        EXPECT_EQ(writer.packets_written(), 1);

        writer.flush();
    }

    // Verify file size:
    // 24 bytes: PCAP global header
    // 16 bytes: PCAP packet record header
    // 14 bytes: Ethernet link-layer header
    // 48 bytes: VRT packet (2 words header + stream ID, 10 words payload = 12 words * 4)
    size_t expected_min = 24 + 16 + 14 + (2 + 10) * 4;
    EXPECT_GE(std::filesystem::file_size(path), expected_min);

    // Cleanup
    std::filesystem::remove(path);
}

TEST(PCAPWriterTest, WriteMultiplePackets) {
    std::filesystem::path path = "test_pcap_write_multiple.pcap";

    // Write multiple packets
    {
        PCAPVRTWriter writer(path.c_str());

        for (uint32_t i = 0; i < 5; i++) {
            auto vrt_pkt = create_simple_data_packet(0x1000 + i, 5);
            auto pkt_result = parse_test_packet(vrt_pkt);
            ASSERT_TRUE(pkt_result.ok());
            EXPECT_TRUE(writer.write_packet(pkt_result.value()));
        }

        EXPECT_EQ(writer.packets_written(), 5);
    }

    // Verify file exists
    EXPECT_TRUE(std::filesystem::exists(path));

    // Cleanup
    std::filesystem::remove(path);
}

TEST(PCAPWriterTest, RawLinkType) {
    std::filesystem::path path = "test_pcap_write_raw.pcap";

    // Write with no link-layer header
    {
        PCAPVRTWriter writer(path.c_str(), 0);

        auto vrt_pkt = create_simple_data_packet(0x99999999);
        auto pkt_result = parse_test_packet(vrt_pkt);
        ASSERT_TRUE(pkt_result.ok());

        EXPECT_TRUE(writer.write_packet(pkt_result.value()));
        EXPECT_EQ(writer.link_header_size(), 0);
    }

    // Verify smaller file size (no link-layer header):
    // 24 bytes: PCAP global header
    // 16 bytes: PCAP packet record header
    // 48 bytes: VRT packet (2 + 10 words = 12 words * 4 bytes)
    size_t expected_min = 24 + 16 + (2 + 10) * 4;
    EXPECT_GE(std::filesystem::file_size(path), expected_min);

    // Cleanup
    std::filesystem::remove(path);
}

// =============================================================================
// Round-Trip Tests (Write then Read)
// =============================================================================

TEST(PCAPWriterTest, RoundTripSinglePacket) {
    std::filesystem::path path = "test_pcap_roundtrip_single.pcap";

    // Write packet
    uint32_t expected_stream_id = 0xABCDEF01;
    {
        PCAPVRTWriter writer(path.c_str());

        auto vrt_pkt = create_simple_data_packet(expected_stream_id);
        auto pkt_result = parse_test_packet(vrt_pkt);
        ASSERT_TRUE(pkt_result.ok());

        EXPECT_TRUE(writer.write_packet(pkt_result.value()));
    }

    // Read it back
    {
        PCAPVRTReader<> reader(path.c_str());

        auto pkt_result = reader.read_next_packet();
        ASSERT_TRUE(pkt_result.has_value());
        ASSERT_TRUE(pkt_result->ok());

        auto* data_pkt = std::get_if<DataPacket>(&pkt_result->value());
        ASSERT_NE(data_pkt, nullptr);

        auto sid = data_pkt->stream_id();
        ASSERT_TRUE(sid.has_value());
        EXPECT_EQ(*sid, expected_stream_id);
    }

    // Cleanup
    std::filesystem::remove(path);
}

TEST(PCAPWriterTest, RoundTripMultiplePackets) {
    std::filesystem::path path = "test_pcap_roundtrip_multiple.pcap";

    // Write packets
    std::vector<uint32_t> expected_ids = {0x11111111, 0x22222222, 0x33333333};
    {
        PCAPVRTWriter writer(path.c_str());

        for (auto sid : expected_ids) {
            auto vrt_pkt = create_simple_data_packet(sid);
            auto pkt_result = parse_test_packet(vrt_pkt);
            ASSERT_TRUE(pkt_result.ok());
            EXPECT_TRUE(writer.write_packet(pkt_result.value()));
        }
    }

    // Read them back
    {
        PCAPVRTReader<> reader(path.c_str());

        std::vector<uint32_t> read_ids;
        while (auto pkt_result = reader.read_next_packet()) {
            if (!pkt_result->ok())
                continue;
            if (auto* data_pkt = std::get_if<DataPacket>(&pkt_result->value())) {
                auto sid = data_pkt->stream_id();
                if (sid.has_value()) {
                    read_ids.push_back(*sid);
                }
            }
        }

        ASSERT_EQ(read_ids.size(), expected_ids.size());
        for (size_t i = 0; i < expected_ids.size(); i++) {
            EXPECT_EQ(read_ids[i], expected_ids[i]);
        }
    }

    // Cleanup
    std::filesystem::remove(path);
}

TEST(PCAPWriterTest, RoundTripRawLinkType) {
    std::filesystem::path path = "test_pcap_roundtrip_raw.pcap";

    // Write with no link header
    uint32_t expected_stream_id = 0x88888888;
    {
        PCAPVRTWriter writer(path.c_str(), 0);

        auto vrt_pkt = create_simple_data_packet(expected_stream_id);
        auto pkt_result = parse_test_packet(vrt_pkt);
        ASSERT_TRUE(pkt_result.ok());

        EXPECT_TRUE(writer.write_packet(pkt_result.value()));
    }

    // Read with no link header
    {
        PCAPVRTReader<> reader(path.c_str(), 0);

        auto pkt_result = reader.read_next_packet();
        ASSERT_TRUE(pkt_result.has_value());
        ASSERT_TRUE(pkt_result->ok());

        auto* data_pkt = std::get_if<DataPacket>(&pkt_result->value());
        ASSERT_NE(data_pkt, nullptr);

        EXPECT_EQ(data_pkt->stream_id().value(), expected_stream_id);
    }

    // Cleanup
    std::filesystem::remove(path);
}

// =============================================================================
// Error Handling Tests
// =============================================================================

// Note: InvalidPacket was removed from PacketVariant.
// Parse errors are now returned via ParseResult<PacketVariant>.error()
// Writers only accept PacketVariant which contains valid packets.

TEST(PCAPWriterTest, FileCreationError) {
    // Try to create file in non-existent directory
    EXPECT_THROW({ PCAPVRTWriter writer("/nonexistent/directory/test.pcap"); }, std::runtime_error);
}

TEST(PCAPWriterTest, OversizedLinkHeaderRejected) {
    std::filesystem::path path = "test_pcap_oversized_header.pcap";

    // Try to create writer with link header size exceeding maximum
    EXPECT_THROW(
        { PCAPVRTWriter writer(path.c_str(), MAX_LINK_HEADER_SIZE + 1); }, std::invalid_argument);

    // Verify file was not created (constructor should fail before opening file)
    // Note: file might exist if open() succeeded before validation - this tests current behavior
    if (std::filesystem::exists(path)) {
        std::filesystem::remove(path);
    }
}

TEST(PCAPWriterTest, MaxLinkHeaderAccepted) {
    std::filesystem::path path = "test_pcap_max_header.pcap";

    // Create writer with maximum allowed link header size
    {
        PCAPVRTWriter writer(path.c_str(), MAX_LINK_HEADER_SIZE);
        EXPECT_TRUE(writer.is_open());
        EXPECT_EQ(writer.link_header_size(), MAX_LINK_HEADER_SIZE);

        // Write a packet to verify it works
        auto vrt_pkt = create_simple_data_packet(0x12345678);
        auto pkt_result = parse_test_packet(vrt_pkt);
        ASSERT_TRUE(pkt_result.ok());
        EXPECT_TRUE(writer.write_packet(pkt_result.value()));
    }

    // Cleanup
    std::filesystem::remove(path);
}

// =============================================================================
// Integration Test: Convert VRT file to PCAP
// =============================================================================

TEST(PCAPWriterTest, ConvertVRTFileToPCAP) {
    std::filesystem::path vrt_path = "test_convert_source.vrt";
    std::filesystem::path pcap_path = "test_convert_output.pcap";

    // Create source VRT file
    {
        vrtigo::VRTFileWriter<> vrt_writer(vrt_path.c_str());

        for (uint32_t i = 0; i < 3; i++) {
            auto vrt_pkt = create_simple_data_packet(0x2000 + i);
            auto pkt_result = parse_test_packet(vrt_pkt);
            ASSERT_TRUE(pkt_result.ok());
            vrt_writer.write_packet(pkt_result.value());
        }
    }

    // Convert VRT to PCAP
    {
        vrtigo::VRTFileReader<> reader(vrt_path.c_str());
        PCAPVRTWriter writer(pcap_path.c_str());

        while (auto pkt_result = reader.read_next_packet()) {
            if (pkt_result->ok()) {
                writer.write_packet(pkt_result->value());
            }
        }
    }

    // Verify PCAP file
    {
        PCAPVRTReader<> reader(pcap_path.c_str());
        EXPECT_EQ(reader.packets_read(), 0);

        size_t count = 0;
        while (auto pkt_result = reader.read_next_packet()) {
            if (pkt_result->ok())
                count++;
        }
        EXPECT_EQ(count, 3);
    }

    // Cleanup
    std::filesystem::remove(vrt_path);
    std::filesystem::remove(pcap_path);
}
