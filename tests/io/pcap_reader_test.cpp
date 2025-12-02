#include <array>
#include <filesystem>
#include <fstream>
#include <vector>

#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <vrtigo/utils/detail/reader_error.hpp>
#include <vrtigo/vrtigo_utils.hpp>

#include "pcap_test_helpers.hpp"

using namespace vrtigo::utils::pcapio;
using namespace vrtigo::utils::pcapio::test;
using vrtigo::PacketType;
using vrtigo::ValidationError;
using vrtigo::dynamic::DataPacketView;
using DataPacket = vrtigo::dynamic::DataPacketView;

// =============================================================================
// Basic Functionality Tests
// =============================================================================

TEST(PCAPReaderTest, OpenValidPCAPFile) {
    PCAPTestFile test_file("test_pcap_open.pcap");

    // Create PCAP with one VRT packet
    auto vrt_pkt = create_simple_data_packet(0x12345678);
    test_file.create_with_vrt_packets({vrt_pkt});

    // Open with PCAPVRTReader
    PCAPVRTReader<> reader(test_file.path().c_str());

    EXPECT_TRUE(reader.is_open());
    EXPECT_GT(reader.size(), 0);
    EXPECT_EQ(reader.packets_read(), 0);
}

TEST(PCAPReaderTest, ReadSinglePacket) {
    PCAPTestFile test_file("test_pcap_single.pcap");

    // Create PCAP with one VRT packet
    auto vrt_pkt = create_simple_data_packet(0x12345678, 10);
    test_file.create_with_vrt_packets({vrt_pkt});

    // Read packet
    PCAPVRTReader<> reader(test_file.path().c_str());
    auto pkt_result = reader.read_next_packet();

    ASSERT_TRUE(pkt_result.has_value()) << vrtigo::utils::error_message(pkt_result.error());
    EXPECT_EQ(reader.packets_read(), 1);

    // Verify it's a data packet
    auto* data_pkt = std::get_if<DataPacket>(&*pkt_result);
    ASSERT_NE(data_pkt, nullptr);

    // Verify stream ID
    auto sid = data_pkt->stream_id();
    ASSERT_TRUE(sid.has_value());
    EXPECT_EQ(*sid, 0x12345678);
}

TEST(PCAPReaderTest, ReadMultiplePackets) {
    PCAPTestFile test_file("test_pcap_multiple.pcap");

    // Create PCAP with multiple VRT packets
    std::vector<std::vector<uint8_t>> vrt_packets;
    vrt_packets.push_back(create_simple_data_packet(0x11111111, 5));
    vrt_packets.push_back(create_simple_data_packet(0x22222222, 10));
    vrt_packets.push_back(create_simple_data_packet(0x33333333, 15));

    test_file.create_with_vrt_packets(vrt_packets);

    // Read all packets
    PCAPVRTReader<> reader(test_file.path().c_str());

    std::vector<uint32_t> stream_ids;
    while (true) {
        auto pkt_result = reader.read_next_packet();
        if (!pkt_result.has_value()) {
            if (vrtigo::utils::is_eof(pkt_result.error()))
                break;
            continue; // skip errors
        }
        if (auto* data_pkt = std::get_if<DataPacket>(&*pkt_result)) {
            auto sid = data_pkt->stream_id();
            if (sid.has_value()) {
                stream_ids.push_back(*sid);
            }
        }
    }

    ASSERT_EQ(stream_ids.size(), 3);
    EXPECT_EQ(stream_ids[0], 0x11111111);
    EXPECT_EQ(stream_ids[1], 0x22222222);
    EXPECT_EQ(stream_ids[2], 0x33333333);
    EXPECT_EQ(reader.packets_read(), 3);
}

TEST(PCAPReaderTest, RewindAndReread) {
    PCAPTestFile test_file("test_pcap_rewind.pcap");

    // Create PCAP with two VRT packets
    std::vector<std::vector<uint8_t>> vrt_packets;
    vrt_packets.push_back(create_simple_data_packet(0xAAAAAAAA));
    vrt_packets.push_back(create_simple_data_packet(0xBBBBBBBB));

    test_file.create_with_vrt_packets(vrt_packets);

    // Read first packet
    PCAPVRTReader<> reader(test_file.path().c_str());
    auto first = reader.read_next_packet();
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(reader.packets_read(), 1);

    // Rewind
    reader.rewind();
    EXPECT_EQ(reader.packets_read(), 0);

    // Read again
    auto second = reader.read_next_packet();
    ASSERT_TRUE(second.has_value());

    // Verify same packet
    auto* first_data = std::get_if<DataPacket>(&*first);
    auto* second_data = std::get_if<DataPacket>(&*second);
    ASSERT_NE(first_data, nullptr);
    ASSERT_NE(second_data, nullptr);
    EXPECT_EQ(first_data->stream_id().value(), second_data->stream_id().value());
}

TEST(PCAPReaderTest, RawLinkType) {
    PCAPTestFile test_file("test_pcap_raw.pcap");

    // Create PCAP with no link-layer header (raw IP)
    auto vrt_pkt = create_simple_data_packet(0x99999999);
    test_file.create_with_vrt_packets({vrt_pkt}, 0); // 0 = no link-layer header

    // Read with link_header_size = 0
    PCAPVRTReader<> reader(test_file.path().c_str(), 0);
    auto pkt_result = reader.read_next_packet();

    ASSERT_TRUE(pkt_result.has_value());

    auto* data_pkt = std::get_if<DataPacket>(&*pkt_result);
    ASSERT_NE(data_pkt, nullptr);
    EXPECT_EQ(data_pkt->stream_id().value(), 0x99999999);
}

TEST(PCAPReaderTest, ConfigurableLinkHeaderSize) {
    PCAPTestFile test_file("test_pcap_custom_link.pcap");

    // Create PCAP with custom link-layer header size (16 bytes for Linux SLL)
    auto vrt_pkt = create_simple_data_packet(0x88888888);
    test_file.create_with_vrt_packets({vrt_pkt}, 16);

    // Read with link_header_size = 16
    PCAPVRTReader<> reader(test_file.path().c_str(), 16);
    auto pkt_result = reader.read_next_packet();

    ASSERT_TRUE(pkt_result.has_value());

    auto* data_pkt = std::get_if<DataPacket>(&*pkt_result);
    ASSERT_NE(data_pkt, nullptr);
    EXPECT_EQ(data_pkt->stream_id().value(), 0x88888888);
}

// =============================================================================
// Iteration Helpers Tests
// =============================================================================

TEST(PCAPReaderTest, ForEachDataPacket) {
    PCAPTestFile test_file("test_pcap_iteration.pcap");

    // Create PCAP with multiple packets
    std::vector<std::vector<uint8_t>> vrt_packets;
    for (uint32_t i = 0; i < 5; i++) {
        vrt_packets.push_back(create_simple_data_packet(0x1000 + i));
    }
    test_file.create_with_vrt_packets(vrt_packets);

    // Iterate using for_each_data_packet
    PCAPVRTReader<> reader(test_file.path().c_str());

    size_t count = 0;
    reader.for_each_data_packet([&count](const DataPacket& pkt) {
        auto sid = pkt.stream_id();
        if (sid.has_value()) {
            EXPECT_GE(*sid, 0x1000);
            EXPECT_LE(*sid, 0x1004);
        }
        count++;
        return true;
    });

    EXPECT_EQ(count, 5);
}

TEST(PCAPReaderTest, ForEachPacketWithStreamID) {
    PCAPTestFile test_file("test_pcap_stream_filter.pcap");

    // Create PCAP with mixed stream IDs
    std::vector<std::vector<uint8_t>> vrt_packets;
    vrt_packets.push_back(create_simple_data_packet(0xAAAA));
    vrt_packets.push_back(create_simple_data_packet(0xBBBB));
    vrt_packets.push_back(create_simple_data_packet(0xAAAA));
    vrt_packets.push_back(create_simple_data_packet(0xCCCC));
    vrt_packets.push_back(create_simple_data_packet(0xAAAA));

    test_file.create_with_vrt_packets(vrt_packets);

    // Filter by stream ID 0xAAAA
    PCAPVRTReader<> reader(test_file.path().c_str());

    size_t count = 0;
    reader.for_each_packet_with_stream_id(0xAAAA, [&count](const auto& pkt) {
        if (auto* data_pkt = std::get_if<DataPacket>(&pkt)) {
            EXPECT_EQ(data_pkt->stream_id().value(), 0xAAAA);
            count++;
        }
        return true;
    });

    EXPECT_EQ(count, 3); // Should find 3 packets with stream ID 0xAAAA
}

// =============================================================================
// Error Handling Tests
// =============================================================================

TEST(PCAPReaderTest, InvalidMagicNumber) {
    // Create file with invalid magic number
    std::filesystem::path path = "test_pcap_invalid_magic.pcap";
    {
        std::ofstream file(path, std::ios::binary);
        uint32_t bad_magic = 0xDEADBEEF; // Invalid magic
        file.write(reinterpret_cast<const char*>(&bad_magic), 4);
    }

    // Should throw on construction
    EXPECT_THROW({ PCAPVRTReader<> reader(path.c_str()); }, std::runtime_error);

    // Cleanup
    std::filesystem::remove(path);
}

TEST(PCAPReaderTest, ReadsBigEndianPCAP) {
    PCAPTestFile test_file("test_pcap_big_endian.pcap", true);

    std::vector<std::vector<uint8_t>> vrt_packets;
    vrt_packets.push_back(create_simple_data_packet(0xABCD0001, 4));
    vrt_packets.push_back(create_simple_data_packet(0xABCD0002, 6));
    test_file.create_with_vrt_packets(vrt_packets);

    PCAPVRTReader<> reader(test_file.path().c_str());

    size_t count = 0;
    while (true) {
        auto pkt_result = reader.read_next_packet();
        if (!pkt_result.has_value()) {
            if (vrtigo::utils::is_eof(pkt_result.error()))
                break;
            continue; // skip errors
        }
        auto* data_pkt = std::get_if<DataPacket>(&*pkt_result);
        ASSERT_NE(data_pkt, nullptr);
        auto sid = data_pkt->stream_id();
        ASSERT_TRUE(sid.has_value());
        if (count == 0) {
            EXPECT_EQ(*sid, 0xABCD0001);
        } else if (count == 1) {
            EXPECT_EQ(*sid, 0xABCD0002);
        }
        count++;
    }

    EXPECT_EQ(count, 2);
    EXPECT_EQ(reader.packets_read(), 2);
}

TEST(PCAPReaderTest, BigEndianPCAPMovePreservesEndianness) {
    // Test that moving a reader for big-endian file preserves endianness flag
    PCAPTestFile test_file("test_pcap_big_endian_move.pcap", true);

    std::vector<std::vector<uint8_t>> vrt_packets;
    vrt_packets.push_back(create_simple_data_packet(0xDEADBEEF, 5));
    test_file.create_with_vrt_packets(vrt_packets);

    // Create and move reader
    auto create_reader = [&]() -> PCAPVRTReader<> {
        return PCAPVRTReader<>(test_file.path().c_str());
    };

    PCAPVRTReader<> moved_reader = create_reader();

    // Should still correctly read big-endian file after move
    auto pkt_result = moved_reader.read_next_packet();
    ASSERT_TRUE(pkt_result.has_value());

    auto* data_pkt = std::get_if<DataPacket>(&*pkt_result);
    ASSERT_NE(data_pkt, nullptr);
    EXPECT_EQ(data_pkt->stream_id().value(), 0xDEADBEEF);
}

TEST(PCAPReaderTest, EmptyFile) {
    // Create empty file
    std::filesystem::path path = "test_pcap_empty.pcap";
    {
        std::ofstream file(path, std::ios::binary);
        // Empty file
    }

    // Should throw on construction (can't read global header)
    EXPECT_THROW({ PCAPVRTReader<> reader(path.c_str()); }, std::runtime_error);

    // Cleanup
    std::filesystem::remove(path);
}

TEST(PCAPReaderTest, NonExistentFile) {
    // Try to open non-existent file
    EXPECT_THROW({ PCAPVRTReader<> reader("does_not_exist.pcap"); }, std::runtime_error);
}
