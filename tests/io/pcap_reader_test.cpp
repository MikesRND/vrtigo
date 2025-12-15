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

// =============================================================================
// PCAPVRTReader Raw API Tests (read_next_raw, last_status)
// =============================================================================

TEST(PCAPReaderTest, PCAPVRTReaderReadNextRaw) {
    PCAPTestFile test_file("test_pcap_raw_api.pcap");

    // Create PCAP with multiple VRT packets
    std::vector<std::vector<uint8_t>> vrt_packets;
    vrt_packets.push_back(create_simple_data_packet(0xAABBCCDD, 5));
    vrt_packets.push_back(create_simple_data_packet(0x11223344, 8));
    vrt_packets.push_back(create_simple_data_packet(0x99887766, 3));

    test_file.create_with_vrt_packets(vrt_packets);

    // Read packets using read_next_raw()
    PCAPVRTReader<> reader(test_file.path().c_str());

    size_t count = 0;
    while (true) {
        auto raw_bytes = reader.read_next_raw();
        if (raw_bytes.empty()) {
            break;
        }

        // Verify the raw bytes are valid VRT packets
        ASSERT_GE(raw_bytes.size(), 4) << "VRT packet too small at index " << count;

        // Parse the raw bytes to verify they're valid
        auto parse_result = vrtigo::parse_packet(raw_bytes);
        ASSERT_TRUE(parse_result.has_value()) << "Failed to parse raw packet at index " << count;

        // Verify it's a data packet with correct stream ID
        auto* data_pkt = std::get_if<DataPacket>(&*parse_result);
        ASSERT_NE(data_pkt, nullptr) << "Not a data packet at index " << count;

        auto sid = data_pkt->stream_id();
        ASSERT_TRUE(sid.has_value()) << "No stream ID at index " << count;

        // Check expected stream IDs
        if (count == 0) {
            EXPECT_EQ(*sid, 0xAABBCCDD);
        } else if (count == 1) {
            EXPECT_EQ(*sid, 0x11223344);
        } else if (count == 2) {
            EXPECT_EQ(*sid, 0x99887766);
        }

        count++;
    }

    EXPECT_EQ(count, 3);
    EXPECT_EQ(reader.packets_read(), 3);
}

TEST(PCAPReaderTest, PCAPVRTReaderLastStatusOnSuccess) {
    PCAPTestFile test_file("test_pcap_last_status_success.pcap");

    // Create PCAP with one VRT packet
    auto vrt_pkt = create_simple_data_packet(0x12345678, 10);
    test_file.create_with_vrt_packets({vrt_pkt});

    // Read packet
    PCAPVRTReader<> reader(test_file.path().c_str());

    // Initially, status should be ok
    EXPECT_EQ(reader.last_status(), PCAPReadStatus::ok);

    // Read the packet
    auto raw_bytes = reader.read_next_raw();
    ASSERT_FALSE(raw_bytes.empty());

    // After successful read, last_status() should be ok
    EXPECT_EQ(reader.last_status(), PCAPReadStatus::ok);

    // Verify the packet is valid
    auto parse_result = vrtigo::parse_packet(raw_bytes);
    ASSERT_TRUE(parse_result.has_value());
    auto* data_pkt = std::get_if<DataPacket>(&*parse_result);
    ASSERT_NE(data_pkt, nullptr);
    EXPECT_EQ(data_pkt->stream_id().value(), 0x12345678);
}

TEST(PCAPReaderTest, PCAPVRTReaderLastStatusOnEof) {
    PCAPTestFile test_file("test_pcap_last_status_eof.pcap");

    // Create PCAP with two VRT packets
    std::vector<std::vector<uint8_t>> vrt_packets;
    vrt_packets.push_back(create_simple_data_packet(0xAAAAAAAA, 5));
    vrt_packets.push_back(create_simple_data_packet(0xBBBBBBBB, 5));

    test_file.create_with_vrt_packets(vrt_packets);

    // Read all packets
    PCAPVRTReader<> reader(test_file.path().c_str());

    // Read first packet
    auto raw1 = reader.read_next_raw();
    ASSERT_FALSE(raw1.empty());
    EXPECT_EQ(reader.last_status(), PCAPReadStatus::ok);

    // Read second packet
    auto raw2 = reader.read_next_raw();
    ASSERT_FALSE(raw2.empty());
    EXPECT_EQ(reader.last_status(), PCAPReadStatus::ok);

    // Try to read third packet (should hit EOF)
    auto raw3 = reader.read_next_raw();
    EXPECT_TRUE(raw3.empty());

    // After EOF, last_status() should be eof
    EXPECT_EQ(reader.last_status(), PCAPReadStatus::eof);

    // Additional reads should continue to report EOF
    auto raw4 = reader.read_next_raw();
    EXPECT_TRUE(raw4.empty());
    EXPECT_EQ(reader.last_status(), PCAPReadStatus::eof);
}

TEST(PCAPReaderTest, PCAPVRTReaderRawThenParsed) {
    PCAPTestFile test_file("test_pcap_raw_then_parsed.pcap");

    // Create PCAP with multiple VRT packets
    std::vector<std::vector<uint8_t>> vrt_packets;
    vrt_packets.push_back(create_simple_data_packet(0x11111111, 5));
    vrt_packets.push_back(create_simple_data_packet(0x22222222, 8));
    vrt_packets.push_back(create_simple_data_packet(0x33333333, 3));
    vrt_packets.push_back(create_simple_data_packet(0x44444444, 10));

    test_file.create_with_vrt_packets(vrt_packets);

    // Read using mix of raw and parsed API
    PCAPVRTReader<> reader(test_file.path().c_str());

    // Read first packet using raw API
    auto raw1 = reader.read_next_raw();
    ASSERT_FALSE(raw1.empty());
    EXPECT_EQ(reader.last_status(), PCAPReadStatus::ok);
    auto parse1 = vrtigo::parse_packet(raw1);
    ASSERT_TRUE(parse1.has_value());
    auto* data1 = std::get_if<DataPacket>(&*parse1);
    ASSERT_NE(data1, nullptr);
    EXPECT_EQ(data1->stream_id().value(), 0x11111111);

    // Read second packet using parsed API
    auto pkt2 = reader.read_next_packet();
    ASSERT_TRUE(pkt2.has_value());
    auto* data2 = std::get_if<DataPacket>(&*pkt2);
    ASSERT_NE(data2, nullptr);
    EXPECT_EQ(data2->stream_id().value(), 0x22222222);

    // Read third packet using raw API
    auto raw3 = reader.read_next_raw();
    ASSERT_FALSE(raw3.empty());
    EXPECT_EQ(reader.last_status(), PCAPReadStatus::ok);
    auto parse3 = vrtigo::parse_packet(raw3);
    ASSERT_TRUE(parse3.has_value());
    auto* data3 = std::get_if<DataPacket>(&*parse3);
    ASSERT_NE(data3, nullptr);
    EXPECT_EQ(data3->stream_id().value(), 0x33333333);

    // Read fourth packet using parsed API
    auto pkt4 = reader.read_next_packet();
    ASSERT_TRUE(pkt4.has_value());
    auto* data4 = std::get_if<DataPacket>(&*pkt4);
    ASSERT_NE(data4, nullptr);
    EXPECT_EQ(data4->stream_id().value(), 0x44444444);

    // Verify we've read all packets
    EXPECT_EQ(reader.packets_read(), 4);

    // Next read should hit EOF
    auto eof_raw = reader.read_next_raw();
    EXPECT_TRUE(eof_raw.empty());
    EXPECT_EQ(reader.last_status(), PCAPReadStatus::eof);
}
