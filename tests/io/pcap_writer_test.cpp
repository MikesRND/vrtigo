#include <array>
#include <filesystem>
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
        ASSERT_TRUE(pkt_result.has_value());

        bool result = writer.write_packet(pkt_result.value());
        EXPECT_TRUE(result);
        EXPECT_EQ(writer.packets_written(), 1);

        writer.flush();
    }

    // Verify file size:
    // 24 bytes: PCAP global header
    // 16 bytes: PCAP packet record header
    // 42 bytes: Ethernet + IPv4 + UDP headers
    // 48 bytes: VRT packet (2 words header + stream ID, 10 words payload = 12 words * 4)
    size_t expected_min = 24 + 16 + 42 + (2 + 10) * 4;
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
            ASSERT_TRUE(pkt_result.has_value());
            EXPECT_TRUE(writer.write_packet(pkt_result.value()));
        }

        EXPECT_EQ(writer.packets_written(), 5);
    }

    // Verify file exists
    EXPECT_TRUE(std::filesystem::exists(path));

    // Cleanup
    std::filesystem::remove(path);
}

TEST(PCAPWriterTest, CustomPorts) {
    std::filesystem::path path = "test_pcap_write_custom_ports.pcap";

    // Write with custom UDP ports
    {
        PCAPVRTWriter writer(path.c_str(), 5000, 60000);

        auto vrt_pkt = create_simple_data_packet(0x99999999);
        auto pkt_result = parse_test_packet(vrt_pkt);
        ASSERT_TRUE(pkt_result.has_value());

        EXPECT_TRUE(writer.write_packet(pkt_result.value()));
        EXPECT_EQ(writer.dst_port(), 5000);
        EXPECT_EQ(writer.src_port(), 60000);
    }

    // Verify file exists
    EXPECT_TRUE(std::filesystem::exists(path));

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
        ASSERT_TRUE(pkt_result.has_value());

        EXPECT_TRUE(writer.write_packet(*pkt_result));
    }

    // Read it back
    {
        PCAPVRTReader<> reader(path.c_str());

        auto pkt_result = reader.read_next_packet();
        ASSERT_TRUE(pkt_result.has_value());

        auto* data_pkt = std::get_if<DataPacket>(&*pkt_result);
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
            ASSERT_TRUE(pkt_result.has_value());
            EXPECT_TRUE(writer.write_packet(*pkt_result));
        }
    }

    // Read them back
    {
        PCAPVRTReader<> reader(path.c_str());

        std::vector<uint32_t> read_ids;
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

TEST(PCAPWriterTest, RoundTripCustomPorts) {
    std::filesystem::path path = "test_pcap_roundtrip_custom.pcap";

    // Write with custom ports
    uint32_t expected_stream_id = 0x88888888;
    uint16_t expected_dst_port = 5000;
    uint16_t expected_src_port = 60000;
    {
        PCAPVRTWriter writer(path.c_str(), expected_dst_port, expected_src_port);

        auto vrt_pkt = create_simple_data_packet(expected_stream_id);
        auto pkt_result = parse_test_packet(vrt_pkt);
        ASSERT_TRUE(pkt_result.has_value());

        EXPECT_TRUE(writer.write_packet(*pkt_result));
    }

    // Read back and verify ports are extracted
    {
        PCAPVRTReader<> reader(path.c_str());

        auto pkt_result = reader.read_next_packet();
        ASSERT_TRUE(pkt_result.has_value());

        auto* data_pkt = std::get_if<DataPacket>(&*pkt_result);
        ASSERT_NE(data_pkt, nullptr);

        EXPECT_EQ(data_pkt->stream_id().value(), expected_stream_id);

        // Verify reader extracted correct ports
        EXPECT_EQ(reader.last_src_port(), expected_src_port);
        EXPECT_EQ(reader.last_dst_port(), expected_dst_port);
    }

    // Cleanup
    std::filesystem::remove(path);
}

TEST(PCAPWriterTest, RoundTripIPAddresses) {
    std::filesystem::path path = "test_pcap_roundtrip_ips.pcap";

    // Write with custom IPs
    std::string expected_src_ip = "192.168.10.20";
    std::string expected_dst_ip = "192.168.10.30";
    {
        PCAPVRTWriter writer(path.c_str(), 4991, 50000, expected_src_ip, expected_dst_ip);

        auto vrt_pkt = create_simple_data_packet(0x12345678);
        auto pkt_result = parse_test_packet(vrt_pkt);
        ASSERT_TRUE(pkt_result.has_value());

        EXPECT_TRUE(writer.write_packet(*pkt_result));
    }

    // Read back and verify IPs
    {
        PCAPVRTReader<> reader(path.c_str());

        auto pkt_result = reader.read_next_packet();
        ASSERT_TRUE(pkt_result.has_value());

        // Verify IPs match (in network byte order)
        EXPECT_EQ(reader.last_src_ip(), *parse_ipv4(expected_src_ip));
        EXPECT_EQ(reader.last_dst_ip(), *parse_ipv4(expected_dst_ip));
    }

    // Cleanup
    std::filesystem::remove(path);
}

TEST(PCAPWriterTest, ReaderTimestampExtraction) {
    std::filesystem::path path = "test_pcap_timestamp.pcap";

    // Write a packet
    {
        PCAPVRTWriter writer(path.c_str());

        auto vrt_pkt = create_simple_data_packet(0xAAAABBBB);
        auto pkt_result = parse_test_packet(vrt_pkt);
        ASSERT_TRUE(pkt_result.has_value());

        EXPECT_TRUE(writer.write_packet(*pkt_result));
    }

    // Read back and verify timestamp is non-zero
    {
        PCAPVRTReader<> reader(path.c_str());

        // Verify microsecond precision (default PCAP format)
        EXPECT_FALSE(reader.is_nanosecond_precision());

        auto pkt_result = reader.read_next_packet();
        ASSERT_TRUE(pkt_result.has_value());

        // Timestamp should be set (non-zero for any reasonable time)
        EXPECT_GT(reader.last_timestamp_sec(), 0u);
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

TEST(PCAPWriterTest, InvalidIPAddressRejected) {
    std::filesystem::path path = "test_pcap_invalid_ip.pcap";

    // Try to create writer with invalid IP address
    EXPECT_THROW(
        { PCAPVRTWriter writer(path.c_str(), 4991, 50000, "invalid.ip.address"); },
        std::invalid_argument);

    // Verify file was not created
    if (std::filesystem::exists(path)) {
        std::filesystem::remove(path);
    }
}

TEST(PCAPWriterTest, CustomIPAddresses) {
    std::filesystem::path path = "test_pcap_custom_ip.pcap";

    // Create writer with custom IP addresses
    {
        PCAPVRTWriter writer(path.c_str(), 4991, 50000, "192.168.1.100", "192.168.1.200");
        EXPECT_TRUE(writer.is_open());

        // Write a packet to verify it works
        auto vrt_pkt = create_simple_data_packet(0x12345678);
        auto pkt_result = parse_test_packet(vrt_pkt);
        ASSERT_TRUE(pkt_result.has_value());
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
            ASSERT_TRUE(pkt_result.has_value());
            vrt_writer.write_packet(*pkt_result);
        }
    }

    // Convert VRT to PCAP
    {
        vrtigo::VRTFileReader<> reader(vrt_path.c_str());
        PCAPVRTWriter writer(pcap_path.c_str());

        while (true) {
            auto pkt_result = reader.read_next_packet();
            if (!pkt_result.has_value()) {
                if (vrtigo::utils::is_eof(pkt_result.error()))
                    break;
                continue; // skip errors
            }
            writer.write_packet(*pkt_result);
        }
    }

    // Verify PCAP file
    {
        PCAPVRTReader<> reader(pcap_path.c_str());
        EXPECT_EQ(reader.packets_read(), 0);

        size_t count = 0;
        while (true) {
            auto pkt_result = reader.read_next_packet();
            if (!pkt_result.has_value()) {
                if (vrtigo::utils::is_eof(pkt_result.error()))
                    break;
                continue; // skip errors
            }
            count++;
        }
        EXPECT_EQ(count, 3);
    }

    // Cleanup
    std::filesystem::remove(vrt_path);
    std::filesystem::remove(pcap_path);
}

// =============================================================================
// PCAPVRTWriter Raw Bytes API Tests
// =============================================================================

TEST(PCAPVRTWriterRawBytesOverload, WriteRawPacketBytes) {
    std::filesystem::path path = "test_pcap_raw_bytes.pcap";

    // Create raw VRT packet bytes manually
    auto raw_packet = create_simple_data_packet(0xDEADBEEF, 15);

    // Write using raw bytes overload (no timestamp - extracts from VRT)
    {
        PCAPVRTWriter writer(path.c_str());

        std::span<const uint8_t> packet_span(raw_packet.data(), raw_packet.size());
        bool result = writer.write_packet(packet_span);
        EXPECT_TRUE(result);
        EXPECT_EQ(writer.packets_written(), 1);

        writer.flush();
    }

    // Read back and verify packet was written correctly
    {
        PCAPVRTReader<> reader(path.c_str());

        auto pkt_result = reader.read_next_packet();
        ASSERT_TRUE(pkt_result.has_value());

        auto* data_pkt = std::get_if<DataPacket>(&*pkt_result);
        ASSERT_NE(data_pkt, nullptr);

        auto sid = data_pkt->stream_id();
        ASSERT_TRUE(sid.has_value());
        EXPECT_EQ(*sid, 0xDEADBEEF);

        // Verify payload size matches
        EXPECT_EQ(data_pkt->payload_size_words(), 15);
    }

    // Cleanup
    std::filesystem::remove(path);
}

TEST(PCAPVRTWriterRawBytesOverload, WriteMultipleRawPackets) {
    std::filesystem::path path = "test_pcap_raw_bytes_multiple.pcap";

    // Create multiple raw VRT packets
    std::vector<std::vector<uint8_t>> raw_packets;
    std::vector<uint32_t> expected_stream_ids = {0x11111111, 0x22222222, 0x33333333, 0x44444444};

    for (auto sid : expected_stream_ids) {
        raw_packets.push_back(create_simple_data_packet(sid, 8));
    }

    // Write using raw bytes overload
    {
        PCAPVRTWriter writer(path.c_str());

        for (const auto& packet : raw_packets) {
            std::span<const uint8_t> packet_span(packet.data(), packet.size());
            EXPECT_TRUE(writer.write_packet(packet_span));
        }

        EXPECT_EQ(writer.packets_written(), raw_packets.size());
        writer.flush();
    }

    // Read back and verify all packets
    {
        PCAPVRTReader<> reader(path.c_str());

        std::vector<uint32_t> read_stream_ids;
        while (true) {
            auto pkt_result = reader.read_next_packet();
            if (!pkt_result.has_value()) {
                if (vrtigo::utils::is_eof(pkt_result.error()))
                    break;
                continue;
            }

            if (auto* data_pkt = std::get_if<DataPacket>(&*pkt_result)) {
                auto sid = data_pkt->stream_id();
                if (sid.has_value()) {
                    read_stream_ids.push_back(*sid);
                }
            }
        }

        ASSERT_EQ(read_stream_ids.size(), expected_stream_ids.size());
        for (size_t i = 0; i < expected_stream_ids.size(); i++) {
            EXPECT_EQ(read_stream_ids[i], expected_stream_ids[i]);
        }
    }

    // Cleanup
    std::filesystem::remove(path);
}

TEST(PCAPVRTWriterRawBytesOverload, WriteWithExplicitTimestamp) {
    std::filesystem::path path = "test_pcap_raw_bytes_timestamp.pcap";

    // Create raw VRT packet bytes
    auto raw_packet = create_simple_data_packet(0xCAFEBABE, 20);

    // Write using raw bytes overload with explicit timestamp
    uint32_t expected_ts_sec = 1234567890;
    uint32_t expected_ts_usec = 500000; // 0.5 seconds
    {
        PCAPVRTWriter writer(path.c_str());

        std::span<const uint8_t> packet_span(raw_packet.data(), raw_packet.size());
        bool result = writer.write_packet(packet_span, expected_ts_sec, expected_ts_usec);
        EXPECT_TRUE(result);
        EXPECT_EQ(writer.packets_written(), 1);

        writer.flush();
    }

    // Read back and verify timestamp was preserved
    {
        PCAPVRTReader<> reader(path.c_str());

        auto pkt_result = reader.read_next_packet();
        ASSERT_TRUE(pkt_result.has_value());

        auto* data_pkt = std::get_if<DataPacket>(&*pkt_result);
        ASSERT_NE(data_pkt, nullptr);

        // Verify stream ID
        auto sid = data_pkt->stream_id();
        ASSERT_TRUE(sid.has_value());
        EXPECT_EQ(*sid, 0xCAFEBABE);

        // Verify timestamp was written correctly
        EXPECT_EQ(reader.last_timestamp_sec(), expected_ts_sec);
        EXPECT_EQ(reader.last_timestamp_subsec(), expected_ts_usec);
    }

    // Cleanup
    std::filesystem::remove(path);
}

TEST(PCAPVRTWriterRawBytesOverload, WriteInvalidPacketTooSmall) {
    std::filesystem::path path = "test_pcap_raw_bytes_invalid.pcap";

    // Create invalid packet (too small - less than 4 bytes)
    std::vector<uint8_t> invalid_packet = {0x01, 0x02};

    {
        PCAPVRTWriter writer(path.c_str());

        std::span<const uint8_t> packet_span(invalid_packet.data(), invalid_packet.size());
        bool result = writer.write_packet(packet_span);
        EXPECT_FALSE(result); // Should fail - packet too small
        EXPECT_EQ(writer.packets_written(), 0);
    }

    // Cleanup
    std::filesystem::remove(path);
}

TEST(PCAPVRTWriterRawBytesOverload, RawBytesWithCustomPorts) {
    std::filesystem::path path = "test_pcap_raw_bytes_custom_ports.pcap";

    // Create raw VRT packet
    auto raw_packet = create_simple_data_packet(0xAAAABBBB, 12);

    // Write with custom ports
    uint16_t custom_dst_port = 6000;
    uint16_t custom_src_port = 55555;
    {
        PCAPVRTWriter writer(path.c_str(), custom_dst_port, custom_src_port);

        std::span<const uint8_t> packet_span(raw_packet.data(), raw_packet.size());
        EXPECT_TRUE(writer.write_packet(packet_span));

        EXPECT_EQ(writer.dst_port(), custom_dst_port);
        EXPECT_EQ(writer.src_port(), custom_src_port);

        writer.flush();
    }

    // Read back and verify ports were preserved
    {
        PCAPVRTReader<> reader(path.c_str());

        auto pkt_result = reader.read_next_packet();
        ASSERT_TRUE(pkt_result.has_value());

        auto* data_pkt = std::get_if<DataPacket>(&*pkt_result);
        ASSERT_NE(data_pkt, nullptr);

        EXPECT_EQ(reader.last_src_port(), custom_src_port);
        EXPECT_EQ(reader.last_dst_port(), custom_dst_port);
    }

    // Cleanup
    std::filesystem::remove(path);
}

TEST(PCAPVRTWriterRawBytesOverload, RawBytesSequentialTimestamps) {
    std::filesystem::path path = "test_pcap_raw_bytes_sequential_ts.pcap";

    // Create several raw packets with sequential timestamps
    std::vector<uint32_t> stream_ids = {0x1000, 0x2000, 0x3000};
    std::vector<uint32_t> timestamps_sec = {1000000000, 1000000001, 1000000002};
    std::vector<uint32_t> timestamps_usec = {100000, 200000, 300000};

    // Write packets with explicit timestamps
    {
        PCAPVRTWriter writer(path.c_str());

        for (size_t i = 0; i < stream_ids.size(); i++) {
            auto raw_packet = create_simple_data_packet(stream_ids[i], 5);
            std::span<const uint8_t> packet_span(raw_packet.data(), raw_packet.size());
            EXPECT_TRUE(writer.write_packet(packet_span, timestamps_sec[i], timestamps_usec[i]));
        }

        EXPECT_EQ(writer.packets_written(), stream_ids.size());
        writer.flush();
    }

    // Read back and verify timestamps are correct and sequential
    {
        PCAPVRTReader<> reader(path.c_str());

        for (size_t i = 0; i < stream_ids.size(); i++) {
            auto pkt_result = reader.read_next_packet();
            ASSERT_TRUE(pkt_result.has_value()) << "Failed to read packet " << i;

            auto* data_pkt = std::get_if<DataPacket>(&*pkt_result);
            ASSERT_NE(data_pkt, nullptr);

            // Verify stream ID matches
            auto sid = data_pkt->stream_id();
            ASSERT_TRUE(sid.has_value());
            EXPECT_EQ(*sid, stream_ids[i]);

            // Verify timestamp matches
            EXPECT_EQ(reader.last_timestamp_sec(), timestamps_sec[i]);
            EXPECT_EQ(reader.last_timestamp_subsec(), timestamps_usec[i]);
        }
    }

    // Cleanup
    std::filesystem::remove(path);
}
