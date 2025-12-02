// Copyright (c) 2025 Michael Smith
// SPDX-License-Identifier: MIT

#include <array>
#include <filesystem>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include <vrtigo/utils/detail/reader_error.hpp>
#include <vrtigo/vrtigo_utils.hpp>

// PacketVariant and related types are now in vrtigo namespace
using namespace vrtigo::utils::fileio;

// Import specific types from vrtigo namespace to avoid ambiguity
using vrtigo::NoClassId;
using vrtigo::NoTimestamp;
using vrtigo::NoTrailer;
using vrtigo::PacketVariant;
using vrtigo::UtcRealTimestamp;
using vrtigo::typed::SignalDataPacketBuilder;

// Test fixture for file writer tests
class FileWriterTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temp directory for test outputs
        temp_dir_ = std::filesystem::temp_directory_path() / "vrtigo_writer_test";
        std::filesystem::create_directories(temp_dir_);
    }

    void TearDown() override {
        // Clean up test files
        if (std::filesystem::exists(temp_dir_)) {
            std::filesystem::remove_all(temp_dir_);
        }
    }

    std::filesystem::path temp_dir_;
};

// =============================================================================
// Basic Functionality Tests
// =============================================================================

TEST_F(FileWriterTest, CreateWriter) {
    auto test_file = temp_dir_ / "test_create.vrt";

    VRTFileWriter<> writer(test_file.string());

    EXPECT_TRUE(writer.is_open());
    EXPECT_EQ(writer.packets_written(), 0);
    EXPECT_EQ(writer.bytes_written(), 0);
    EXPECT_EQ(writer.status(), WriterStatus::ready);
}

TEST_F(FileWriterTest, WriteCompileTimePacket) {
    auto test_file = temp_dir_ / "test_compile_time.vrt";

    // Create a simple data packet using correct API
    using PacketType = vrtigo::typed::SignalDataPacketBuilder<256, UtcRealTimestamp>;
    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    PacketType packet(buffer);
    packet.set_stream_id(0x12345678);
    packet.set_timestamp(UtcRealTimestamp::now());
    packet.set_packet_count(1);

    VRTFileWriter<> writer(test_file.string());
    EXPECT_TRUE(writer.write_packet(packet));
    EXPECT_EQ(writer.packets_written(), 1);
    EXPECT_GT(writer.bytes_written(), 0);

    writer.flush();

    // Verify file exists and has content
    EXPECT_TRUE(std::filesystem::exists(test_file));
    EXPECT_GT(std::filesystem::file_size(test_file), 0);
}

TEST_F(FileWriterTest, WriteMultiplePackets) {
    auto test_file = temp_dir_ / "test_multiple.vrt";

    using PacketType = vrtigo::typed::SignalDataPacketBuilder<64>;
    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    VRTFileWriter<> writer(test_file.string());

    // Write 10 packets
    for (uint32_t i = 0; i < 10; i++) {
        PacketType packet(buffer);
        packet.set_stream_id(i);
        packet.set_packet_count(static_cast<uint8_t>(i));
        EXPECT_TRUE(writer.write_packet(packet));
    }

    EXPECT_EQ(writer.packets_written(), 10);
    writer.flush();
}

TEST_F(FileWriterTest, FlushBufferedData) {
    auto test_file = temp_dir_ / "test_flush.vrt";

    using PacketType = vrtigo::typed::SignalDataPacketBuilder<64>;
    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    PacketType packet(buffer);
    packet.set_stream_id(0x1234);
    packet.set_packet_count(1);

    VRTFileWriter<> writer(test_file.string());
    writer.write_packet(packet);

    // Data may be buffered
    EXPECT_TRUE(writer.flush());
    EXPECT_EQ(writer.status(), WriterStatus::ready);

    // After flush, file should have all data
    size_t size_after_flush = std::filesystem::file_size(test_file);
    EXPECT_GT(size_after_flush, 0);
}

// =============================================================================
// Round-Trip Tests (Write then Read)
// =============================================================================

TEST_F(FileWriterTest, RoundTripSinglePacket) {
    auto test_file = temp_dir_ / "test_roundtrip_single.vrt";

    // Write packet
    using PacketType = vrtigo::typed::SignalDataPacketBuilder<256, UtcRealTimestamp>;
    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    const uint32_t test_stream_id = 0xABCDEF01;
    auto test_timestamp = UtcRealTimestamp::now();

    PacketType write_packet(buffer);
    write_packet.set_stream_id(test_stream_id);
    write_packet.set_timestamp(test_timestamp);
    write_packet.set_packet_count(1);

    {
        VRTFileWriter<> writer(test_file.string());
        EXPECT_TRUE(writer.write_packet(write_packet));
        writer.flush();
    }

    // Read packet back
    VRTFileReader<> reader(test_file.string());
    auto read_packet_var = reader.read_next_packet();

    ASSERT_TRUE(read_packet_var.has_value())
        << vrtigo::utils::error_message(read_packet_var.error());
    EXPECT_TRUE(vrtigo::is_data_packet(*read_packet_var));

    auto read_packet = std::get<vrtigo::dynamic::DataPacketView>(*read_packet_var);
    EXPECT_EQ(read_packet.stream_id(), test_stream_id);

    auto ts = read_packet.timestamp();
    ASSERT_TRUE(ts.has_value());
    EXPECT_EQ(ts->tsi(), test_timestamp.tsi());
    EXPECT_EQ(ts->tsf(), test_timestamp.tsf());
}

TEST_F(FileWriterTest, RoundTripMultiplePackets) {
    auto test_file = temp_dir_ / "test_roundtrip_multi.vrt";

    const size_t num_packets = 100;
    using PacketType = vrtigo::typed::SignalDataPacketBuilder<64>;
    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    // Write packets
    {
        VRTFileWriter<> writer(test_file.string());
        for (size_t i = 0; i < num_packets; i++) {
            PacketType packet(buffer);
            packet.set_stream_id(static_cast<uint32_t>(i));
            packet.set_packet_count(static_cast<uint8_t>(i & 0xFF));
            EXPECT_TRUE(writer.write_packet(packet));
        }
        writer.flush();
    }

    // Read packets back
    VRTFileReader<> reader(test_file.string());
    size_t count = 0;

    while (true) {
        auto pkt_var = reader.read_next_packet();
        if (!pkt_var.has_value()) {
            if (vrtigo::utils::is_eof(pkt_var.error()))
                break;
            continue; // skip errors
        }
        EXPECT_TRUE(vrtigo::is_data_packet(*pkt_var));

        auto pkt = std::get<vrtigo::dynamic::DataPacketView>(*pkt_var);
        EXPECT_EQ(pkt.stream_id(), count);
        count++;
    }

    EXPECT_EQ(count, num_packets);
}

TEST_F(FileWriterTest, RoundTripContextPacket) {
    auto test_file = temp_dir_ / "test_roundtrip_context.vrt";

    // Write context packet
    using PacketType = vrtigo::typed::ContextPacketBuilder<
        NoTimestamp, NoClassId, vrtigo::field::reference_point_id, vrtigo::field::bandwidth>;
    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    const uint32_t test_stream_id = 0x87654321;
    const uint32_t test_ref_point = 0x12345678;

    PacketType write_packet(buffer);
    write_packet.set_stream_id(test_stream_id);
    write_packet[vrtigo::field::reference_point_id].set_encoded(test_ref_point);
    write_packet[vrtigo::field::bandwidth].set_value(1000000.0); // 1 MHz

    {
        VRTFileWriter<> writer(test_file.string());
        EXPECT_TRUE(writer.write_packet(write_packet));
        writer.flush();
    }

    // Read packet back
    VRTFileReader<> reader(test_file.string());
    auto read_packet_var = reader.read_next_packet();

    ASSERT_TRUE(read_packet_var.has_value())
        << vrtigo::utils::error_message(read_packet_var.error());
    EXPECT_TRUE(vrtigo::is_context_packet(*read_packet_var));

    auto read_packet = std::get<vrtigo::dynamic::ContextPacketView>(*read_packet_var);
    EXPECT_EQ(read_packet.stream_id(), test_stream_id);
}

// Note: InvalidPacket handling test removed - InvalidPacket is no longer part of PacketVariant.
// Parse errors are now represented as ParseResult<PacketVariant> with error() method.

// =============================================================================
// Large Buffer Tests
// =============================================================================

TEST_F(FileWriterTest, WriteLargePacket) {
    auto test_file = temp_dir_ / "test_large.vrt";

    // Create packet with large payload (exceeds default buffer size)
    const size_t payload_words = 32 * 1024; // 128 KB
    using PacketType = vrtigo::typed::SignalDataPacketBuilder<payload_words>;
    std::vector<uint8_t> large_buffer(PacketType::size_bytes());

    std::vector<uint8_t> payload(payload_words * 4, 0xAA);
    std::span<uint8_t, PacketType::size_bytes()> buffer_span(large_buffer.data(),
                                                             PacketType::size_bytes());
    PacketType packet(buffer_span);
    packet.set_stream_id(0x99999999);
    packet.set_packet_count(1);
    packet.set_payload(payload.data(), payload.size());

    VRTFileWriter<> writer(test_file.string());
    EXPECT_TRUE(writer.write_packet(packet));
    writer.flush();

    // Verify file size
    EXPECT_GT(std::filesystem::file_size(test_file), payload.size());
}

// =============================================================================
// RawVRTFileWriter Tests
// =============================================================================

TEST_F(FileWriterTest, RawWriterBasic) {
    auto test_file = temp_dir_ / "test_raw.vrt";

    RawVRTFileWriter<> raw_writer(test_file.string());

    // Create minimal VRT packet (header only)
    std::array<uint8_t, 4> minimal_packet = {0x00, 0x00, 0x00, 0x01}; // 1 word packet

    EXPECT_TRUE(raw_writer.write_packet(minimal_packet.data(), minimal_packet.size()));
    EXPECT_EQ(raw_writer.packets_written(), 1);

    raw_writer.flush();
}

TEST_F(FileWriterTest, RawWriterInvalidSize) {
    auto test_file = temp_dir_ / "test_raw_invalid.vrt";

    RawVRTFileWriter<> raw_writer(test_file.string());

    // Try to write packet with invalid size (not multiple of 4)
    std::array<uint8_t, 5> bad_packet = {0x00, 0x00, 0x00, 0x01, 0xFF};

    EXPECT_FALSE(raw_writer.write_packet(bad_packet.data(), bad_packet.size()));
    EXPECT_EQ(raw_writer.packets_written(), 0);
}

TEST_F(FileWriterTest, RawWriterErrorState) {
    auto test_file = temp_dir_ / "test_raw_error.vrt";

    RawVRTFileWriter<> raw_writer(test_file.string());

    std::array<uint8_t, 4> packet = {0x00, 0x00, 0x00, 0x01};
    EXPECT_TRUE(raw_writer.write_packet(packet.data(), packet.size()));

    EXPECT_FALSE(raw_writer.has_error());
    EXPECT_EQ(raw_writer.last_errno(), 0);

    // After successful write, error state should be clear
    raw_writer.clear_error();
    EXPECT_FALSE(raw_writer.has_error());
}
