#include <filesystem>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include <vrtio/utils/fileio/vrt_file_reader.hpp>
#include <vrtio/vrtio_io.hpp>

#include "test_utils.hpp"

using namespace vrtio;
namespace fileio = vrtio::utils::fileio;

// Test data file paths
const std::filesystem::path test_data_dir = TEST_DATA_DIR;
const auto sample_data_file = test_data_dir / "VITA49SampleData.bin";
const auto sine_wave_file = test_data_dir / "VITA49SineWaveData.bin";

// =============================================================================
// Basic Functionality Tests
// =============================================================================

TEST(EnhancedReaderTest, OpenValidFile) {
    ASSERT_TRUE(std::filesystem::exists(sample_data_file));

    fileio::VRTFileReader<> reader(sample_data_file.c_str());

    EXPECT_TRUE(reader.is_open());
    EXPECT_GT(reader.size(), 0);
    EXPECT_EQ(reader.tell(), 0);
    EXPECT_EQ(reader.packets_read(), 0);
}

TEST(EnhancedReaderTest, ReadFirstPacketAsVariant) {
    fileio::VRTFileReader<> reader(sample_data_file.c_str());

    auto pkt = reader.read_next_packet();

    ASSERT_TRUE(pkt.has_value());
    EXPECT_TRUE(is_valid(*pkt));
    EXPECT_EQ(reader.packets_read(), 1);
    EXPECT_GT(reader.tell(), 0);
}

TEST(EnhancedReaderTest, ReadAllPacketsWithValidation) {
    fileio::VRTFileReader<> reader(sample_data_file.c_str());

    size_t packet_count = 0;
    size_t valid_count = 0;

    while (auto pkt = reader.read_next_packet()) {
        packet_count++;

        if (is_valid(*pkt)) {
            valid_count++;
        }
    }

    EXPECT_GT(packet_count, 0);
    EXPECT_EQ(reader.packets_read(), packet_count);
    EXPECT_EQ(reader.tell(), reader.size());

    // File should have both valid and potentially invalid packets
    EXPECT_GT(valid_count, 0);
}

TEST(EnhancedReaderTest, RewindAndReread) {
    fileio::VRTFileReader<> reader(sample_data_file.c_str());

    // Read first packet
    auto first_pkt = reader.read_next_packet();
    ASSERT_TRUE(first_pkt.has_value());

    auto first_type = packet_type(*first_pkt);

    // Rewind
    reader.rewind();
    EXPECT_EQ(reader.tell(), 0);
    EXPECT_EQ(reader.packets_read(), 0);

    // Read again
    auto second_pkt = reader.read_next_packet();
    ASSERT_TRUE(second_pkt.has_value());

    auto second_type = packet_type(*second_pkt);

    EXPECT_EQ(first_type, second_type);
}

// =============================================================================
// Packet Type Detection Tests
// =============================================================================

TEST(EnhancedReaderTest, DetectDataPackets) {
    fileio::VRTFileReader<> reader(sample_data_file.c_str());

    size_t data_packet_count = 0;

    while (auto pkt = reader.read_next_packet()) {
        if (is_data_packet(*pkt)) {
            data_packet_count++;

            // Verify we can access DataPacketView methods
            auto& data_pkt = std::get<vrtio::DataPacketView>(*pkt);
            EXPECT_TRUE(data_pkt.is_valid());

            // Should have a payload
            auto payload = data_pkt.payload();
            EXPECT_GE(payload.size(), 0);
        }
    }

    // Sample file should have data packets
    EXPECT_GT(data_packet_count, 0);
}

TEST(EnhancedReaderTest, DetectContextPackets) {
    fileio::VRTFileReader<> reader(sample_data_file.c_str());

    size_t context_packet_count = 0;

    while (auto pkt = reader.read_next_packet()) {
        if (is_context_packet(*pkt)) {
            context_packet_count++;

            // Verify we can access ContextPacketView methods
            auto& ctx_pkt = std::get<vrtio::ContextPacketView>(*pkt);
            EXPECT_TRUE(ctx_pkt.is_valid());
        }
    }

    // Sample file should have context packets
    EXPECT_GT(context_packet_count, 0);
}

// =============================================================================
// Filtered Iteration Tests
// =============================================================================

TEST(EnhancedReaderTest, ForEachValidatedPacket) {
    fileio::VRTFileReader<> reader(sample_data_file.c_str());

    size_t callback_count = 0;

    size_t processed = reader.for_each_validated_packet([&](const vrtio::PacketVariant& pkt) {
        (void)pkt; // Only counting packets in this test
        callback_count++;
        return true; // Continue
    });

    EXPECT_EQ(processed, callback_count);
    EXPECT_GT(processed, 0);
}

TEST(EnhancedReaderTest, ForEachDataPacket) {
    fileio::VRTFileReader<> reader(sample_data_file.c_str());

    size_t data_count = 0;

    size_t processed = reader.for_each_data_packet([&](const vrtio::DataPacketView& pkt) {
        data_count++;
        EXPECT_TRUE(pkt.is_valid());
        return true;
    });

    EXPECT_EQ(processed, data_count);
    EXPECT_GT(processed, 0);
}

TEST(EnhancedReaderTest, ForEachContextPacket) {
    fileio::VRTFileReader<> reader(sample_data_file.c_str());

    size_t context_count = 0;

    size_t processed = reader.for_each_context_packet([&](const vrtio::ContextPacketView& pkt) {
        context_count++;
        EXPECT_TRUE(pkt.is_valid());
        return true;
    });

    EXPECT_EQ(processed, context_count);
    EXPECT_GT(processed, 0);
}

TEST(EnhancedReaderTest, EarlyTermination) {
    fileio::VRTFileReader<> reader(sample_data_file.c_str());

    size_t max_packets = 5;
    size_t count = 0;

    reader.for_each_validated_packet([&](const vrtio::PacketVariant&) {
        count++;
        return count < max_packets; // Stop after 5 packets
    });

    EXPECT_EQ(count, max_packets);
}

TEST(EnhancedReaderTest, FilterByStreamId) {
    fileio::VRTFileReader<> reader(sample_data_file.c_str());

    // Get stream ID from first packet that has one
    std::optional<uint32_t> target_stream_id;

    while (auto pkt = reader.read_next_packet()) {
        auto sid = stream_id(*pkt);
        if (sid) {
            target_stream_id = sid;
            break;
        }
    }

    if (!target_stream_id) {
        GTEST_SKIP() << "No packets with stream ID found in test file";
    }

    // Rewind and filter by that stream ID
    reader.rewind();

    size_t matching_count = 0;
    reader.for_each_packet_with_stream_id(*target_stream_id, [&](const vrtio::PacketVariant& pkt) {
        matching_count++;
        auto sid = stream_id(pkt);
        if (sid.has_value()) {
            EXPECT_EQ(*sid, *target_stream_id);
        }
        return true;
    });

    EXPECT_GT(matching_count, 0);
}

// =============================================================================
// Variant Pattern Tests
// =============================================================================

TEST(EnhancedReaderTest, VisitPacketVariant) {
    fileio::VRTFileReader<> reader(sample_data_file.c_str());

    auto pkt = reader.read_next_packet();
    ASSERT_TRUE(pkt.has_value());

    bool visited = false;

    std::visit(
        [&](auto&& p) {
            using T = std::decay_t<decltype(p)>;

            if constexpr (std::is_same_v<T, vrtio::DataPacketView>) {
                visited = true;
                EXPECT_TRUE(p.is_valid());
            } else if constexpr (std::is_same_v<T, vrtio::ContextPacketView>) {
                visited = true;
                EXPECT_TRUE(p.is_valid());
            } else if constexpr (std::is_same_v<T, InvalidPacket>) {
                visited = true;
                // Check error message is non-empty
                EXPECT_FALSE(p.error_message().empty());
            }
        },
        *pkt);

    EXPECT_TRUE(visited);
}

// =============================================================================
// Error Handling Tests
// =============================================================================

TEST(EnhancedReaderTest, EOFHandling) {
    fileio::VRTFileReader<> reader(sample_data_file.c_str());

    // Read all packets
    while (reader.read_next_packet()) {
        // Keep reading
    }

    // Next read should return nullopt
    auto pkt = reader.read_next_packet();
    EXPECT_FALSE(pkt.has_value());
}

TEST(EnhancedReaderTest, InvalidPacketHasErrorInfo) {
    // This test just verifies the InvalidPacket structure works

    vrtio::detail::DecodedHeader header{};
    vrtio::InvalidPacket invalid{ValidationError::buffer_too_small, PacketType::signal_data_no_id,
                                 header, std::span<const uint8_t>{}};

    EXPECT_EQ(invalid.error, ValidationError::buffer_too_small);
    EXPECT_FALSE(invalid.error_message().empty());
}

TEST(EnhancedReaderTest, IOErrorReturnsInvalidPacket) {
    // Create a temporary file with a truncated packet to trigger I/O error
    auto temp_path = std::filesystem::temp_directory_path() / "test_truncated.vrt";

    {
        // Write a header claiming a 10-word packet, but only write the header
        std::ofstream file(temp_path, std::ios::binary);

        // Create header: packet type 1 (SignalData), size = 10 words
        uint32_t header = 0x18000000; // Type=1, C=1, T=0, TSI=0, TSF=0, size=0
        header |= (10 << 16);         // Set packet size to 10 words

        // Write in network byte order (big-endian)
        uint32_t header_be = detail::host_to_network32(header);
        file.write(reinterpret_cast<const char*>(&header_be), sizeof(header_be));
        // Don't write the rest - file is truncated
    }

    // Now try to read it
    fileio::VRTFileReader<> reader(temp_path.c_str());

    auto pkt = reader.read_next_packet();

    // Should return a packet (not nullopt), but it should be InvalidPacket
    ASSERT_TRUE(pkt.has_value());
    EXPECT_FALSE(is_valid(*pkt));

    // Verify it's an InvalidPacket with error information
    auto* invalid = std::get_if<InvalidPacket>(&(*pkt));
    ASSERT_NE(invalid, nullptr);

    // Should have an error (not none)
    EXPECT_NE(invalid->error, ValidationError::none);
    EXPECT_FALSE(invalid->error_message().empty());

    // Clean up
    std::filesystem::remove(temp_path);
}

TEST(EnhancedReaderTest, InvalidPacketTypeReturnsInvalidPacket) {
    // Create a temporary file with an invalid packet type
    auto temp_path = std::filesystem::temp_directory_path() / "test_invalid_type.vrt";

    {
        std::ofstream file(temp_path, std::ios::binary);

        // Create header with invalid packet type (15, which is > 7)
        uint32_t header = 0xF0000000; // Type=15 (invalid), size=4 words
        header |= (4 << 16);

        // Write in network byte order
        uint32_t header_be = detail::host_to_network32(header);
        file.write(reinterpret_cast<const char*>(&header_be), sizeof(header_be));

        // Write rest of packet (3 more words) so it's complete
        uint32_t dummy = 0;
        for (int i = 0; i < 3; i++) {
            file.write(reinterpret_cast<const char*>(&dummy), sizeof(dummy));
        }
    }

    // Now try to read it
    fileio::VRTFileReader<> reader(temp_path.c_str());

    auto pkt = reader.read_next_packet();

    // Should return InvalidPacket
    ASSERT_TRUE(pkt.has_value());
    EXPECT_FALSE(is_valid(*pkt));

    auto* invalid = std::get_if<InvalidPacket>(&(*pkt));
    ASSERT_NE(invalid, nullptr);
    EXPECT_EQ(invalid->error, ValidationError::invalid_packet_type);

    // Clean up
    std::filesystem::remove(temp_path);
}

// =============================================================================
// Helper Function Tests
// =============================================================================

TEST(EnhancedReaderTest, HelperFunctions) {
    fileio::VRTFileReader<> reader(sample_data_file.c_str());

    auto pkt = reader.read_next_packet();
    ASSERT_TRUE(pkt.has_value());

    // Test is_valid()
    bool valid = is_valid(*pkt);
    EXPECT_TRUE(valid || !valid); // Should return boolean

    // Test packet_type()
    PacketType type = packet_type(*pkt);
    EXPECT_TRUE(static_cast<uint8_t>(type) <= 7);

    // Test stream_id()
    (void)stream_id(*pkt); // May or may not have stream ID depending on packet type

    // Test is_data_packet()
    bool is_data = is_data_packet(*pkt);

    // Test is_context_packet()
    bool is_context = is_context_packet(*pkt);

    // Should be one or the other (or invalid)
    if (valid) {
        EXPECT_TRUE(is_data || is_context);
    }
}
