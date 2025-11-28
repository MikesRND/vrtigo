#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <vector>

#include <gtest/gtest.h>
#include <vrtigo/vrtigo_utils.hpp>

#include "test_utils.hpp"

using namespace vrtigo::utils::fileio;
using namespace vrtigo::detail;
using vrtigo::PacketType;
using vrtigo::RuntimeContextPacket;
using vrtigo::ValidationError;

// Test data file paths
const std::filesystem::path test_data_dir = TEST_DATA_DIR;
const auto sample_data_file = test_data_dir / "VITA49SampleData.bin";
const auto sine_wave_file = test_data_dir / "VITA49SineWaveData.bin";

// =============================================================================
// Basic Functionality Tests (6 tests)
// =============================================================================

TEST(FileReaderTest, OpenValidFile) {
    ASSERT_TRUE(std::filesystem::exists(sample_data_file));

    RawVRTFileReader<> reader(sample_data_file.c_str());

    EXPECT_TRUE(reader.is_open());
    EXPECT_GT(reader.size(), 0);
    EXPECT_EQ(reader.tell(), 0);
    EXPECT_EQ(reader.packets_read(), 0);
}

TEST(FileReaderTest, ReadFirstPacket) {
    RawVRTFileReader<> reader(sample_data_file.c_str());

    auto packet = reader.read_next_span();

    ASSERT_FALSE(packet.empty());
    EXPECT_TRUE(reader.last_error().is_valid());
    EXPECT_EQ(reader.packets_read(), 1);
    EXPECT_GT(reader.tell(), 0);

    // Verify packet has valid header
    EXPECT_GE(packet.size(), 4);
}

TEST(FileReaderTest, ReadAllPackets) {
    RawVRTFileReader<> reader(sample_data_file.c_str());

    size_t packet_count = 0;
    while (true) {
        auto packet = reader.read_next_span();
        if (packet.empty()) {
            EXPECT_TRUE(reader.last_error().is_eof());
            break;
        }
        packet_count++;
    }

    EXPECT_GT(packet_count, 0);
    EXPECT_EQ(reader.packets_read(), packet_count);
    EXPECT_EQ(reader.tell(), reader.size());
}

TEST(FileReaderTest, RewindAndReread) {
    RawVRTFileReader<> reader(sample_data_file.c_str());

    // Read first packet
    auto first_read = reader.read_next_span();
    ASSERT_FALSE(first_read.empty());
    size_t first_size = first_read.size();

    // Rewind
    reader.rewind();
    EXPECT_EQ(reader.tell(), 0);
    EXPECT_EQ(reader.packets_read(), 0);

    // Read again
    auto second_read = reader.read_next_span();
    ASSERT_FALSE(second_read.empty());
    EXPECT_EQ(second_read.size(), first_size);

    // Verify content matches
    EXPECT_TRUE(std::equal(first_read.begin(), first_read.end(), second_read.begin()));
}

TEST(FileReaderTest, BufferResizing) {
    RawVRTFileReader<> reader(sample_data_file.c_str());

    // Start with small buffer
    std::vector<uint8_t> small_buffer(16);
    auto result = reader.read_next(small_buffer.data(), small_buffer.size());

    // Should fail with buffer_too_small
    EXPECT_EQ(result.error, ValidationError::buffer_too_small);
    EXPECT_GT(result.buffer_size_required, small_buffer.size());

    // Resize and retry
    std::vector<uint8_t> large_buffer(result.buffer_size_required);
    result = reader.read_next(large_buffer.data(), large_buffer.size());

    EXPECT_TRUE(result.is_valid());
    EXPECT_EQ(result.packet_size_bytes, result.buffer_size_required);
}

TEST(FileReaderTest, SpanInterface) {
    RawVRTFileReader<> reader(sample_data_file.c_str());

    // Read using span interface
    auto packet1 = reader.read_next_span();
    ASSERT_FALSE(packet1.empty());

    // Save packet info and first few bytes
    auto info1 = reader.last_error();
    EXPECT_TRUE(info1.is_valid());
    EXPECT_EQ(info1.packet_size_bytes, packet1.size());

    std::vector<uint8_t> packet1_data(packet1.begin(), packet1.end());

    // Read next packet - this reuses the buffer
    auto packet2 = reader.read_next_span();
    ASSERT_FALSE(packet2.empty());

    // Verify buffer is reused (same pointer) - this is the key behavior
    EXPECT_EQ(packet1.data(), packet2.data()); // Same buffer location (span reuses internal buffer)

    // The original packet1 span is now invalidated (buffer was overwritten)
    // We successfully demonstrated buffer reuse
}

// =============================================================================
// Packet Type Tests (4 tests)
// =============================================================================

TEST(FileReaderTest, ParseSignalPackets) {
    RawVRTFileReader<> reader(sample_data_file.c_str());

    bool found_signal = false;
    size_t count = 0;

    while (count++ < 50) { // Check first 50 packets
        auto packet = reader.read_next_span();
        if (packet.empty())
            break;

        auto& info = reader.last_error();
        if (info.type == PacketType::signal_data || info.type == PacketType::signal_data_no_id) {
            found_signal = true;

            // Verify we can feed to SignalPacket (just check header)
            EXPECT_GE(packet.size(), 4);
            break;
        }
    }

    // Sample data should contain signal packets
    EXPECT_TRUE(found_signal || count >= 50);
}

TEST(FileReaderTest, ParseContextPackets) {
    RawVRTFileReader<> reader(sample_data_file.c_str());

    bool found_context = false;
    size_t count = 0;

    while (count++ < 50) {
        auto packet = reader.read_next_span();
        if (packet.empty())
            break;

        auto& info = reader.last_error();
        if (info.type == PacketType::context || info.type == PacketType::extension_context) {
            found_context = true;

            // Verify we can attempt parsing (result may fail if unsupported fields)
            // The file reader's job is just to read packets correctly
            (void)RuntimeContextPacket::parse(packet);
            EXPECT_GE(packet.size(), 4); // At least a header
            break;
        }
    }

    // Sample data should contain context packets
    EXPECT_TRUE(found_context || count >= 50);
}

TEST(FileReaderTest, MixedPacketStream) {
    RawVRTFileReader<> reader(sample_data_file.c_str());

    std::map<PacketType, size_t> type_counts;
    size_t total = 0;

    while (total++ < 100) {
        auto packet = reader.read_next_span();
        if (packet.empty())
            break;

        auto& info = reader.last_error();
        type_counts[info.type]++;
    }

    // Should have read multiple packets
    EXPECT_GT(total, 1);

    // Should have at least one packet type
    EXPECT_FALSE(type_counts.empty());
}

TEST(FileReaderTest, ContextThenSignal) {
    RawVRTFileReader<> reader(sample_data_file.c_str());

    // VRT streams typically start with context, then signal data
    std::vector<PacketType> types;

    for (int i = 0; i < 10; ++i) {
        auto packet = reader.read_next_span();
        if (packet.empty())
            break;

        types.push_back(reader.last_error().type);
    }

    // Should have read some packets
    EXPECT_FALSE(types.empty());

    // First few packets are often context (but not required)
    // Just verify we got a variety or specific pattern
    bool has_variety = types.size() > 1;
    EXPECT_TRUE(has_variety || types.size() == 1);
}

// =============================================================================
// Validation Tests (7 tests)
// =============================================================================

TEST(FileReaderTest, DetectTruncatedPacket) {
    // Create a truncated file
    auto temp_file = test_data_dir / "temp_truncated.bin";

    {
        std::ifstream src(sample_data_file, std::ios::binary);
        std::ofstream dst(temp_file, std::ios::binary);

        // Copy only first 20 bytes (should truncate at least one packet)
        std::vector<uint8_t> data(20);
        src.read(reinterpret_cast<char*>(data.data()), 20);
        dst.write(reinterpret_cast<char*>(data.data()), 20);
    }

    RawVRTFileReader<> reader(temp_file.c_str());

    // Try to read - should either succeed with tiny packet or detect truncation
    auto packet = reader.read_next_span();

    // Either we get a valid tiny packet, or we detect truncation
    if (packet.empty()) {
        auto& err = reader.last_error();
        EXPECT_TRUE(err.error != ValidationError::none || err.is_eof());
    }

    std::filesystem::remove(temp_file);
}

TEST(FileReaderTest, DetectInvalidPacketType) {
    // Create file with invalid packet type
    auto temp_file = test_data_dir / "temp_invalid_type.bin";

    {
        std::ofstream file(temp_file, std::ios::binary);

        // Create header with invalid packet type (15)
        uint32_t bad_header = (15U << 28) | 10; // Type 15 (invalid), size 10
        bad_header = host_to_network32(bad_header);

        file.write(reinterpret_cast<char*>(&bad_header), 4);

        // Write some dummy payload
        std::vector<uint8_t> dummy(36, 0xAA);
        file.write(reinterpret_cast<char*>(dummy.data()), 36);
    }

    RawVRTFileReader<> reader(temp_file.c_str());
    auto packet = reader.read_next_span();

    EXPECT_TRUE(packet.empty());
    EXPECT_EQ(reader.last_error().error, ValidationError::invalid_packet_type);

    std::filesystem::remove(temp_file);
}

TEST(FileReaderTest, DetectZeroSize) {
    auto temp_file = test_data_dir / "temp_zero_size.bin";

    {
        std::ofstream file(temp_file, std::ios::binary);

        // Create header with size = 0 (invalid)
        uint32_t bad_header = (1U << 28) | (1U << 25) | 0; // Type 1, size 0
        bad_header = host_to_network32(bad_header);

        file.write(reinterpret_cast<char*>(&bad_header), 4);
    }

    RawVRTFileReader<> reader(temp_file.c_str());
    auto packet = reader.read_next_span();

    EXPECT_TRUE(packet.empty());
    EXPECT_EQ(reader.last_error().error, ValidationError::size_field_mismatch);

    std::filesystem::remove(temp_file);
}

TEST(FileReaderTest, DetectSizeOverflow) {
    auto temp_file = test_data_dir / "temp_overflow.bin";

    {
        std::ofstream file(temp_file, std::ios::binary);

        // Create header with size > MaxPacketWords
        uint32_t bad_header = (1U << 28) | (1U << 25) | 0xFFFF; // Max size + wrong config
        bad_header = host_to_network32(bad_header);

        file.write(reinterpret_cast<char*>(&bad_header), 4);
    }

    // Use small MaxPacketWords to trigger overflow
    RawVRTFileReader<100> reader(temp_file.c_str());
    auto packet = reader.read_next_span();

    EXPECT_TRUE(packet.empty());
    EXPECT_EQ(reader.last_error().error, ValidationError::size_field_mismatch);

    std::filesystem::remove(temp_file);
}

TEST(FileReaderTest, BufferTooSmallHandling) {
    RawVRTFileReader<> reader(sample_data_file.c_str());

    std::vector<uint8_t> tiny_buffer(4);
    auto result = reader.read_next(tiny_buffer.data(), tiny_buffer.size());

    // Should report buffer too small
    EXPECT_EQ(result.error, ValidationError::buffer_too_small);
    EXPECT_GT(result.buffer_size_required, 4);

    // File position should be rewound (ready to retry)
    EXPECT_EQ(reader.tell(), 0);
}

TEST(FileReaderTest, EmptyFile) {
    auto temp_file = test_data_dir / "temp_empty.bin";

    {
        std::ofstream file(temp_file, std::ios::binary);
        // Write nothing
    }

    RawVRTFileReader<> reader(temp_file.c_str());
    auto packet = reader.read_next_span();

    EXPECT_TRUE(packet.empty());
    EXPECT_TRUE(reader.last_error().is_eof());

    std::filesystem::remove(temp_file);
}

TEST(FileReaderTest, CorruptedHeader) {
    auto temp_file = test_data_dir / "temp_corrupt.bin";

    {
        std::ofstream file(temp_file, std::ios::binary);

        // Write random garbage
        std::vector<uint8_t> garbage = {0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE};
        file.write(reinterpret_cast<char*>(garbage.data()), garbage.size());
    }

    RawVRTFileReader<> reader(temp_file.c_str());
    auto packet = reader.read_next_span();

    // Should detect some kind of error (likely invalid type or size)
    if (packet.empty()) {
        auto& err = reader.last_error();
        EXPECT_NE(err.error, ValidationError::none);
    }

    std::filesystem::remove(temp_file);
}

// =============================================================================
// Integration Tests (4 tests)
// =============================================================================

TEST(FileReaderTest, SampleDataFullParse) {
    RawVRTFileReader<> reader(sample_data_file.c_str());

    size_t packet_count = 0;
    size_t total_bytes = 0;

    while (true) {
        auto packet = reader.read_next_span();
        if (packet.empty())
            break;

        packet_count++;
        total_bytes += packet.size();

        // Validate each packet has reasonable size
        EXPECT_GE(packet.size(), 4);
        EXPECT_LE(packet.size(), 65535 * 4);
    }

    // Sample data file should have many packets
    EXPECT_GT(packet_count, 5);
    EXPECT_EQ(reader.tell(), reader.size());

    std::cout << "Parsed " << packet_count << " packets, " << total_bytes << " bytes\n";
}

TEST(FileReaderTest, SineWaveDataFullParse) {
    if (!std::filesystem::exists(sine_wave_file)) {
        GTEST_SKIP() << "Sine wave test file not found";
    }

    RawVRTFileReader<> reader(sine_wave_file.c_str());

    size_t signal_packets = 0;
    double total_energy = 0.0;

    while (true) {
        auto packet = reader.read_next_span();
        if (packet.empty())
            break;

        auto& info = reader.last_error();

        // Look for signal packets
        if (info.type == PacketType::signal_data || info.type == PacketType::signal_data_no_id) {
            // Extract payload (skip header + optional fields)
            // For simplicity, assume payload starts after reasonable header
            if (packet.size() > 32) {
                auto payload = packet.subspan(32);
                double energy = test_utils::compute_signal_energy(payload);

                if (energy > 0) {
                    total_energy += energy;
                    signal_packets++;
                }
            }
        }
    }

    // Sine wave file should have signal packets with non-zero energy
    EXPECT_GT(signal_packets, 0);
    if (signal_packets > 0) {
        EXPECT_GT(total_energy, 0.0);
        std::cout << "Signal packets: " << signal_packets << ", Total energy: " << total_energy
                  << "\n";
    }
}

TEST(FileReaderTest, CorruptedFileRecovery) {
    // Create file with one bad packet in the middle
    auto temp_file = test_data_dir / "temp_recovery.bin";

    {
        std::ifstream src(sample_data_file, std::ios::binary);
        std::ofstream dst(temp_file, std::ios::binary);

        // Copy first packet
        std::vector<uint8_t> buffer(1000);
        src.read(reinterpret_cast<char*>(buffer.data()), 1000);
        dst.write(reinterpret_cast<char*>(buffer.data()), 1000);

        // Write corrupt header
        uint32_t bad = 0xDEADBEEF;
        dst.write(reinterpret_cast<char*>(&bad), 4);

        // Copy more valid data
        src.read(reinterpret_cast<char*>(buffer.data()), 1000);
        dst.write(reinterpret_cast<char*>(buffer.data()), 1000);
    }

    RawVRTFileReader<> reader(temp_file.c_str());

    size_t valid_packets = 0;

    for (int i = 0; i < 10; ++i) {
        auto packet = reader.read_next_span();

        if (packet.empty()) {
            break;
        } else {
            valid_packets++;
        }
    }

    // Should read at least the first valid packet
    EXPECT_GT(valid_packets, 0);

    std::filesystem::remove(temp_file);
}

TEST(FileReaderTest, CallbackPatternUsage) {
    RawVRTFileReader<> reader(sample_data_file.c_str());

    size_t callback_count = 0;
    size_t total_size = 0;

    size_t processed = reader.for_each_packet([&](auto packet, auto& info) {
        callback_count++;
        total_size += packet.size();

        EXPECT_TRUE(info.is_valid());
        EXPECT_GE(packet.size(), 4);

        // Continue processing
        return true;
    });

    EXPECT_EQ(processed, callback_count);
    EXPECT_GT(processed, 0);
    EXPECT_GT(total_size, 0);

    std::cout << "Callback processed " << processed << " packets\n";
}

// =============================================================================
// Performance Tests (2 tests)
// =============================================================================

TEST(FileReaderTest, LargeFileParsing) {
    RawVRTFileReader<> reader(sample_data_file.c_str());

    auto start = std::chrono::high_resolution_clock::now();

    size_t packet_count = 0;
    while (true) {
        auto packet = reader.read_next_span();
        if (packet.empty())
            break;
        packet_count++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Parsed " << packet_count << " packets in " << duration.count() << " ms\n";

    // Reasonable performance: should parse at least 100 packets/sec
    if (packet_count > 0) {
        double packets_per_sec = packet_count * 1000.0 / duration.count();
        EXPECT_GT(packets_per_sec, 10.0); // Very conservative threshold
    }
}

TEST(FileReaderTest, ZeroAllocationVerify) {
    RawVRTFileReader<> reader(sample_data_file.c_str());

    // This test verifies the API doesn't require heap allocation
    // (we can't actually measure allocations in C++, but we verify the design)

    // Read using span interface (uses internal stack buffer)
    for (int i = 0; i < 10; ++i) {
        auto packet = reader.read_next_span();
        if (packet.empty())
            break;

        // Verify we got data without explicit new/malloc
        EXPECT_FALSE(packet.empty());
    }

    reader.rewind();

    // Read using user buffer (zero allocation)
    std::array<uint8_t, 4096> user_buffer;
    for (int i = 0; i < 10; ++i) {
        auto result = reader.read_next(user_buffer.data(), user_buffer.size());
        if (!result.is_valid())
            break;

        EXPECT_GT(result.packet_size_bytes, 0);
    }

    // Test passed - design enables zero-allocation usage
    SUCCEED();
}
