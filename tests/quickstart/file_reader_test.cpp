// Quickstart Test: Read VRT File
//
// This test demonstrates reading a VRT file with the high-level
// VRTFileReader API. It shows type-safe packet iteration with
// automatic validation and elegant visitor pattern usage.
//
// The code between snippet markers can be extracted for documentation.

#include <iostream>

#include <gtest/gtest.h>
#include <vrtio/vrtio_io.hpp>

// Test data file paths
#include <filesystem>

using vrtio::field::sample_rate;

const std::filesystem::path test_data_dir = TEST_DATA_DIR;
const auto sine_wave_file = test_data_dir / "VITA49SineWaveData.bin";

TEST(QuickstartSnippet, ReadVRTFile) {
    // [QUICKSTART-DESC]
    // This example demonstrates reading a VRT file with the high-level reader:
    // - Automatic packet validation
    // - Type-safe variant access (RuntimeDataPacket, RuntimeContextPacket)
    // - Elegant iteration with for_each helpers
    // - Zero-copy access to packet data
    // [/QUICKSTART-DESC]

    // Skip if test file doesn't exist
    if (!std::filesystem::exists(sine_wave_file)) {
        GTEST_SKIP() << "Sine wave test file not found";
    }

    // [QUICKSTART]
    // Read VRT packets from a file

    // Open VRT file - that's it!
    vrtio::VRTFileReader<> reader(sine_wave_file.c_str());
    ASSERT_TRUE(reader.is_open());

    // Count packets and samples
    size_t data_packets = 0;
    size_t context_packets = 0;
    size_t total_samples = 0;

    // Iterate through all valid packets
    reader.for_each_validated_packet([&](const vrtio::PacketVariant& pkt) {
        if (vrtio::is_data_packet(pkt)) {
            data_packets++;

            // Access type-safe data packet view
            const auto& data = std::get<vrtio::RuntimeDataPacket>(pkt);

            // Get payload - zero-copy span into file buffer!
            auto payload = data.payload();
            total_samples += payload.size() / 4; // 4 bytes per I/Q sample

        } else if (vrtio::is_context_packet(pkt)) {
            context_packets++;

            // Access context packet view
            const auto& ctx = std::get<vrtio::RuntimeContextPacket>(pkt);
            if (auto sr = ctx[sample_rate]) {
                std::cout << "Context packet sample rate: " << sr.value() << " Hz\n";
            }
        }

        return true; // Continue processing
    });
    // [/QUICKSTART]

    // Verification
    EXPECT_GT(data_packets, 0) << "Should have read some data packets";
    EXPECT_GT(total_samples, 0) << "Should have extracted samples";

    std::cout << "Read " << data_packets << " data packets, " << context_packets
              << " context packets, " << total_samples << " samples total\n";
}

TEST(QuickstartSnippet, ReadVRTFileManual) {
    // [QUICKSTART-DESC-MANUAL]
    // This example shows manual packet iteration for more control.
    // Use this when you need to handle invalid packets or implement
    // custom processing logic.
    // [/QUICKSTART-DESC-MANUAL]

    // Skip if test file doesn't exist
    if (!std::filesystem::exists(sine_wave_file)) {
        GTEST_SKIP() << "Sine wave test file not found";
    }

    // [QUICKSTART-MANUAL]
    // Manual packet iteration with full control

    vrtio::VRTFileReader<> reader(sine_wave_file.c_str());
    ASSERT_TRUE(reader.is_open());

    size_t valid_packets = 0;
    size_t invalid_packets = 0;

    // Read packets one at a time
    while (auto pkt = reader.read_next_packet()) {
        if (vrtio::is_valid(*pkt)) {
            valid_packets++;

            // Access packet type
            auto type = vrtio::packet_type(*pkt);
            std::cout << "Type: " << static_cast<int>(type) << "\n";

            // Process based on type
            if (vrtio::is_data_packet(*pkt)) {
                const auto& data = std::get<vrtio::RuntimeDataPacket>(*pkt);
                auto payload = data.payload();
                // Process payload...
                (void)payload;
            }

        } else {
            invalid_packets++;

            // Get error details from invalid packet
            const auto& invalid = std::get<vrtio::InvalidPacket>(*pkt);
            auto error = invalid.error;
            // Handle error...
            (void)error;
        }
    }
    // [/QUICKSTART-MANUAL]

    // Verification
    EXPECT_GT(valid_packets, 0) << "Should have read valid packets";
    EXPECT_EQ(invalid_packets, 0) << "Test file should have no invalid packets";

    std::cout << "Read " << valid_packets << " valid packets, " << invalid_packets
              << " invalid packets\n";
}
