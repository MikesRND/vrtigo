// [TITLE]
// Reading VRT Files
// [/TITLE]
//
// This test demonstrates reading a VRT file with the high-level
// VRTFileReader API. It shows type-safe packet iteration with
// automatic validation and elegant visitor pattern usage.

#include <iostream>

#include <gtest/gtest.h>
#include <vrtigo/sample_span.hpp>
#include <vrtigo/vrtigo_io.hpp>

// Test data file paths
#include <filesystem>

using vrtigo::field::sample_rate;

const std::filesystem::path test_data_dir = TEST_DATA_DIR;
const auto sine_wave_file = test_data_dir / "VITA49SineWaveData.bin";

// [EXAMPLE]
// High-Level File Reading
// [/EXAMPLE]

// [DESCRIPTION]
// This example demonstrates reading a VRT file with the high-level reader:
// - Automatic packet validation
// - Type-safe variant access (dynamic::DataPacketView, dynamic::ContextPacketView)
// - Elegant iteration with for_each helpers
// - Zero-copy access to packet data
// [/DESCRIPTION]

TEST(QuickstartSnippet, ReadVRTFile) {
    // Skip if test file doesn't exist
    if (!std::filesystem::exists(sine_wave_file)) {
        GTEST_SKIP() << "Sine wave test file not found";
    }

    // [SNIPPET]
    using namespace vrtigo::dynamic; // File reader returns dynamic packet views
    using namespace vrtigo::field;

    // Open VRT file - that's it!
    vrtigo::VRTFileReader<> reader(sine_wave_file.c_str());
    ASSERT_TRUE(reader.is_open());

    // Count packets and samples
    size_t data_packets = 0;
    size_t context_packets = 0;
    size_t total_samples = 0;

    // Iterate through all valid packets
    reader.for_each_validated_packet([&](const vrtigo::PacketVariant& pkt) {
        if (vrtigo::is_data_packet(pkt)) {
            data_packets++;

            // Access type-safe data packet view
            const auto& data = std::get<DataPacketView>(pkt);

            // Typed sample access - count I/Q samples directly
            auto samples = vrtigo::SampleSpanView<std::complex<int16_t>>(data.payload());
            total_samples += samples.count();

        } else if (vrtigo::is_context_packet(pkt)) {
            context_packets++;

            // Access context packet view
            const auto& ctx = std::get<ContextPacketView>(pkt);
            if (auto sr = ctx[sample_rate]) {
                std::cout << "Context packet sample rate: " << sr.value() << " Hz\n";
            }
        }

        return true; // Continue processing
    });
    // [/SNIPPET]

    // Verification
    EXPECT_GT(data_packets, 0) << "Should have read some data packets";
    EXPECT_GT(total_samples, 0) << "Should have extracted samples";

    std::cout << "Read " << data_packets << " data packets, " << context_packets
              << " context packets, " << total_samples << " samples total\n";
}

// [EXAMPLE]
// Manual Packet Iteration
// [/EXAMPLE]

// [DESCRIPTION]
// This example shows manual packet iteration for more control.
// Use this when you need to handle invalid packets or implement
// custom processing logic.
// [/DESCRIPTION]

TEST(QuickstartSnippet, ReadVRTFileManual) {
    // Skip if test file doesn't exist
    if (!std::filesystem::exists(sine_wave_file)) {
        GTEST_SKIP() << "Sine wave test file not found";
    }

    // [SNIPPET]
    using namespace vrtigo::dynamic; // File reader returns dynamic packet views

    // Manual packet iteration with full control
    vrtigo::VRTFileReader<> reader(sine_wave_file.c_str());
    ASSERT_TRUE(reader.is_open());

    size_t valid_packets = 0;
    size_t invalid_packets = 0;

    // Read packets one at a time
    while (auto pkt = reader.read_next_packet()) {
        if (pkt.has_value()) {
            valid_packets++;

            // Access packet type
            auto type = vrtigo::packet_type(pkt.value());
            std::cout << "Type: " << static_cast<int>(type) << "\n";

            // Process based on type
            if (vrtigo::is_data_packet(pkt.value())) {
                const auto& data = std::get<DataPacketView>(pkt.value());
                auto payload = data.payload();
                // Process payload...
                (void)payload;
            }

        } else {
            invalid_packets++;

            // Get error details from parse result
            auto error = pkt.error();
            // Handle error...
            (void)error;
        }
    }
    // [/SNIPPET]

    // Verification
    EXPECT_GT(valid_packets, 0) << "Should have read valid packets";
    EXPECT_EQ(invalid_packets, 0) << "Test file should have no invalid packets";

    std::cout << "Read " << valid_packets << " valid packets, " << invalid_packets
              << " invalid packets\n";
}
