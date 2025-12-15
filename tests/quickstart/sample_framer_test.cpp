// [TITLE]
// Sample Framer
// [/TITLE]
//
// This test demonstrates using SampleFramer to accumulate VRT payload
// samples into fixed-size frames with automatic endian conversion.

// Suppress nodiscard warnings for cleaner quickstart examples
#pragma GCC diagnostic ignored "-Wunused-result"

#include <array>
#include <complex>
#include <vector>

#include <cstdint>
#include <gtest/gtest.h>
#include <vrtigo/utils/sample_framer.hpp>

using namespace vrtigo;
using namespace vrtigo::utils;

// [TEXT]
// All examples assume `using namespace vrtigo;` and `using namespace vrtigo::utils;`.
// [/TEXT]

// ============================================================================
// Test Helpers
// ============================================================================

// Create int16 payload
static std::vector<uint8_t> make_int16_payload(int16_t base, size_t count) {
    std::vector<uint8_t> payload;
    for (size_t i = 0; i < count; ++i) {
        auto v = static_cast<uint16_t>(base + static_cast<int16_t>(i));
        payload.push_back(static_cast<uint8_t>(v >> 8));
        payload.push_back(static_cast<uint8_t>(v & 0xFF));
    }
    return payload;
}

// Create int32 payload
static std::vector<uint8_t> make_int32_payload(int32_t base, size_t count) {
    std::vector<uint8_t> payload;
    for (size_t i = 0; i < count; ++i) {
        auto v = static_cast<uint32_t>(base + static_cast<int32_t>(i));
        payload.push_back(static_cast<uint8_t>(v >> 24));
        payload.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
        payload.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
        payload.push_back(static_cast<uint8_t>(v & 0xFF));
    }
    return payload;
}

// Create complex float payload
static std::vector<uint8_t> make_complex_float_payload(float base, size_t count) {
    std::vector<uint8_t> payload;
    auto push_float = [&](float f) {
        uint32_t bits = std::bit_cast<uint32_t>(f);
        payload.push_back(static_cast<uint8_t>(bits >> 24));
        payload.push_back(static_cast<uint8_t>((bits >> 16) & 0xFF));
        payload.push_back(static_cast<uint8_t>((bits >> 8) & 0xFF));
        payload.push_back(static_cast<uint8_t>(bits & 0xFF));
    };
    for (size_t i = 0; i < count; ++i) {
        push_float(base + static_cast<float>(i) * 2.0f);        // real
        push_float(base + static_cast<float>(i) * 2.0f + 1.0f); // imag
    }
    return payload;
}

// ============================================================================
// Examples
// ============================================================================

// [EXAMPLE]
// Basic Sample Framing
// [/EXAMPLE]

// [DESCRIPTION]
// SampleFramer accumulates payload bytes into fixed-size frames.
// Your callback fires when each frame is complete.
// [/DESCRIPTION]

TEST(QuickstartSnippet, BasicSampleFraming) {
    // [SNIPPET]
    // Create storage for a frame of samples (8 samples per frame)
    std::array<int32_t, 8> frame_buf{};

    // Define a simple callback that computes sum of samples in the frame
    int64_t sum = 0;
    auto on_frame = [&](std::span<const int32_t> frame) {
        for (int32_t sample : frame) {
            sum += sample;
        }
        return true;
    };

    // Create the framer
    SimpleSampleFramer<int32_t> framer(frame_buf, 8, on_frame);

    // First packet: 4 samples (not enough for frame)
    framer.ingest_payload(make_int32_payload(1, 4)); // [1,2,3,4]

    // Second packet: 4 samples (completes frame)
    framer.ingest_payload(make_int32_payload(5, 4)); // [5,6,7,8]

    // Callback fired with frame [1,2,3,4,5,6,7,8]
    // sum == 36
    // [/SNIPPET]

    EXPECT_EQ(sum, 36);
    EXPECT_EQ(framer.emitted_frames(), 1);
}

// [EXAMPLE]
// Complex Float Samples
// [/EXAMPLE]

// [DESCRIPTION]
// SampleFramer supports complex types via the template parameter.
// [/DESCRIPTION]

TEST(QuickstartSnippet, ComplexFloatSamples) {
    // [SNIPPET]
    std::array<std::complex<float>, 4> frame_buf{};
    std::vector<std::complex<float>> received;

    SimpleSampleFramer<std::complex<float>> framer(
        frame_buf, 4, [&](std::span<const std::complex<float>> frame) {
            received.insert(received.end(), frame.begin(), frame.end());
            return true;
        });

    auto payload = make_complex_float_payload(1.0f, 4);
    framer.ingest_payload(payload);

    // received[0] == (1.0f, 2.0f)
    // received[1] == (3.0f, 4.0f)
    // ...
    // [/SNIPPET]

    ASSERT_EQ(received.size(), 4);
    EXPECT_FLOAT_EQ(received[0].real(), 1.0f);
    EXPECT_FLOAT_EQ(received[0].imag(), 2.0f);
    EXPECT_FLOAT_EQ(received[1].real(), 3.0f);
    EXPECT_FLOAT_EQ(received[1].imag(), 4.0f);
}

// [EXAMPLE]
// Stopping Early
// [/EXAMPLE]

// [DESCRIPTION]
// Return `false` from the callback to stop processing. The framer returns
// `FrameError::stop_requested` so you know processing was halted.
// [/DESCRIPTION]

TEST(QuickstartSnippet, StoppingEarly) {
    // [SNIPPET]
    std::array<int16_t, 4> frame_buf{};
    size_t frames_seen = 0;

    SimpleSampleFramer<int16_t> framer(frame_buf, 4, [&](std::span<const int16_t>) {
        ++frames_seen;
        return frames_seen < 2; // stop after 2 frames
    });

    // Would produce 3 frames, but callback stops after 2
    auto payload = make_int16_payload(0, 12);
    auto result = framer.ingest_payload(payload);

    // result.error() == FrameError::stop_requested
    // frames_seen == 2
    // [/SNIPPET]

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), FrameError::stop_requested);
    EXPECT_EQ(frames_seen, 2);
}
