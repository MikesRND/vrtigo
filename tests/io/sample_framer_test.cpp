#include <array>
#include <bit>
#include <complex>
#include <vector>

#include <cstring>
#include <gtest/gtest.h>
#include <vrtigo/utils/sample_framer.hpp>

using namespace vrtigo;
using namespace vrtigo::utils;

// ============================================================================
// Test Helpers
// ============================================================================

class SampleFramerTest : public ::testing::Test {
protected:
    // Create big-endian int16 bytes
    static std::array<uint8_t, 2> make_be_int16(int16_t value) {
        uint16_t u = static_cast<uint16_t>(value);
        return {static_cast<uint8_t>(u >> 8), static_cast<uint8_t>(u & 0xFF)};
    }

    // Create big-endian float32 bytes
    static std::array<uint8_t, 4> make_be_float32(float value) {
        uint32_t bits = std::bit_cast<uint32_t>(value);
        return {static_cast<uint8_t>(bits >> 24), static_cast<uint8_t>((bits >> 16) & 0xFF),
                static_cast<uint8_t>((bits >> 8) & 0xFF), static_cast<uint8_t>(bits & 0xFF)};
    }

    // Create payload of N int16 samples starting at base value
    static std::vector<uint8_t> make_int16_payload(int16_t base, size_t count) {
        std::vector<uint8_t> payload;
        for (size_t i = 0; i < count; ++i) {
            auto bytes = make_be_int16(static_cast<int16_t>(base + i));
            payload.insert(payload.end(), bytes.begin(), bytes.end());
        }
        return payload;
    }

    // Create payload of N complex float32 samples
    static std::vector<uint8_t> make_float_complex_payload(float base, size_t count) {
        std::vector<uint8_t> payload;
        for (size_t i = 0; i < count; ++i) {
            auto real_bytes = make_be_float32(base + static_cast<float>(i) * 2.0f);
            auto imag_bytes = make_be_float32(base + static_cast<float>(i) * 2.0f + 1.0f);
            payload.insert(payload.end(), real_bytes.begin(), real_bytes.end());
            payload.insert(payload.end(), imag_bytes.begin(), imag_bytes.end());
        }
        return payload;
    }

    // Helper to create dynamic span from vector
    template <typename T>
    static std::span<T> dspan(std::vector<T>& v) {
        return std::span<T>(v.data(), v.size());
    }
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(SampleFramerTest, ConstructLinearMode) {
    std::vector<int16_t> buffer(100);
    size_t callback_count = 0;

    SimpleSampleFramer<int16_t> framer(dspan(buffer), 50, [&](std::span<const int16_t>) {
        ++callback_count;
        return true;
    });

    EXPECT_EQ(framer.samples_per_frame(), 50);
    EXPECT_EQ(framer.frame_size_bytes(), 100); // 50 * 2 bytes
    EXPECT_FALSE(framer.is_ping_pong());
    EXPECT_EQ(framer.emitted_frames(), 0);
}

TEST_F(SampleFramerTest, ConstructPingPongMode) {
    std::vector<int16_t> buf_a(100);
    std::vector<int16_t> buf_b(100);

    SimpleSampleFramer<int16_t> framer(dspan(buf_a), dspan(buf_b), 50,
                                       [](std::span<const int16_t>) { return true; });

    EXPECT_EQ(framer.samples_per_frame(), 50);
    EXPECT_TRUE(framer.is_ping_pong());
}

TEST_F(SampleFramerTest, ConstructPingPongWithEmptyBFallsBackToLinear) {
    std::vector<int16_t> buf_a(100);
    std::span<int16_t> buf_b{}; // empty

    SimpleSampleFramer<int16_t> framer(dspan(buf_a), buf_b, 50,
                                       [](std::span<const int16_t>) { return true; });

    EXPECT_FALSE(framer.is_ping_pong());
}

TEST_F(SampleFramerTest, RejectsZeroSamplesPerFrame) {
    std::vector<int16_t> buffer(100);

    EXPECT_THROW((SimpleSampleFramer<int16_t>(dspan(buffer), 0,
                                              [](std::span<const int16_t>) { return true; })),
                 std::invalid_argument);
}

TEST_F(SampleFramerTest, RejectsBufferTooSmall) {
    std::vector<int16_t> buffer(10);

    EXPECT_THROW((SimpleSampleFramer<int16_t>(dspan(buffer), 50, // buffer too small for 50 samples
                                              [](std::span<const int16_t>) { return true; })),
                 std::invalid_argument);
}

TEST_F(SampleFramerTest, RejectsPingPongBufferBTooSmall) {
    std::vector<int16_t> buf_a(100);
    std::vector<int16_t> buf_b(10); // too small

    EXPECT_THROW((SimpleSampleFramer<int16_t>(dspan(buf_a), dspan(buf_b), 50,
                                              [](std::span<const int16_t>) { return true; })),
                 std::invalid_argument);
}

// ============================================================================
// Ingest Tests
// ============================================================================

TEST_F(SampleFramerTest, IngestExactFrame) {
    std::vector<int16_t> buffer(10);
    std::vector<int16_t> received;

    SimpleSampleFramer<int16_t> framer(dspan(buffer), 10, [&](std::span<const int16_t> frame) {
        received.assign(frame.begin(), frame.end());
        return true;
    });

    auto payload = make_int16_payload(100, 10);
    auto result = framer.ingest_payload(payload);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 1); // One frame emitted
    EXPECT_EQ(framer.emitted_frames(), 1);
    EXPECT_EQ(framer.buffered_samples(), 0);

    ASSERT_EQ(received.size(), 10);
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(received[i], static_cast<int16_t>(100 + i));
    }
}

TEST_F(SampleFramerTest, IngestPartialFrame) {
    std::vector<int16_t> buffer(10);
    bool callback_called = false;

    SimpleSampleFramer<int16_t> framer(dspan(buffer), 10, [&](std::span<const int16_t>) {
        callback_called = true;
        return true;
    });

    auto payload = make_int16_payload(100, 5); // Only 5 samples
    auto result = framer.ingest_payload(payload);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0); // No frames emitted yet
    EXPECT_FALSE(callback_called);
    EXPECT_EQ(framer.buffered_samples(), 5);
}

TEST_F(SampleFramerTest, IngestAccumulatesAcrossMultipleCalls) {
    std::vector<int16_t> buffer(10);
    std::vector<int16_t> received;

    SimpleSampleFramer<int16_t> framer(dspan(buffer), 10, [&](std::span<const int16_t> frame) {
        received.assign(frame.begin(), frame.end());
        return true;
    });

    // First 6 samples
    auto payload1 = make_int16_payload(100, 6);
    auto result1 = framer.ingest_payload(payload1);
    ASSERT_TRUE(result1.has_value());
    EXPECT_EQ(*result1, 0);
    EXPECT_EQ(framer.buffered_samples(), 6);

    // Next 4 samples (completes frame)
    auto payload2 = make_int16_payload(106, 4);
    auto result2 = framer.ingest_payload(payload2);
    ASSERT_TRUE(result2.has_value());
    EXPECT_EQ(*result2, 1);
    EXPECT_EQ(framer.buffered_samples(), 0);

    ASSERT_EQ(received.size(), 10);
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(received[i], static_cast<int16_t>(100 + i));
    }
}

TEST_F(SampleFramerTest, IngestMultipleFramesAtOnce) {
    std::vector<int16_t> buffer(10);
    size_t frame_count = 0;
    std::vector<std::vector<int16_t>> frames;

    SimpleSampleFramer<int16_t> framer(dspan(buffer), 10, [&](std::span<const int16_t> frame) {
        frames.push_back(std::vector<int16_t>(frame.begin(), frame.end()));
        ++frame_count;
        return true;
    });

    // 25 samples = 2 complete frames + 5 remaining
    auto payload = make_int16_payload(100, 25);
    auto result = framer.ingest_payload(payload);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 2);
    EXPECT_EQ(frame_count, 2);
    EXPECT_EQ(framer.buffered_samples(), 5);

    // Verify frame contents
    ASSERT_EQ(frames.size(), 2);
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(frames[0][i], static_cast<int16_t>(100 + i));
        EXPECT_EQ(frames[1][i], static_cast<int16_t>(110 + i));
    }
}

TEST_F(SampleFramerTest, IngestPayloadNotAligned) {
    std::vector<int16_t> buffer(10);

    SimpleSampleFramer<int16_t> framer(dspan(buffer), 10,
                                       [](std::span<const int16_t>) { return true; });

    // 5 bytes is not aligned to 2-byte int16 samples
    std::vector<uint8_t> payload = {0x00, 0x64, 0x00, 0x65, 0x00};
    auto result = framer.ingest_payload(payload);

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), FrameError::payload_not_aligned);
}

TEST_F(SampleFramerTest, IngestStopRequestedByCallback) {
    std::vector<int16_t> buffer(10);
    size_t frame_count = 0;

    SimpleSampleFramer<int16_t> framer(dspan(buffer), 10, [&](std::span<const int16_t>) {
        ++frame_count;
        return false; // Stop after first frame
    });

    // 20 samples = would be 2 frames, but callback stops after first
    auto payload = make_int16_payload(100, 20);
    auto result = framer.ingest_payload(payload);

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), FrameError::stop_requested);
    EXPECT_EQ(frame_count, 1);
}

// ============================================================================
// Flush Tests
// ============================================================================

TEST_F(SampleFramerTest, FlushPartialEmitsRemainder) {
    std::vector<int16_t> buffer(10);
    std::vector<int16_t> received;

    SimpleSampleFramer<int16_t> framer(dspan(buffer), 10, [&](std::span<const int16_t> frame) {
        received.assign(frame.begin(), frame.end());
        return true;
    });

    // Ingest 7 samples (partial frame)
    auto payload = make_int16_payload(100, 7);
    (void)framer.ingest_payload(payload);
    EXPECT_EQ(framer.buffered_samples(), 7);

    // Flush partial
    auto result = framer.flush_partial();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 1);
    EXPECT_EQ(framer.buffered_samples(), 0);

    // Verify received only 7 samples (no padding)
    ASSERT_EQ(received.size(), 7);
    for (size_t i = 0; i < 7; ++i) {
        EXPECT_EQ(received[i], static_cast<int16_t>(100 + i));
    }
}

TEST_F(SampleFramerTest, FlushPartialNoOpWhenEmpty) {
    std::vector<int16_t> buffer(10);
    bool callback_called = false;

    SimpleSampleFramer<int16_t> framer(dspan(buffer), 10, [&](std::span<const int16_t>) {
        callback_called = true;
        return true;
    });

    auto result = framer.flush_partial();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 0);
    EXPECT_FALSE(callback_called);
}

TEST_F(SampleFramerTest, FlushPartialStopRequested) {
    std::vector<int16_t> buffer(10);

    SimpleSampleFramer<int16_t> framer(dspan(buffer), 10,
                                       [](std::span<const int16_t>) { return false; });

    auto payload = make_int16_payload(100, 5);
    (void)framer.ingest_payload(payload);

    auto result = framer.flush_partial();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), FrameError::stop_requested);
}

// ============================================================================
// Ping-Pong Tests
// ============================================================================

TEST_F(SampleFramerTest, PingPongAlternatesBuffers) {
    std::vector<int16_t> buf_a(10);
    std::vector<int16_t> buf_b(10);

    std::vector<const int16_t*> frame_ptrs;

    SimpleSampleFramer<int16_t> framer(dspan(buf_a), dspan(buf_b), 10,
                                       [&](std::span<const int16_t> frame) {
                                           frame_ptrs.push_back(frame.data());
                                           return true;
                                       });

    // Emit 3 frames
    auto payload = make_int16_payload(100, 30);
    (void)framer.ingest_payload(payload);

    ASSERT_EQ(frame_ptrs.size(), 3);
    EXPECT_EQ(frame_ptrs[0], buf_a.data()); // Frame 1 in buf_a
    EXPECT_EQ(frame_ptrs[1], buf_b.data()); // Frame 2 in buf_b
    EXPECT_EQ(frame_ptrs[2], buf_a.data()); // Frame 3 back to buf_a
}

// ============================================================================
// Reset Tests
// ============================================================================

TEST_F(SampleFramerTest, ResetClearsState) {
    std::vector<int16_t> buffer(10);

    SimpleSampleFramer<int16_t> framer(dspan(buffer), 10,
                                       [](std::span<const int16_t>) { return true; });

    // Ingest some data
    auto payload = make_int16_payload(100, 15); // 1 frame + 5 buffered
    (void)framer.ingest_payload(payload);
    EXPECT_EQ(framer.emitted_frames(), 1);
    EXPECT_EQ(framer.buffered_samples(), 5);

    // Reset
    framer.reset();

    EXPECT_EQ(framer.emitted_frames(), 0);
    EXPECT_EQ(framer.buffered_samples(), 0);
}

// ============================================================================
// Complex Float Tests
// ============================================================================

TEST_F(SampleFramerTest, ComplexFloatSamples) {
    std::vector<std::complex<float>> buffer(10);
    std::vector<std::complex<float>> received;

    SimpleSampleFramer<std::complex<float>> framer(dspan(buffer), 10,
                                                   [&](std::span<const std::complex<float>> frame) {
                                                       received.assign(frame.begin(), frame.end());
                                                       return true;
                                                   });

    auto payload = make_float_complex_payload(1.0f, 10);
    auto result = framer.ingest_payload(payload);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 1);

    ASSERT_EQ(received.size(), 10);
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(received[i].real(), 1.0f + static_cast<float>(i) * 2.0f);
        EXPECT_FLOAT_EQ(received[i].imag(), 1.0f + static_cast<float>(i) * 2.0f + 1.0f);
    }
}

// ============================================================================
// Frame Error String Test
// ============================================================================

TEST_F(SampleFramerTest, FrameErrorStrings) {
    EXPECT_STRNE(frame_error_string(FrameError::payload_not_aligned), "Unknown error");
    EXPECT_STRNE(frame_error_string(FrameError::stop_requested), "Unknown error");
}
