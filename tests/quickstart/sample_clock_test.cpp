// [TITLE]
// Sample Clock
// [/TITLE]
//
// This test demonstrates using SampleClock to generate deterministic
// timestamps for sample-based systems, including the StartTime API
// for wall-clock synchronization and PPS alignment.

#include <gtest/gtest.h>
#include <vrtigo/vrtigo_utils.hpp>

using namespace vrtigo;
using namespace std::chrono_literals;
using utils::SampleClock;
using utils::StartTime;

// [TEXT]
// All examples assume `using namespace vrtigo;` and `using namespace std::chrono_literals;`.
// [/TEXT]

// [EXAMPLE]
// Basic Timestamp Generation
// [/EXAMPLE]

// [DESCRIPTION]
// SampleClock generates deterministic timestamps at a fixed sample rate.
// Create a clock with a sample period, then use `tick()` to advance time.
// By default, timestamps start at zero (epoch).
// [/DESCRIPTION]

TEST(QuickstartSnippet, BasicTimestampGeneration) {
    // [SNIPPET]
    // Create clock at 1 MHz sample rate (1 microsecond period)
    // Timestamps start at zero by default
    SampleClock<> clock(1e-6);

    // Query current time without advancing (starts at 0)
    auto t0 = clock.now(); // 0.000000 seconds

    // Advance by one sample
    auto t1 = clock.tick(); // 0.000001 seconds

    // Advance by multiple samples
    auto t2 = clock.tick(99); // 0.000100 seconds (100 samples total)
    // [/SNIPPET]

    EXPECT_EQ(t0.tsi(), 0u);
    EXPECT_EQ(t0.tsf(), 0u);

    EXPECT_EQ(t1.tsi(), 0u);
    EXPECT_EQ(t1.tsf(), 1'000'000ULL); // 1 microsecond in picoseconds

    EXPECT_EQ(t2.tsi(), 0u);
    EXPECT_EQ(t2.tsf(), 100'000'000ULL); // 100 microseconds in picoseconds

    EXPECT_EQ(clock.elapsed_samples(), 100u);
}

// [EXAMPLE]
// Absolute Start Time
// [/EXAMPLE]

// [DESCRIPTION]
// Pass a timestamp directly to start the clock at a specific time.
// This is useful for replaying recorded data or testing with known timestamps.
// [/DESCRIPTION]

TEST(QuickstartSnippet, AbsoluteStartTime) {
    // [SNIPPET]
    // Start clock at a specific timestamp (e.g., from recorded data)
    UtcRealTimestamp start(100u, 500'000'000'000ULL); // 100.5 seconds
    SampleClock<> clock(1e-6, start);

    auto t0 = clock.now();
    // t0.tsi() == 100, t0.tsf() == 500ms in picoseconds
    // [/SNIPPET]

    EXPECT_EQ(t0.tsi(), 100u);
    EXPECT_EQ(t0.tsf(), 500'000'000'000ULL);
}

// [EXAMPLE]
// Wall-Clock Start Time
// [/EXAMPLE]

// [DESCRIPTION]
// Use `StartTime::now()` to start the clock from the current wall-clock time.
// This is useful when timestamps need to reflect actual UTC time.
// [/DESCRIPTION]

TEST(QuickstartSnippet, WallClockStartTime) {
    // [SNIPPET]
    // Start clock at current UTC wall-clock time
    SampleClock<TsiType::utc> clock(1e-6, StartTime::now());

    // First timestamp reflects actual current time
    auto ts = clock.now();
    // ts.tsi() contains UTC seconds since epoch
    // [/SNIPPET]

    // Verify timestamp is reasonable (after year 2020)
    EXPECT_GT(ts.tsi(), 1577836800u); // Jan 1, 2020
}

// [EXAMPLE]
// PPS Alignment
// [/EXAMPLE]

// [DESCRIPTION]
// Use `StartTime::at_next_second()` to align the clock to the next whole-second
// boundary. This is essential for PPS (pulse-per-second) synchronization where
// timestamps must start exactly on second edges.
// [/DESCRIPTION]

TEST(QuickstartSnippet, PPSAlignment) {
    // [SNIPPET]
    // Align clock to next second boundary (for PPS sync)
    SampleClock<TsiType::utc> clock(1e-6, StartTime::at_next_second());

    // First timestamp is exactly on a second boundary
    auto ts = clock.now();
    // ts.tsf() == 0 (no fractional seconds)
    // [/SNIPPET]

    EXPECT_EQ(ts.tsf(), 0u);
    EXPECT_GT(ts.tsi(), 0u);
}

// [EXAMPLE]
// Delayed Start with Offset
// [/EXAMPLE]

// [DESCRIPTION]
// Use `StartTime::now_plus()` for a delayed start, or `StartTime::at_next_second_plus()`
// to start at a fixed offset after a second boundary. The latter is useful for
// systems that need processing time after PPS edges.
// [/DESCRIPTION]

TEST(QuickstartSnippet, DelayedStartWithOffset) {
    // [SNIPPET]
    // Start 500ms from now (setup/coordination time)
    SampleClock<TsiType::utc> clock1(1e-6, StartTime::now_plus(500ms));

    // Start 100ms after next second boundary (PPS + processing offset)
    SampleClock<TsiType::utc> clock2(1e-6, StartTime::at_next_second_plus(100ms));

    auto ts = clock2.now();
    // ts.tsf() == 100ms in picoseconds
    // [/SNIPPET]

    // Verify clock2 starts at exactly 100ms into the second
    EXPECT_EQ(ts.tsf(), 100'000'000'000ULL); // 100ms in picoseconds
}
