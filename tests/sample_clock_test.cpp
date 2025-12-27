#include <chrono>
#include <limits>
#include <thread>

#include <cmath>
#include <gtest/gtest.h>
#include <vrtigo/vrtigo_utils.hpp>

using namespace vrtigo;
using namespace vrtigo::utils;

TEST(SampleClockTest, TickAdvancesByPeriod) {
    SampleClock<> clock(0.001); // 1 ms

    auto first = clock.tick();
    EXPECT_EQ(first.tsi(), 0u);
    EXPECT_EQ(first.tsf(), 1'000'000'000ULL);

    auto second = clock.tick(999); // total 1000 samples = 1 second
    EXPECT_EQ(second.tsi(), 1u);
    EXPECT_EQ(second.tsf(), 0u);
    EXPECT_EQ(clock.elapsed_samples(), 1000u);
}

TEST(SampleClockTest, HonorsStartEpoch) {
    using time_point = SampleClock<>::time_point;
    time_point start(2u, 900'000'000'000ULL); // 2.9 seconds

    SampleClock<> clock(0.2, start); // 200 ms - uses time_point convenience ctor

    auto after_tick = clock.tick();
    EXPECT_EQ(after_tick.tsi(), 3u);
    EXPECT_EQ(after_tick.tsf(), 100'000'000'000ULL); // 3.1 seconds total
}

TEST(SampleClockTest, RoundingReported) {
    SampleClock<> clock(1.1e-12); // rounds to 1 picosecond

    EXPECT_EQ(clock.period().picoseconds(), 1u);
    EXPECT_FALSE(clock.period().is_exact());
}

TEST(SampleClockTest, ResetClearsProgress) {
    SampleClock<> clock(0.0005); // 500 microseconds
    (void)clock.tick(4);         // 2 milliseconds total

    EXPECT_EQ(clock.elapsed_samples(), 4u);

    clock.reset();
    auto after_reset = clock.now();

    EXPECT_EQ(after_reset.tsi(), 0u);
    EXPECT_EQ(after_reset.tsf(), 0u);
    EXPECT_EQ(clock.elapsed_samples(), 0u);
}

// Error Condition Tests
TEST(SampleClockTest, ThrowsOnZeroPeriod) {
    EXPECT_THROW(SampleClock<> clock(0.0), std::invalid_argument);
}

TEST(SampleClockTest, ThrowsOnNegativePeriod) {
    EXPECT_THROW(SampleClock<> clock(-0.001), std::invalid_argument);
}

TEST(SampleClockTest, ThrowsOnNaNPeriod) {
    EXPECT_THROW(SampleClock<> clock(std::nan("")), std::invalid_argument);
}

TEST(SampleClockTest, ThrowsOnInfinitePeriod) {
    EXPECT_THROW(SampleClock<> clock(std::numeric_limits<double>::infinity()),
                 std::invalid_argument);
}

TEST(SampleClockTest, ThrowsOnPeriodThatRoundsToZero) {
    // Period smaller than 0.5 picoseconds rounds to zero
    EXPECT_THROW(SampleClock<> clock(0.4e-12), std::invalid_argument);
}

// Saturation Tests (overflow now saturates to max timestamp)
TEST(SampleClockTest, SaturatesOnHugeTickCount) {
    // Start near UINT32_MAX seconds so adding Duration::max() causes saturation
    UtcRealTimestamp start(3'000'000'000U, 0); // ~95 years since epoch
    SampleClock<> clock(1.0, start);           // 1 second per sample

    // Use a large positive value that fits in int64_t but will still overflow Duration
    // INT64_MAX seconds = ~292 billion years, so Duration multiplication will saturate
    // Then Duration::max() (~68 years) + 95 years > 136 years (UINT32_MAX)
    auto result = clock.tick(static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));

    // Should saturate to max timestamp since 3e9 + 2.1e9 > UINT32_MAX
    EXPECT_TRUE(saturated(result));
    EXPECT_EQ(result.tsi(), std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(result.tsf(), UtcRealTimestamp::MAX_FRACTIONAL);
}

TEST(SampleClockTest, SaturatesOnMultiplicationOverflow) {
    // Start near UINT32_MAX seconds so Duration::max() causes saturation
    UtcRealTimestamp start(3'000'000'000U, 0);
    SampleClock<> clock(1.0, start); // 1 second = 1e12 picoseconds

    // Tick by huge number that would overflow the multiplication -> Duration::max()
    // Then adding Duration::max() to start causes timestamp saturation
    auto result = clock.tick(UINT64_MAX / 1000);

    // Should saturate to max timestamp
    EXPECT_TRUE(saturated(result));
}

// Precision Tests
TEST(SampleClockTest, InexactPeriodsReportAsInexact) {
    // Most periods are not exactly representable due to floating-point precision
    SampleClock<> clock1(1.0 / 61.44e6); // 61.44 MHz (LTE) - not exactly representable
    EXPECT_FALSE(clock1.period().is_exact());

    SampleClock<> clock2(1.0 / 30.72e6); // 30.72 MHz (LTE) - not exactly representable
    EXPECT_FALSE(clock2.period().is_exact());

    // Even clean powers of 10 may have floating-point precision issues
    SampleClock<> clock3(1e-3); // 1 ms
    // is_exact() depends on femtosecond tolerance
}

TEST(SampleClockTest, PeriodErrorReporting) {
    // Test ppm error calculation for RF sample rates
    SampleClock<> clock_lte(1.0 / 61.44e6); // 61.44 MHz (LTE sample rate)

    // Should have some non-zero error
    EXPECT_NE(clock_lte.period().error_picoseconds(), 0.0);

    // ppm error should be small (well under 10 ppm for typical RF rates)
    double ppm = clock_lte.period().error_ppm();
    EXPECT_LT(std::abs(ppm), 10.0); // Less than 10 ppm

    // For a hypothetically exact period, error should be zero
    // (though in practice floating-point may prevent this)
    SampleClock<> clock_100mhz(1.0 / 100e6);
    if (clock_100mhz.period().is_exact()) {
        EXPECT_EQ(clock_100mhz.period().error_picoseconds(), 0.0);
        EXPECT_EQ(clock_100mhz.period().error_ppm(), 0.0);
    }
}

TEST(SampleClockTest, DeterministicTicking) {
    // Verify tick(N) produces same result as N individual ticks
    SampleClock<> clock1(1.0 / 100e6); // 100 MHz
    SampleClock<> clock2(1.0 / 100e6);

    auto result1 = clock1.tick(1000);
    for (int i = 0; i < 1000; ++i) {
        (void)clock2.tick();
    }
    auto result2 = clock2.now();

    EXPECT_EQ(result1.tsi(), result2.tsi());
    EXPECT_EQ(result1.tsf(), result2.tsf());
}

// Reset re-resolution tests
TEST(SampleClockTest, ResetReResolvesStartTimeNow) {
    // StartTime::now() should capture fresh wall-clock on each reset()
    SampleClock<TsiType::utc> clock(1e-6, StartTime::now());

    auto first_start = clock.now();
    (void)clock.tick(1000);

    // Small delay to ensure wall clock advances
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    clock.reset(); // Should re-resolve StartTime::now()
    auto second_start = clock.now();

    // Second start should be later than first (fresh wall-clock capture)
    EXPECT_GT(second_start.tsi() * 1'000'000'000'000ULL + second_start.tsf(),
              first_start.tsi() * 1'000'000'000'000ULL + first_start.tsf());
}

TEST(SampleClockTest, ResetPreservesAbsoluteStart) {
    // StartTime::absolute() should return same time on reset
    using time_point = SampleClock<>::time_point;
    time_point fixed_start(100u, 500'000'000'000ULL);

    SampleClock<> clock(1e-3, fixed_start);
    (void)clock.tick(1000);
    clock.reset();

    auto after_reset = clock.now();
    EXPECT_EQ(after_reset.tsi(), 100u);
    EXPECT_EQ(after_reset.tsf(), 500'000'000'000ULL);
}

// Long-running tests - verifying no saturation for normal use cases
TEST(SampleClockTest, LongRunningNoSaturation) {
    // Simulate 1 year of continuous operation at 10 MHz
    // This is well within the 68-year Duration range
    SampleClock<> clock(1e-7); // 100ns period = 10 MHz

    // Advance by 1 year worth of samples (but do it in seconds increments)
    // 1 year = 31536000 seconds
    // At 10 MHz, that's 315 trillion samples - too many to actually tick
    // Instead, advance by 1 million samples at a time for a shorter test
    for (int i = 0; i < 100; ++i) {
        clock.tick(1'000'000); // 0.1 second per iteration = 10 seconds total
    }

    auto result = clock.now();
    // Should be exactly 10 seconds from start
    EXPECT_EQ(result.tsi(), 10u);
    EXPECT_EQ(result.tsf(), 0u);
    EXPECT_FALSE(saturated(result));
}

TEST(SampleClockTest, ElapsedSamplesWorksOver106Days) {
    // Verify elapsed_samples() works correctly for durations > 106 days
    // (Regression test: previously used total_picoseconds() which saturates)
    SampleClock<> clock(1.0); // 1 second per sample

    // Advance by 1 year = 31,536,000 samples (well over 106 days = 9.16M samples)
    constexpr uint64_t ONE_YEAR_SECONDS = 365ULL * 24 * 60 * 60;
    clock.tick(ONE_YEAR_SECONDS);

    // elapsed_samples() should return the correct count, not a saturated value
    EXPECT_EQ(clock.elapsed_samples(), ONE_YEAR_SECONDS);
}

TEST(SampleClockTest, DurationDivisionWorksOver106Days) {
    // Verify Duration division works for spans > 106 days
    Duration one_year = Duration::from_seconds(static_cast<int64_t>(365) * 24 * 60 * 60);
    Duration one_day = Duration::from_seconds(static_cast<int64_t>(24 * 60 * 60));

    // Should return 365, not a bogus saturated value
    EXPECT_EQ(one_year / one_day, 365);
}
