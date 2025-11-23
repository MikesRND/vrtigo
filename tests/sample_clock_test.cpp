#include <limits>

#include <cmath>
#include <gtest/gtest.h>
#include <vrtigo/vrtigo_utils.hpp>

using namespace vrtigo;

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
    using OtherStamp = Timestamp<TsiType::other, TsfType::real_time>;
    OtherStamp start(2u, 900'000'000'000ULL); // 2.9 seconds

    SampleClock<> clock(0.2, start); // 200 ms

    auto after_tick = clock.tick();
    EXPECT_EQ(after_tick.tsi(), 3u);
    EXPECT_EQ(after_tick.tsf(), 100'000'000'000ULL); // 3.1 seconds total
}

TEST(SampleClockTest, RoundingReported) {
    SampleClock<> clock(1.1e-12); // rounds to 1 picosecond

    EXPECT_EQ(clock.sample_period_picoseconds(), 1u);
    EXPECT_FALSE(clock.is_exact());
}

TEST(SampleClockTest, ResetClearsProgress) {
    SampleClock<> clock(0.0005); // 500 microseconds
    clock.tick(4);               // 2 milliseconds total

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

// Overflow Tests
TEST(SampleClockTest, ThrowsOnAccumulatorOverflow) {
    SampleClock<> clock(1.0); // 1 second per sample

    // Attempt to tick by UINT64_MAX seconds - should overflow the accumulator
    EXPECT_THROW(clock.tick(UINT64_MAX), std::overflow_error);
}

TEST(SampleClockTest, ThrowsOnMultiplicationOverflow) {
    // Create clock with large period
    SampleClock<> clock(1.0); // 1 second = 1e12 picoseconds

    // Tick by huge number that would overflow the multiplication
    EXPECT_THROW(clock.tick(UINT64_MAX / 1000), std::overflow_error);
}

// Precision Tests
TEST(SampleClockTest, InexactPeriodsReportAsInexact) {
    // Most periods are not exactly representable due to floating-point precision
    SampleClock<> clock1(1.0 / 61.44e6); // 61.44 MHz (LTE) - not exactly representable
    EXPECT_FALSE(clock1.is_exact());

    SampleClock<> clock2(1.0 / 30.72e6); // 30.72 MHz (LTE) - not exactly representable
    EXPECT_FALSE(clock2.is_exact());

    // Even clean powers of 10 may have floating-point precision issues
    SampleClock<> clock3(1e-3); // 1 ms
    // is_exact() depends on long double precision
}

TEST(SampleClockTest, PeriodErrorReporting) {
    // Test ppm error calculation for RF sample rates
    SampleClock<> clock_lte(1.0 / 61.44e6); // 61.44 MHz (LTE sample rate)

    // Should have some non-zero error
    EXPECT_NE(clock_lte.error_picoseconds(), 0.0L);

    // ppm error should be small (well under 10 ppm for typical RF rates)
    double ppm = clock_lte.error_ppm();
    EXPECT_LT(std::abs(ppm), 10.0); // Less than 10 ppm

    // For a hypothetically exact period, error should be zero
    // (though in practice floating-point may prevent this)
    SampleClock<> clock_100mhz(1.0 / 100e6);
    if (clock_100mhz.is_exact()) {
        EXPECT_EQ(clock_100mhz.error_picoseconds(), 0.0L);
        EXPECT_EQ(clock_100mhz.error_ppm(), 0.0);
    }
}

TEST(SampleClockTest, DeterministicTicking) {
    // Verify tick(N) produces same result as N individual ticks
    SampleClock<> clock1(1.0 / 100e6); // 100 MHz
    SampleClock<> clock2(1.0 / 100e6);

    auto result1 = clock1.tick(1000);
    for (int i = 0; i < 1000; ++i) {
        clock2.tick();
    }
    auto result2 = clock2.now();

    EXPECT_EQ(result1.tsi(), result2.tsi());
    EXPECT_EQ(result1.tsf(), result2.tsf());
}
