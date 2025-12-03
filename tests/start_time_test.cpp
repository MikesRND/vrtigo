#include <chrono>
#include <thread>

#include <gtest/gtest.h>
#include <vrtigo/vrtigo_utils.hpp>

using namespace vrtigo;
using namespace std::chrono_literals;

// =============================================================================
// Basic Resolution Tests
// =============================================================================

TEST(StartTimeTest, ZeroResolvesToEpoch) {
    auto st = StartTime::zero();
    auto resolved = st.resolve();

    EXPECT_EQ(resolved.tsi(), 0u);
    EXPECT_EQ(resolved.tsf(), 0u);
}

TEST(StartTimeTest, DefaultConstructionEqualsZero) {
    // Default-constructed StartTime should equal StartTime::zero()
    StartTime default_st{};
    auto zero_st = StartTime::zero();

    EXPECT_EQ(default_st.base(), zero_st.base());
    EXPECT_EQ(default_st.resolve().tsi(), zero_st.resolve().tsi());
    EXPECT_EQ(default_st.resolve().tsf(), zero_st.resolve().tsf());
}

TEST(StartTimeTest, AbsoluteResolvesToExactTimestamp) {
    UtcRealTimestamp ts(12345u, 500'000'000'000ULL); // 12345.5 seconds
    auto st = StartTime::absolute(ts);
    auto resolved = st.resolve();

    EXPECT_EQ(resolved.tsi(), 12345u);
    EXPECT_EQ(resolved.tsf(), 500'000'000'000ULL);
}

TEST(StartTimeTest, NowResolvesToApproximateCurrentTime) {
    auto before = UtcRealTimestamp::now();
    auto st = StartTime::now();
    auto resolved = st.resolve();
    auto after = UtcRealTimestamp::now();

    // resolved should be between before and after (inclusive)
    EXPECT_GE(resolved, before);
    EXPECT_LE(resolved, after);
}

// =============================================================================
// now_plus Tests
// =============================================================================

TEST(StartTimeTest, NowPlusPositiveOffset) {
    auto before = UtcRealTimestamp::now();
    auto st = StartTime::now_plus(100ms);
    auto resolved = st.resolve();

    // resolved should be at least 100ms after 'before'
    auto expected_min = before + 100ms;
    EXPECT_GE(resolved, expected_min);
}

TEST(StartTimeTest, NowPlusZeroOffset) {
    auto before = UtcRealTimestamp::now();
    auto st = StartTime::now_plus(0ns);
    auto resolved = st.resolve();
    auto after = UtcRealTimestamp::now();

    EXPECT_GE(resolved, before);
    EXPECT_LE(resolved, after);
}

TEST(StartTimeTest, NowPlusNegativeOffset) {
    auto now_approx = UtcRealTimestamp::now();
    auto st = StartTime::now_plus(-100ms);
    auto resolved = st.resolve();

    // resolved should be approximately 100ms before now
    auto expected_max = now_approx;
    EXPECT_LT(resolved, expected_max);
}

// =============================================================================
// at_next_second Tests
// =============================================================================

TEST(StartTimeTest, AtNextSecondHasZeroFractional) {
    auto st = StartTime::at_next_second();
    auto resolved = st.resolve();

    // Fractional should be zero (on second boundary)
    EXPECT_EQ(resolved.tsf(), 0u);
}

TEST(StartTimeTest, AtNextSecondIsAtOrAfterNow) {
    auto before = UtcRealTimestamp::now();
    auto st = StartTime::at_next_second();
    auto resolved = st.resolve();

    // resolved should be >= before (we round up)
    EXPECT_GE(resolved, before);
}

TEST(StartTimeTest, AtNextSecondAdvancesIfNotOnBoundary) {
    // Get current time
    auto now = UtcRealTimestamp::now();

    // If we're not exactly on a boundary, next second should be tsi + 1
    if (now.tsf() > 0) {
        auto st = StartTime::at_next_second();
        auto resolved = st.resolve();

        // Should have advanced to at least the next second
        EXPECT_GT(resolved.tsi(), now.tsi());
        EXPECT_EQ(resolved.tsf(), 0u);
    }
}

// =============================================================================
// at_next_second_plus Tests
// =============================================================================

TEST(StartTimeTest, AtNextSecondPlusAppliesOffset) {
    auto st = StartTime::at_next_second_plus(100ms);
    auto resolved = st.resolve();

    // Fractional should be 100ms in picoseconds
    EXPECT_EQ(resolved.tsf(), 100'000'000'000ULL);
}

TEST(StartTimeTest, AtNextSecondPlusLargeOffsetCarriesSeconds) {
    auto st = StartTime::at_next_second_plus(1500ms); // 1.5 seconds
    auto resolved = st.resolve();

    // Should have 500ms fractional (1 second carried to tsi)
    EXPECT_EQ(resolved.tsf(), 500'000'000'000ULL);
}

// =============================================================================
// Overflow/Saturation Tests
// =============================================================================

TEST(StartTimeTest, AtNextSecondSaturatesAtMaxSeconds) {
    // Create a timestamp at UINT32_MAX seconds with some fractional
    UtcRealTimestamp max_ts(UINT32_MAX, 500'000'000'000ULL);

    // We can't easily test at_next_second() with a custom "now", but we can
    // verify the behavior with absolute timestamps and large offsets
    auto st = StartTime::absolute(max_ts);
    auto resolved = st.resolve();

    EXPECT_EQ(resolved.tsi(), UINT32_MAX);
    EXPECT_EQ(resolved.tsf(), 500'000'000'000ULL);
}

TEST(StartTimeTest, LargeOffsetSaturates) {
    // Use absolute with a high tsi and add a huge offset
    UtcRealTimestamp near_max(UINT32_MAX - 1, 0);
    auto st = StartTime::absolute(near_max + std::chrono::nanoseconds::max());

    // Timestamp arithmetic should saturate
    auto resolved = st.resolve();
    EXPECT_EQ(resolved.tsi(), UINT32_MAX);
}

// =============================================================================
// Base/Offset Accessor Tests
// =============================================================================

TEST(StartTimeTest, BaseAccessorReturnsCorrectType) {
    EXPECT_EQ(StartTime::now().base(), StartTime::Base::now);
    EXPECT_EQ(StartTime::now_plus(100ms).base(), StartTime::Base::now);
    EXPECT_EQ(StartTime::at_next_second().base(), StartTime::Base::next_second);
    EXPECT_EQ(StartTime::at_next_second_plus(100ms).base(), StartTime::Base::next_second);
    EXPECT_EQ(StartTime::absolute(UtcRealTimestamp{}).base(), StartTime::Base::absolute);
    EXPECT_EQ(StartTime::zero().base(), StartTime::Base::zero);
}

TEST(StartTimeTest, OffsetAccessorReturnsCorrectValue) {
    EXPECT_EQ(StartTime::now().offset(), 0ns);
    EXPECT_EQ(StartTime::now_plus(100ms).offset(), 100ms);
    EXPECT_EQ(StartTime::now_plus(-50ms).offset(), -50ms);
    EXPECT_EQ(StartTime::at_next_second().offset(), 0ns);
    EXPECT_EQ(StartTime::at_next_second_plus(200ms).offset(), 200ms);
}

// =============================================================================
// SampleClock Integration Tests
// =============================================================================

TEST(StartTimeTest, SampleClockWithStartTimeNow) {
    // Create clock with StartTime::now()
    SampleClock<TsiType::utc> clock(1e-6, StartTime::now());

    auto ts = clock.now();
    auto current = UtcRealTimestamp::now();

    // Clock's time should be close to current time (within a few ms)
    auto diff_ns = current - ts;
    EXPECT_LT(std::abs(diff_ns.count()), 100'000'000); // Within 100ms
}

TEST(StartTimeTest, SampleClockWithStartTimeZero) {
    SampleClock<TsiType::utc> clock(1e-6, StartTime::zero());

    auto ts = clock.now();
    EXPECT_EQ(ts.tsi(), 0u);
    EXPECT_EQ(ts.tsf(), 0u);
}

TEST(StartTimeTest, SampleClockWithStartTimeAbsolute) {
    UtcRealTimestamp start(1000u, 500'000'000'000ULL);
    SampleClock<TsiType::utc> clock(1e-3, StartTime::absolute(start)); // 1ms period

    auto ts = clock.now();
    EXPECT_EQ(ts.tsi(), 1000u);
    EXPECT_EQ(ts.tsf(), 500'000'000'000ULL);

    // After one tick
    auto after_tick = clock.tick();
    EXPECT_EQ(after_tick.tsi(), 1000u);
    EXPECT_EQ(after_tick.tsf(), 501'000'000'000ULL); // +1ms
}

TEST(StartTimeTest, SampleClockWithStartTimeAtNextSecond) {
    SampleClock<TsiType::utc> clock(1e-6, StartTime::at_next_second());

    auto ts = clock.now();
    // Should be on a second boundary
    EXPECT_EQ(ts.tsf(), 0u);
}

TEST(StartTimeTest, SampleClockOtherTsiWithStartTime) {
    // TsiType::other should also work with StartTime
    SampleClock<TsiType::other> clock(1e-6, StartTime::now());

    auto ts = clock.now();
    // Just verify it compiles and runs
    EXPECT_GE(ts.tsi(), 0u);
}

TEST(StartTimeTest, SampleClockEquivalentConstructors) {
    // Both constructors should produce equivalent clocks for same effective start time
    UtcRealTimestamp start(500u, 0);

    SampleClock<TsiType::utc> clock1(1e-3, start);
    SampleClock<TsiType::utc> clock2(1e-3, StartTime::absolute(start));

    EXPECT_EQ(clock1.now().tsi(), clock2.now().tsi());
    EXPECT_EQ(clock1.now().tsf(), clock2.now().tsf());

    // After ticking
    auto ts1 = clock1.tick(100);
    auto ts2 = clock2.tick(100);

    EXPECT_EQ(ts1.tsi(), ts2.tsi());
    EXPECT_EQ(ts1.tsf(), ts2.tsf());
}

// =============================================================================
// Constexpr Tests (compile-time verification)
// =============================================================================

TEST(StartTimeTest, FactoryMethodsAreConstexpr) {
    // These should all be valid constexpr expressions
    constexpr auto st_zero = StartTime::zero();
    constexpr auto st_abs = StartTime::absolute(UtcRealTimestamp{100u, 0});
    constexpr auto st_now = StartTime::now();
    constexpr auto st_now_plus = StartTime::now_plus(100ms);
    constexpr auto st_next = StartTime::at_next_second();
    constexpr auto st_next_plus = StartTime::at_next_second_plus(100ms);

    // Verify they have expected bases
    static_assert(st_zero.base() == StartTime::Base::zero);
    static_assert(st_abs.base() == StartTime::Base::absolute);
    static_assert(st_now.base() == StartTime::Base::now);
    static_assert(st_now_plus.base() == StartTime::Base::now);
    static_assert(st_next.base() == StartTime::Base::next_second);
    static_assert(st_next_plus.base() == StartTime::Base::next_second);

    // Suppress unused variable warnings
    (void)st_zero;
    (void)st_abs;
    (void)st_now;
    (void)st_now_plus;
    (void)st_next;
    (void)st_next_plus;
}
