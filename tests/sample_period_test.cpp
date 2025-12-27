#include <cmath>
#include <gtest/gtest.h>
#include <vrtigo.hpp>

using namespace vrtigo;

// Test fixture for SamplePeriod tests
class SamplePeriodTest : public ::testing::Test {
protected:
    static constexpr uint64_t ONE_SECOND_PICOS = 1'000'000'000'000ULL;
    static constexpr uint64_t ONE_MS_PICOS = 1'000'000'000ULL;
    static constexpr uint64_t ONE_US_PICOS = 1'000'000ULL;
    static constexpr uint64_t ONE_NS_PICOS = 1'000ULL;
};

// ==============================================================================
// from_picoseconds Factory
// ==============================================================================

TEST_F(SamplePeriodTest, FromPicosecondsBasic) {
    auto sp = SamplePeriod::from_picoseconds(100'000);
    EXPECT_EQ(sp.picoseconds(), 100'000);
    EXPECT_TRUE(sp.is_exact());
    EXPECT_EQ(sp.error_picoseconds(), 0.0);
    EXPECT_EQ(sp.error_ppm(), 0.0);
}

TEST_F(SamplePeriodTest, FromPicosecondsOne) {
    auto sp = SamplePeriod::from_picoseconds(1);
    EXPECT_EQ(sp.picoseconds(), 1);
    EXPECT_TRUE(sp.is_exact());
}

TEST_F(SamplePeriodTest, FromPicosecondsZeroClampsToOne) {
    // Zero period is invalid (would cause div-by-zero), clamps to 1
    auto sp = SamplePeriod::from_picoseconds(0);
    EXPECT_EQ(sp.picoseconds(), 1);
    EXPECT_TRUE(sp.is_exact());
}

TEST_F(SamplePeriodTest, FromPicosecondsLarge) {
    auto sp = SamplePeriod::from_picoseconds(ONE_SECOND_PICOS);
    EXPECT_EQ(sp.picoseconds(), ONE_SECOND_PICOS);
    EXPECT_TRUE(sp.is_exact());
}

// ==============================================================================
// from_rate_hz Factory
// ==============================================================================

TEST_F(SamplePeriodTest, FromRateHz10MHz) {
    // 10 MHz = 100 ns = 100,000 ps
    auto sp = SamplePeriod::from_rate_hz(10e6);
    ASSERT_TRUE(sp.has_value());
    EXPECT_EQ(sp->picoseconds(), 100'000);
    // Clean powers of 10 are exact within femtosecond tolerance
    EXPECT_TRUE(sp->is_exact());
}

TEST_F(SamplePeriodTest, FromRateHz1Hz) {
    // 1 Hz = 1 second = 10^12 ps
    auto sp = SamplePeriod::from_rate_hz(1.0);
    ASSERT_TRUE(sp.has_value());
    EXPECT_EQ(sp->picoseconds(), ONE_SECOND_PICOS);
    EXPECT_TRUE(sp->is_exact());
}

TEST_F(SamplePeriodTest, FromRateHz1kHz) {
    // 1 kHz = 1 ms = 10^9 ps
    auto sp = SamplePeriod::from_rate_hz(1000.0);
    ASSERT_TRUE(sp.has_value());
    EXPECT_EQ(sp->picoseconds(), ONE_MS_PICOS);
    EXPECT_TRUE(sp->is_exact());
}

TEST_F(SamplePeriodTest, FromRateHz1THz) {
    // 1 THz = 1 ps
    auto sp = SamplePeriod::from_rate_hz(1e12);
    ASSERT_TRUE(sp.has_value());
    EXPECT_EQ(sp->picoseconds(), 1);
    EXPECT_TRUE(sp->is_exact());
}

TEST_F(SamplePeriodTest, FromRateHzFractional) {
    // 2.5 MHz = 400 ns = 400,000 ps
    auto sp = SamplePeriod::from_rate_hz(2.5e6);
    ASSERT_TRUE(sp.has_value());
    EXPECT_EQ(sp->picoseconds(), 400'000);
    EXPECT_TRUE(sp->is_exact());
}

TEST_F(SamplePeriodTest, FromRateHzZero) {
    auto sp = SamplePeriod::from_rate_hz(0.0);
    EXPECT_FALSE(sp.has_value());
}

TEST_F(SamplePeriodTest, FromRateHzNegative) {
    auto sp = SamplePeriod::from_rate_hz(-10e6);
    EXPECT_FALSE(sp.has_value());
}

TEST_F(SamplePeriodTest, FromRateHzNaN) {
    auto sp = SamplePeriod::from_rate_hz(std::nan(""));
    EXPECT_FALSE(sp.has_value());
}

TEST_F(SamplePeriodTest, FromRateHzInfinity) {
    auto sp = SamplePeriod::from_rate_hz(std::numeric_limits<double>::infinity());
    EXPECT_FALSE(sp.has_value());
}

TEST_F(SamplePeriodTest, FromRateHzVerySmall) {
    // Very small rate would produce period > UINT64_MAX
    auto sp = SamplePeriod::from_rate_hz(1e-10); // 10^-10 Hz
    EXPECT_FALSE(sp.has_value());
}

TEST_F(SamplePeriodTest, FromRateHzVeryLarge) {
    // Very large rate would round to zero period
    auto sp = SamplePeriod::from_rate_hz(1e20); // Period < 1 ps, rounds to 0
    EXPECT_FALSE(sp.has_value());
}

// ==============================================================================
// from_seconds Factory
// ==============================================================================

TEST_F(SamplePeriodTest, FromSecondsBasic) {
    // 1 second period
    auto sp = SamplePeriod::from_seconds(1.0);
    ASSERT_TRUE(sp.has_value());
    EXPECT_EQ(sp->picoseconds(), ONE_SECOND_PICOS);
    // Clean powers of 10 are exact within femtosecond tolerance
    EXPECT_TRUE(sp->is_exact());
}

TEST_F(SamplePeriodTest, FromSecondsMillisecond) {
    auto sp = SamplePeriod::from_seconds(0.001);
    ASSERT_TRUE(sp.has_value());
    EXPECT_EQ(sp->picoseconds(), ONE_MS_PICOS);
    EXPECT_TRUE(sp->is_exact());
}

TEST_F(SamplePeriodTest, FromSecondsMicrosecond) {
    auto sp = SamplePeriod::from_seconds(1e-6);
    ASSERT_TRUE(sp.has_value());
    EXPECT_EQ(sp->picoseconds(), ONE_US_PICOS);
    EXPECT_TRUE(sp->is_exact());
}

TEST_F(SamplePeriodTest, FromSecondsNanosecond) {
    auto sp = SamplePeriod::from_seconds(1e-9);
    ASSERT_TRUE(sp.has_value());
    EXPECT_EQ(sp->picoseconds(), ONE_NS_PICOS);
    EXPECT_TRUE(sp->is_exact());
}

TEST_F(SamplePeriodTest, FromSecondsPicosecond) {
    auto sp = SamplePeriod::from_seconds(1e-12);
    ASSERT_TRUE(sp.has_value());
    EXPECT_EQ(sp->picoseconds(), 1);
    EXPECT_TRUE(sp->is_exact());
}

TEST_F(SamplePeriodTest, FromSecondsZero) {
    auto sp = SamplePeriod::from_seconds(0.0);
    EXPECT_FALSE(sp.has_value());
}

TEST_F(SamplePeriodTest, FromSecondsNegative) {
    auto sp = SamplePeriod::from_seconds(-1.0);
    EXPECT_FALSE(sp.has_value());
}

TEST_F(SamplePeriodTest, FromSecondsVerySmall) {
    // Smaller than 1 ps rounds to 0
    auto sp = SamplePeriod::from_seconds(1e-15);
    EXPECT_FALSE(sp.has_value());
}

// ==============================================================================
// from_ratio Factory (Exact Rational)
// ==============================================================================

TEST_F(SamplePeriodTest, FromRatio10MHz) {
    // 10 MHz = 10,000,000 / 1 Hz
    auto sp = SamplePeriod::from_ratio(10'000'000, 1);
    ASSERT_TRUE(sp.has_value());
    EXPECT_EQ(sp->picoseconds(), 100'000); // 100 ns
    EXPECT_TRUE(sp->is_exact());           // Integer factory
    EXPECT_EQ(sp->error_ppm(), 0.0);
}

TEST_F(SamplePeriodTest, FromRatio48kHz) {
    // 48 kHz is NOT exactly representable: 10^12 % 48000 != 0
    auto sp = SamplePeriod::from_ratio(48'000, 1);
    // Period = 10^12 / 48000 = 20833333.333... ps - NOT exact, should fail
    EXPECT_FALSE(sp.has_value());
}

TEST_F(SamplePeriodTest, FromRatio48kHzActual) {
    // 48 kHz is NOT exactly representable: 10^12 % 48000 != 0
    auto sp = SamplePeriod::from_ratio(48'000, 1);
    EXPECT_FALSE(sp.has_value()); // Should fail - not exactly divisible
}

TEST_F(SamplePeriodTest, FromRatio44100Hz) {
    // 44.1 kHz CD audio - NOT exactly representable
    auto sp = SamplePeriod::from_ratio(44'100, 1);
    EXPECT_FALSE(sp.has_value());
}

TEST_F(SamplePeriodTest, FromRatio1MHz) {
    // 1 MHz = exact: 10^12 / 10^6 = 10^6 ps
    auto sp = SamplePeriod::from_ratio(1'000'000, 1);
    ASSERT_TRUE(sp.has_value());
    EXPECT_EQ(sp->picoseconds(), ONE_US_PICOS);
    EXPECT_TRUE(sp->is_exact());
}

TEST_F(SamplePeriodTest, FromRatio2500000Hz) {
    // 2.5 MHz = 2,500,000 / 1 Hz - exact: 10^12 / 2.5e6 = 400,000 ps
    auto sp = SamplePeriod::from_ratio(2'500'000, 1);
    ASSERT_TRUE(sp.has_value());
    EXPECT_EQ(sp->picoseconds(), 400'000);
    EXPECT_TRUE(sp->is_exact());
}

TEST_F(SamplePeriodTest, FromRatioNTSC) {
    // NTSC video: 48048 Hz = 48000 * 1001 / 1000
    // Rate = 48048 / 1 Hz, but we need from_ratio(num, den) where rate = num/den
    // For 48048 Hz: period = 10^12 / 48048 = 20820000.0133... NOT exact
    auto sp = SamplePeriod::from_ratio(48048, 1);
    EXPECT_FALSE(sp.has_value()); // Not exactly divisible
}

TEST_F(SamplePeriodTest, FromRatioExactRational) {
    // 100 Hz is exact: 10^12 / 100 = 10^10 ps
    auto sp = SamplePeriod::from_ratio(100, 1);
    ASSERT_TRUE(sp.has_value());
    EXPECT_EQ(sp->picoseconds(), 10'000'000'000ULL);
    EXPECT_TRUE(sp->is_exact());
}

TEST_F(SamplePeriodTest, FromRatio1Hz) {
    // 1 Hz = 1 second = 10^12 ps
    auto sp = SamplePeriod::from_ratio(1, 1);
    ASSERT_TRUE(sp.has_value());
    EXPECT_EQ(sp->picoseconds(), ONE_SECOND_PICOS);
    EXPECT_TRUE(sp->is_exact());
}

TEST_F(SamplePeriodTest, FromRatioZeroNumerator) {
    auto sp = SamplePeriod::from_ratio(0, 1);
    EXPECT_FALSE(sp.has_value());
}

TEST_F(SamplePeriodTest, FromRatioZeroDenominator) {
    auto sp = SamplePeriod::from_ratio(1, 0);
    EXPECT_FALSE(sp.has_value());
}

TEST_F(SamplePeriodTest, FromRatioWithDenominator) {
    // 1/2 Hz = 0.5 Hz → period = 2 seconds = 2e12 ps
    auto sp = SamplePeriod::from_ratio(1, 2);
    ASSERT_TRUE(sp.has_value());
    EXPECT_EQ(sp->picoseconds(), 2 * ONE_SECOND_PICOS);
    EXPECT_TRUE(sp->is_exact());
}

// ==============================================================================
// Accessors
// ==============================================================================

TEST_F(SamplePeriodTest, RateHz) {
    auto sp = SamplePeriod::from_picoseconds(100'000);
    EXPECT_DOUBLE_EQ(sp.rate_hz(), 10e6); // 10 MHz = 100,000 ps = 100 ns period
}

TEST_F(SamplePeriodTest, RateHz10MHz) {
    auto sp = SamplePeriod::from_rate_hz(10e6);
    ASSERT_TRUE(sp.has_value());
    // Round-trip should be close
    EXPECT_NEAR(sp->rate_hz(), 10e6, 1.0);
}

TEST_F(SamplePeriodTest, Seconds) {
    auto sp = SamplePeriod::from_picoseconds(ONE_MS_PICOS);
    EXPECT_DOUBLE_EQ(sp.seconds(), 0.001);
}

// ==============================================================================
// Exactness and Error Reporting
// ==============================================================================

TEST_F(SamplePeriodTest, IsExactInteger) {
    auto sp = SamplePeriod::from_picoseconds(12345);
    EXPECT_TRUE(sp.is_exact());
}

TEST_F(SamplePeriodTest, IsExactRational) {
    auto sp = SamplePeriod::from_ratio(1'000'000, 1);
    ASSERT_TRUE(sp.has_value());
    EXPECT_TRUE(sp->is_exact());
}

TEST_F(SamplePeriodTest, IsExactFPCleanValues) {
    // Clean power-of-10 values are exact within femtosecond tolerance
    auto sp1 = SamplePeriod::from_rate_hz(10e6);
    auto sp2 = SamplePeriod::from_seconds(1e-6);

    ASSERT_TRUE(sp1.has_value());
    ASSERT_TRUE(sp2.has_value());

    EXPECT_TRUE(sp1->is_exact());
    EXPECT_TRUE(sp2->is_exact());
}

TEST_F(SamplePeriodTest, IsInexactFPIrrationalRates) {
    // Rates that don't divide evenly into 10^12 are inexact
    // 61.44 MHz (LTE) - period = 10^12 / 61.44e6 = 16276.04166... ps
    auto sp_lte = SamplePeriod::from_rate_hz(61.44e6);
    ASSERT_TRUE(sp_lte.has_value());
    EXPECT_FALSE(sp_lte->is_exact());

    // 44.1 kHz (CD audio) - period = 10^12 / 44100 = 22675736.961... ps
    auto sp_cd = SamplePeriod::from_rate_hz(44100.0);
    ASSERT_TRUE(sp_cd.has_value());
    EXPECT_FALSE(sp_cd->is_exact());
}

TEST_F(SamplePeriodTest, ErrorPpmExact) {
    auto sp = SamplePeriod::from_picoseconds(100'000);
    EXPECT_EQ(sp.error_ppm(), 0.0);
}

TEST_F(SamplePeriodTest, ErrorPpmInexact) {
    // FP factory has tiny but non-zero error
    auto sp = SamplePeriod::from_rate_hz(10e6);
    ASSERT_TRUE(sp.has_value());
    // Error should be very small for a nice round rate
    EXPECT_NEAR(sp->error_ppm(), 0.0, 1e-6);
}

TEST_F(SamplePeriodTest, ErrorPicosecondsExact) {
    auto sp = SamplePeriod::from_picoseconds(100'000);
    EXPECT_EQ(sp.error_picoseconds(), 0.0);
}

// ==============================================================================
// Conversion to Duration
// ==============================================================================

TEST_F(SamplePeriodTest, ToDurationBasic) {
    auto sp = SamplePeriod::from_picoseconds(100'000);
    auto d = sp.to_duration();
    EXPECT_EQ(d.total_picoseconds(), 100'000);
}

TEST_F(SamplePeriodTest, ToDurationLarge) {
    auto sp = SamplePeriod::from_picoseconds(ONE_SECOND_PICOS);
    auto d = sp.to_duration();
    // 1 second = 1 second + 0 subsecond picoseconds
    EXPECT_EQ(d.seconds(), 1);
    EXPECT_EQ(d.picoseconds(), 0u);
}

TEST_F(SamplePeriodTest, ToDurationLargeDoesNotSaturate) {
    // SamplePeriod can hold values > INT64_MAX picoseconds (~106 days)
    // But new Duration can hold 68 years, so this no longer saturates
    auto sp = SamplePeriod::from_picoseconds(
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1);
    auto d = sp.to_duration();

    // Should fit without saturation: INT64_MAX + 1 picos ≈ 106 days
    EXPECT_FALSE(saturated(d));
    // Verify the actual values
    // (9,223,372,036,854,775,808 picos) / 1e12 = 9,223,372 seconds + 36,854,775,808 picos
    EXPECT_EQ(d.seconds(), 9'223'372);
    EXPECT_EQ(d.picoseconds(), 36'854'775'808ULL);
}

// ==============================================================================
// Comparison
// ==============================================================================

TEST_F(SamplePeriodTest, Equality) {
    auto a = SamplePeriod::from_picoseconds(100'000);
    auto b = SamplePeriod::from_picoseconds(100'000);
    auto c = SamplePeriod::from_picoseconds(200'000);

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST_F(SamplePeriodTest, EqualityIgnoresExactness) {
    // Two periods with same picoseconds compare equal regardless of exactness flag
    auto sp1 = SamplePeriod::from_picoseconds(ONE_US_PICOS); // exact (integer factory)
    auto sp2 = SamplePeriod::from_rate_hz(1e6);              // from FP factory

    ASSERT_TRUE(sp2.has_value());
    EXPECT_EQ(sp1.picoseconds(), sp2->picoseconds());
    EXPECT_EQ(sp1, *sp2); // Equal by picoseconds
}

TEST_F(SamplePeriodTest, Ordering) {
    auto a = SamplePeriod::from_picoseconds(100'000);
    auto b = SamplePeriod::from_picoseconds(200'000);

    EXPECT_LT(a, b);
    EXPECT_LE(a, b);
    EXPECT_GT(b, a);
    EXPECT_GE(b, a);
}

// ==============================================================================
// Duration::from_samples
// ==============================================================================

TEST_F(SamplePeriodTest, DurationFromSamplesBasic) {
    auto sp = SamplePeriod::from_picoseconds(100'000); // 100 ns
    auto d = Duration::from_samples(1000, sp);

    ASSERT_TRUE(d.has_value());
    EXPECT_EQ(d->total_picoseconds(), 1000 * 100'000);
}

TEST_F(SamplePeriodTest, DurationFromSamplesNegativeCount) {
    auto sp = SamplePeriod::from_picoseconds(100'000);
    auto d = Duration::from_samples(-1000, sp);

    ASSERT_TRUE(d.has_value());
    EXPECT_EQ(d->total_picoseconds(), -1000 * 100'000);
}

TEST_F(SamplePeriodTest, DurationFromSamplesZero) {
    auto sp = SamplePeriod::from_picoseconds(100'000);
    auto d = Duration::from_samples(0, sp);

    ASSERT_TRUE(d.has_value());
    EXPECT_EQ(d->total_picoseconds(), 0);
}

TEST_F(SamplePeriodTest, DurationFromSamplesOverflow) {
    auto sp = SamplePeriod::from_picoseconds(ONE_SECOND_PICOS);
    // Trying to compute INT64_MAX * ONE_SECOND_PICOS would overflow
    auto d = Duration::from_samples(std::numeric_limits<int64_t>::max(), sp);

    EXPECT_FALSE(d.has_value());
}

TEST_F(SamplePeriodTest, DurationFromSamplesRealWorld) {
    // 10 MHz rate for 1 million samples = 100 ms
    auto sp = SamplePeriod::from_picoseconds(100'000); // 100 ns = 10 MHz
    auto d = Duration::from_samples(1'000'000, sp);

    ASSERT_TRUE(d.has_value());
    // 100 ms = 0 seconds + 100,000,000,000 picoseconds
    EXPECT_EQ(d->seconds(), 0);
    EXPECT_EQ(d->picoseconds(), 100'000'000'000ULL);
}

// ==============================================================================
// Real-World Scenarios
// ==============================================================================

TEST_F(SamplePeriodTest, RealWorldAudioCD) {
    // CD audio is 44.1 kHz - NOT exactly representable
    auto sp = SamplePeriod::from_ratio(44'100, 1);
    EXPECT_FALSE(sp.has_value());

    // But we can use from_rate_hz with small error
    auto sp_approx = SamplePeriod::from_rate_hz(44100.0);
    ASSERT_TRUE(sp_approx.has_value());
    EXPECT_FALSE(sp_approx->is_exact());
    // Error should be very small
    EXPECT_LT(std::abs(sp_approx->error_ppm()), 1.0);
}

TEST_F(SamplePeriodTest, RealWorldSDR) {
    // SDR at 2.4 GHz sample rate
    auto sp = SamplePeriod::from_rate_hz(2.4e9);
    ASSERT_TRUE(sp.has_value());
    // Period should be ~417 ps
    EXPECT_NEAR(sp->picoseconds(), 417, 1);
}

TEST_F(SamplePeriodTest, RealWorldGigabitEthernet) {
    // 1.25 GHz (Gigabit Ethernet symbol rate)
    auto sp = SamplePeriod::from_ratio(1'250'000'000, 1);
    ASSERT_TRUE(sp.has_value());
    EXPECT_EQ(sp->picoseconds(), 800); // 800 ps
    EXPECT_TRUE(sp->is_exact());
}

TEST_F(SamplePeriodTest, RealWorld10GigE) {
    // 10.3125 Gbps line rate for 10 Gigabit Ethernet (10GBASE-R)
    // This is NOT exactly representable as integer picoseconds
    auto sp = SamplePeriod::from_rate_hz(10.3125e9);
    ASSERT_TRUE(sp.has_value());
    // Period is ~97 ps
    EXPECT_NEAR(sp->picoseconds(), 97, 1);
    EXPECT_FALSE(sp->is_exact());
}

// ==============================================================================
// Femtosecond Tolerance
// ==============================================================================

TEST_F(SamplePeriodTest, FemtosecondToleranceExact) {
    // Verify the tolerance constant is accessible
    EXPECT_EQ(SamplePeriod::EXACTNESS_TOLERANCE_PICOS, 1e-6);
}

TEST_F(SamplePeriodTest, FemtosecondToleranceApplied) {
    // 100 MHz = 10,000 ps - should be exact because FP error is sub-femtosecond
    auto sp = SamplePeriod::from_rate_hz(100e6);
    ASSERT_TRUE(sp.has_value());
    EXPECT_EQ(sp->picoseconds(), 10'000);
    EXPECT_TRUE(sp->is_exact());
    EXPECT_EQ(sp->error_picoseconds(), 0.0);
}

TEST_F(SamplePeriodTest, FemtosecondToleranceNotApplied) {
    // 61.44 MHz (LTE) has significant rounding error > 1 femtosecond
    // Period = 10^12 / 61.44e6 = 16276.041666... ps
    auto sp = SamplePeriod::from_rate_hz(61.44e6);
    ASSERT_TRUE(sp.has_value());
    EXPECT_EQ(sp->picoseconds(), 16276);
    EXPECT_FALSE(sp->is_exact());
    // Error is approximately 0.0416... ps, well above 1 femtosecond
    EXPECT_GT(std::abs(sp->error_picoseconds()), SamplePeriod::EXACTNESS_TOLERANCE_PICOS);
}
