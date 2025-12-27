#include <gtest/gtest.h>
#include <vrtigo.hpp>

using namespace vrtigo;

// Test fixture for Duration tests
class DurationTest : public ::testing::Test {
protected:
    static constexpr uint64_t PICOS_PER_SEC = Duration::PICOSECONDS_PER_SECOND;
    static constexpr int64_t ONE_SECOND_PICOS = static_cast<int64_t>(PICOS_PER_SEC);
    static constexpr int64_t ONE_MS_PICOS = 1'000'000'000LL;
    static constexpr int64_t ONE_US_PICOS = 1'000'000LL;
    static constexpr int64_t ONE_NS_PICOS = 1'000LL;
};

// ==============================================================================
// Construction and Named Constants
// ==============================================================================

TEST_F(DurationTest, DefaultConstruction) {
    Duration d;
    EXPECT_EQ(d.seconds(), 0);
    EXPECT_EQ(d.picoseconds(), 0);
    EXPECT_TRUE(d.is_zero());
}

TEST_F(DurationTest, Zero) {
    auto d = Duration::zero();
    EXPECT_EQ(d.seconds(), 0);
    EXPECT_EQ(d.picoseconds(), 0);
    EXPECT_TRUE(d.is_zero());
}

TEST_F(DurationTest, Min) {
    auto d = Duration::min();
    EXPECT_EQ(d.seconds(), std::numeric_limits<int32_t>::min());
    EXPECT_EQ(d.picoseconds(), 0);
    EXPECT_TRUE(d.is_negative());
    EXPECT_TRUE(saturated(d));
}

TEST_F(DurationTest, Max) {
    auto d = Duration::max();
    EXPECT_EQ(d.seconds(), std::numeric_limits<int32_t>::max());
    EXPECT_EQ(d.picoseconds(), Duration::MAX_PICOSECONDS);
    EXPECT_TRUE(d.is_positive());
    EXPECT_TRUE(saturated(d));
}

// ==============================================================================
// Direct Factories
// ==============================================================================

TEST_F(DurationTest, FromPicoseconds) {
    auto d = Duration::from_picoseconds(12345);
    EXPECT_EQ(d.seconds(), 0);
    EXPECT_EQ(d.picoseconds(), 12345);
    EXPECT_EQ(d.total_picoseconds(), 12345);
}

TEST_F(DurationTest, FromPicosecondsNegative) {
    // -12345 picos = {-1 sec, 10^12 - 12345 picos} in floor semantics
    auto d = Duration::from_picoseconds(-12345);
    EXPECT_EQ(d.seconds(), -1);
    EXPECT_EQ(d.picoseconds(), PICOS_PER_SEC - 12345);
    EXPECT_EQ(d.total_picoseconds(), -12345);
}

TEST_F(DurationTest, FromPicosecondsLargePositive) {
    // 1.5 seconds in picoseconds
    int64_t picos = ONE_SECOND_PICOS + ONE_SECOND_PICOS / 2;
    auto d = Duration::from_picoseconds(picos);
    EXPECT_EQ(d.seconds(), 1);
    EXPECT_EQ(d.picoseconds(), PICOS_PER_SEC / 2);
}

TEST_F(DurationTest, FromNanoseconds) {
    auto d = Duration::from_nanoseconds(100);
    EXPECT_EQ(d.total_picoseconds(), 100'000);
}

TEST_F(DurationTest, FromNanosecondsNegative) {
    auto d = Duration::from_nanoseconds(-100);
    EXPECT_EQ(d.total_picoseconds(), -100'000);
}

TEST_F(DurationTest, FromMicroseconds) {
    auto d = Duration::from_microseconds(5);
    EXPECT_EQ(d.total_picoseconds(), 5'000'000);
}

TEST_F(DurationTest, FromMilliseconds) {
    auto d = Duration::from_milliseconds(3);
    EXPECT_EQ(d.total_picoseconds(), 3'000'000'000);
}

TEST_F(DurationTest, FromSecondsInt) {
    auto d = Duration::from_seconds(int64_t{2});
    EXPECT_EQ(d.seconds(), 2);
    EXPECT_EQ(d.picoseconds(), 0);
}

TEST_F(DurationTest, FromSecondsIntNegative) {
    auto d = Duration::from_seconds(int64_t{-5});
    EXPECT_EQ(d.seconds(), -5);
    EXPECT_EQ(d.picoseconds(), 0);
}

// Overflow protection tests (prevent UB from multiplying before clamping)
TEST_F(DurationTest, FromNanosecondsOverflowSaturates) {
    // INT64_MAX nanoseconds would overflow if multiplied by 1000 before clamping
    auto d = Duration::from_nanoseconds(std::numeric_limits<int64_t>::max());
    EXPECT_EQ(d, Duration::max());
}

TEST_F(DurationTest, FromNanosecondsUnderflowSaturates) {
    auto d = Duration::from_nanoseconds(std::numeric_limits<int64_t>::min());
    EXPECT_EQ(d, Duration::min());
}

TEST_F(DurationTest, FromMillisecondsOverflowSaturates) {
    auto d = Duration::from_milliseconds(std::numeric_limits<int64_t>::max());
    EXPECT_EQ(d, Duration::max());
}

TEST_F(DurationTest, FromMicrosecondsOverflowSaturates) {
    auto d = Duration::from_microseconds(std::numeric_limits<int64_t>::max());
    EXPECT_EQ(d, Duration::max());
}

// ==============================================================================
// Floor Semantics for Negative Values
// ==============================================================================

TEST_F(DurationTest, FloorSemanticsNegativeHalfSecond) {
    // -0.5 seconds = {-1, 500e9 picos}
    auto d = Duration::from_picoseconds(-static_cast<int64_t>(PICOS_PER_SEC / 2));
    EXPECT_EQ(d.seconds(), -1);
    EXPECT_EQ(d.picoseconds(), PICOS_PER_SEC / 2);
    EXPECT_DOUBLE_EQ(d.to_seconds(), -0.5);
}

TEST_F(DurationTest, FloorSemanticsNegativeOneAndHalf) {
    // -1.5 seconds = {-2, 500e9 picos}
    int64_t picos = -ONE_SECOND_PICOS - static_cast<int64_t>(PICOS_PER_SEC / 2);
    auto d = Duration::from_picoseconds(picos);
    EXPECT_EQ(d.seconds(), -2);
    EXPECT_EQ(d.picoseconds(), PICOS_PER_SEC / 2);
    EXPECT_DOUBLE_EQ(d.to_seconds(), -1.5);
}

TEST_F(DurationTest, FloorSemanticsNegativeExactSecond) {
    // -1.0 seconds exactly = {-1, 0 picos}
    auto d = Duration::from_seconds(int64_t{-1});
    EXPECT_EQ(d.seconds(), -1);
    EXPECT_EQ(d.picoseconds(), 0);
}

// ==============================================================================
// Direct Factory Saturation
// ==============================================================================

TEST_F(DurationTest, FromSecondsSaturatesOnOverflow) {
    // INT64_MAX seconds exceeds ±68 year range
    auto d = Duration::from_seconds(std::numeric_limits<int64_t>::max());
    EXPECT_EQ(d, Duration::max());
    EXPECT_TRUE(saturated(d));
}

TEST_F(DurationTest, FromSecondsSaturatesOnUnderflow) {
    auto d = Duration::from_seconds(std::numeric_limits<int64_t>::min());
    EXPECT_EQ(d, Duration::min());
    EXPECT_TRUE(saturated(d));
}

TEST_F(DurationTest, FromSecondsWithinRange) {
    // 68 years in seconds (about 2.1e9 seconds, fits in int32_t)
    auto d = Duration::from_seconds(int64_t{2'000'000'000});
    EXPECT_EQ(d.seconds(), 2'000'000'000);
    EXPECT_FALSE(saturated(d));
}

// ==============================================================================
// Constexpr Factory (compile-time)
// ==============================================================================

TEST_F(DurationTest, ConstexprFactoryKnownSafe) {
    constexpr auto d1 = Duration::from_seconds(int64_t{1});
    constexpr auto d2 = Duration::from_milliseconds(int64_t{1000});
    constexpr auto d3 = Duration::from_microseconds(int64_t{1'000'000});
    constexpr auto d4 = Duration::from_nanoseconds(int64_t{1'000'000'000});

    EXPECT_EQ(d1.seconds(), 1);
    EXPECT_EQ(d2.seconds(), 1);
    EXPECT_EQ(d3.seconds(), 1);
    EXPECT_EQ(d4.seconds(), 1);
}

TEST_F(DurationTest, ConstexprMaxRange) {
    // ~68 years = INT32_MAX seconds
    constexpr auto d = Duration::from_seconds(int64_t{2'147'483'647});
    EXPECT_EQ(d.seconds(), std::numeric_limits<int32_t>::max());
}

// ==============================================================================
// Checked Factory: from_seconds(double)
// ==============================================================================

TEST_F(DurationTest, FromSecondsDoubleExact) {
    auto d = Duration::from_seconds(1.5);
    ASSERT_TRUE(d.has_value());
    EXPECT_EQ(d->seconds(), 1);
    EXPECT_EQ(d->picoseconds(), PICOS_PER_SEC / 2);
}

TEST_F(DurationTest, FromSecondsDoubleNegative) {
    auto d = Duration::from_seconds(-2.5);
    ASSERT_TRUE(d.has_value());
    EXPECT_EQ(d->seconds(), -3); // Floor semantics
    EXPECT_EQ(d->picoseconds(), PICOS_PER_SEC / 2);
}

TEST_F(DurationTest, FromSecondsDoubleZero) {
    auto d = Duration::from_seconds(0.0);
    ASSERT_TRUE(d.has_value());
    EXPECT_TRUE(d->is_zero());
}

TEST_F(DurationTest, FromSecondsDoubleSmall) {
    auto d = Duration::from_seconds(1e-12); // 1 picosecond
    ASSERT_TRUE(d.has_value());
    EXPECT_EQ(d->total_picoseconds(), 1);
}

TEST_F(DurationTest, FromSecondsDoubleOverflow) {
    auto d = Duration::from_seconds(1e15); // Way beyond ±68 years
    EXPECT_FALSE(d.has_value());
}

TEST_F(DurationTest, FromSecondsDoubleUnderflow) {
    auto d = Duration::from_seconds(-1e15);
    EXPECT_FALSE(d.has_value());
}

TEST_F(DurationTest, FromSecondsDoubleNaN) {
    auto d = Duration::from_seconds(std::nan(""));
    EXPECT_FALSE(d.has_value());
}

TEST_F(DurationTest, FromSecondsDoubleInfinity) {
    auto d = Duration::from_seconds(std::numeric_limits<double>::infinity());
    EXPECT_FALSE(d.has_value());
}

TEST_F(DurationTest, FromSecondsDoubleNearMaxValid) {
    // Value near INT32_MAX with fractional part - should be valid
    // INT32_MAX = 2147483647, int_part fits, frac doesn't cause carry
    double near_max = static_cast<double>(std::numeric_limits<int32_t>::max()) + 0.5;
    auto d = Duration::from_seconds(near_max);
    ASSERT_TRUE(d.has_value());
    EXPECT_EQ(d->seconds(), std::numeric_limits<int32_t>::max());
    EXPECT_EQ(d->picoseconds(), PICOS_PER_SEC / 2);
}

TEST_F(DurationTest, FromSecondsDoubleOverMaxRejects) {
    // Value where int_part exceeds INT32_MAX - should reject (prevents UB on cast)
    double over_max = static_cast<double>(std::numeric_limits<int32_t>::max()) + 1.0;
    auto d = Duration::from_seconds(over_max);
    EXPECT_FALSE(d.has_value());
}

TEST_F(DurationTest, FromSecondsDoubleNearMinWithBorrow) {
    // Value near INT32_MIN where negative fraction causes sec-- underflow
    // INT32_MIN = -2147483648, subtracting 0.1 would underflow
    double near_min = static_cast<double>(std::numeric_limits<int32_t>::min()) - 0.1;
    auto d = Duration::from_seconds(near_min);
    EXPECT_FALSE(d.has_value()); // Should reject due to potential underflow
}

TEST_F(DurationTest, FromSecondsDoubleAtMaxExact) {
    // Exactly at INT32_MAX should work (no carry needed)
    double at_max = static_cast<double>(std::numeric_limits<int32_t>::max());
    auto d = Duration::from_seconds(at_max);
    ASSERT_TRUE(d.has_value());
    EXPECT_EQ(d->seconds(), std::numeric_limits<int32_t>::max());
}

TEST_F(DurationTest, FromSecondsDoubleAtMinExact) {
    // Exactly at INT32_MIN should work (no borrow needed)
    double at_min = static_cast<double>(std::numeric_limits<int32_t>::min());
    auto d = Duration::from_seconds(at_min);
    ASSERT_TRUE(d.has_value());
    EXPECT_EQ(d->seconds(), std::numeric_limits<int32_t>::min());
}

// ==============================================================================
// Accessors
// ==============================================================================

TEST_F(DurationTest, SecondsAccessor) {
    auto d = Duration::from_seconds(int64_t{42});
    EXPECT_EQ(d.seconds(), 42);
}

TEST_F(DurationTest, PicosecondsAccessorIsSubsecond) {
    // 5.5 seconds
    auto d = Duration::from_seconds(int64_t{5}) +
             Duration::from_picoseconds(static_cast<int64_t>(PICOS_PER_SEC / 2));
    EXPECT_EQ(d.seconds(), 5);
    EXPECT_EQ(d.picoseconds(), PICOS_PER_SEC / 2);
}

TEST_F(DurationTest, TotalPicosecondsSmall) {
    auto d = Duration::from_picoseconds(12345);
    EXPECT_EQ(d.total_picoseconds(), 12345);
}

TEST_F(DurationTest, TotalPicosecondsSaturatesForLargeValues) {
    // Duration of 1 year = ~31.5e6 seconds
    // total_picoseconds would be > INT64_MAX for values > ~106 days
    auto d = Duration::from_seconds(int64_t{100'000'000}); // ~3.2 years
    // total_picoseconds() should saturate
    EXPECT_EQ(d.total_picoseconds(), std::numeric_limits<int64_t>::max());
}

TEST_F(DurationTest, ToSecondsDouble) {
    auto d = Duration::from_seconds(int64_t{5}) +
             Duration::from_picoseconds(static_cast<int64_t>(PICOS_PER_SEC / 2));
    EXPECT_DOUBLE_EQ(d.to_seconds(), 5.5);
}

// ==============================================================================
// Predicates
// ==============================================================================

TEST_F(DurationTest, IsZero) {
    EXPECT_TRUE(Duration::zero().is_zero());
    EXPECT_FALSE(Duration::from_picoseconds(1).is_zero());
    EXPECT_FALSE(Duration::from_picoseconds(-1).is_zero());
}

TEST_F(DurationTest, IsNegative) {
    EXPECT_TRUE(Duration::from_picoseconds(-1).is_negative());
    EXPECT_TRUE(Duration::min().is_negative());
    EXPECT_FALSE(Duration::zero().is_negative());
    EXPECT_FALSE(Duration::from_picoseconds(1).is_negative());
}

TEST_F(DurationTest, IsPositive) {
    EXPECT_TRUE(Duration::from_picoseconds(1).is_positive());
    EXPECT_TRUE(Duration::max().is_positive());
    EXPECT_FALSE(Duration::zero().is_positive());
    EXPECT_FALSE(Duration::from_picoseconds(-1).is_positive());
}

// ==============================================================================
// Absolute Value
// ==============================================================================

TEST_F(DurationTest, AbsPositive) {
    auto d = Duration::from_picoseconds(100);
    EXPECT_EQ(d.abs().total_picoseconds(), 100);
}

TEST_F(DurationTest, AbsNegative) {
    auto d = Duration::from_picoseconds(-100);
    EXPECT_EQ(d.abs().total_picoseconds(), 100);
}

TEST_F(DurationTest, AbsZero) {
    auto d = Duration::zero();
    EXPECT_TRUE(d.abs().is_zero());
}

TEST_F(DurationTest, AbsMinSaturates) {
    // abs(min) would overflow, should saturate to max
    auto d = Duration::min();
    EXPECT_EQ(d.abs(), Duration::max());
}

// ==============================================================================
// Unary Negation
// ==============================================================================

TEST_F(DurationTest, UnaryNegation) {
    auto d = Duration::from_picoseconds(100);
    auto neg = -d;
    EXPECT_EQ(neg.total_picoseconds(), -100);
}

TEST_F(DurationTest, UnaryNegationNegative) {
    auto d = Duration::from_picoseconds(-100);
    auto neg = -d;
    EXPECT_EQ(neg.total_picoseconds(), 100);
}

TEST_F(DurationTest, UnaryNegationMinSaturates) {
    // -min would overflow, should saturate to max
    auto d = Duration::min();
    EXPECT_EQ(-d, Duration::max());
}

TEST_F(DurationTest, UnaryNegationWithSubseconds) {
    // Negate 1.5 seconds
    auto d = Duration::from_seconds(int64_t{1}) +
             Duration::from_picoseconds(static_cast<int64_t>(PICOS_PER_SEC / 2));
    auto neg = -d;
    EXPECT_EQ(neg.seconds(), -2);
    EXPECT_EQ(neg.picoseconds(), PICOS_PER_SEC / 2);
    EXPECT_DOUBLE_EQ(neg.to_seconds(), -1.5);
}

// ==============================================================================
// Addition
// ==============================================================================

TEST_F(DurationTest, AdditionBasic) {
    auto a = Duration::from_picoseconds(100);
    auto b = Duration::from_picoseconds(50);
    EXPECT_EQ((a + b).total_picoseconds(), 150);
}

TEST_F(DurationTest, AdditionWithCarry) {
    // Adding subseconds that carry over
    auto a = Duration::from_picoseconds(static_cast<int64_t>(PICOS_PER_SEC - 100));
    auto b = Duration::from_picoseconds(200);
    auto result = a + b;
    EXPECT_EQ(result.seconds(), 1);
    EXPECT_EQ(result.picoseconds(), 100);
}

TEST_F(DurationTest, AdditionCompoundAssignment) {
    auto d = Duration::from_picoseconds(100);
    d += Duration::from_picoseconds(50);
    EXPECT_EQ(d.total_picoseconds(), 150);
}

TEST_F(DurationTest, AdditionSaturatesOnOverflow) {
    auto a = Duration::max();
    auto b = Duration::from_picoseconds(1);
    EXPECT_EQ(a + b, Duration::max());
    EXPECT_TRUE(saturated(a + b));
}

TEST_F(DurationTest, AdditionSaturatesOnUnderflow) {
    auto a = Duration::min();
    auto b = Duration::from_picoseconds(-1);
    EXPECT_EQ(a + b, Duration::min());
}

// ==============================================================================
// Subtraction
// ==============================================================================

TEST_F(DurationTest, SubtractionBasic) {
    auto a = Duration::from_picoseconds(100);
    auto b = Duration::from_picoseconds(30);
    EXPECT_EQ((a - b).total_picoseconds(), 70);
}

TEST_F(DurationTest, SubtractionWithBorrow) {
    // 1 second - 100 picos = 0 sec + (10^12 - 100) picos
    auto a = Duration::from_seconds(int64_t{1});
    auto b = Duration::from_picoseconds(100);
    auto result = a - b;
    EXPECT_EQ(result.seconds(), 0);
    EXPECT_EQ(result.picoseconds(), PICOS_PER_SEC - 100);
}

TEST_F(DurationTest, SubtractionCompoundAssignment) {
    auto d = Duration::from_picoseconds(100);
    d -= Duration::from_picoseconds(30);
    EXPECT_EQ(d.total_picoseconds(), 70);
}

TEST_F(DurationTest, SubtractionGoesNegative) {
    auto a = Duration::from_picoseconds(30);
    auto b = Duration::from_picoseconds(100);
    EXPECT_EQ((a - b).total_picoseconds(), -70);
}

TEST_F(DurationTest, SubtractionSaturatesOnOverflow) {
    // max - (-1) would overflow
    auto a = Duration::max();
    auto b = Duration::from_picoseconds(-1);
    EXPECT_EQ(a - b, Duration::max());
}

TEST_F(DurationTest, SubtractionSaturatesOnUnderflow) {
    // min - 1 would underflow
    auto a = Duration::min();
    auto b = Duration::from_picoseconds(1);
    EXPECT_EQ(a - b, Duration::min());
}

// ==============================================================================
// Multiplication
// ==============================================================================

TEST_F(DurationTest, MultiplicationBasic) {
    auto d = Duration::from_picoseconds(100);
    EXPECT_EQ((d * 5).total_picoseconds(), 500);
}

TEST_F(DurationTest, MultiplicationCommutative) {
    auto d = Duration::from_picoseconds(100);
    EXPECT_EQ((d * 5).total_picoseconds(), (5 * d).total_picoseconds());
}

TEST_F(DurationTest, MultiplicationByZero) {
    auto d = Duration::from_seconds(int64_t{1000});
    EXPECT_TRUE((d * 0).is_zero());
}

TEST_F(DurationTest, MultiplicationByOne) {
    auto d = Duration::from_picoseconds(12345);
    EXPECT_EQ((d * 1).total_picoseconds(), 12345);
}

TEST_F(DurationTest, MultiplicationByNegativeOne) {
    auto d = Duration::from_picoseconds(100);
    EXPECT_EQ((d * -1).total_picoseconds(), -100);
}

TEST_F(DurationTest, MultiplicationCompoundAssignment) {
    auto d = Duration::from_picoseconds(100);
    d *= 3;
    EXPECT_EQ(d.total_picoseconds(), 300);
}

TEST_F(DurationTest, MultiplicationSaturatesOnOverflow) {
    auto d = Duration::from_seconds(int64_t{1'000'000'000});
    EXPECT_EQ(d * 100, Duration::max());
}

TEST_F(DurationTest, MultiplicationSaturatesOnUnderflow) {
    auto d = Duration::from_seconds(int64_t{-1'000'000'000});
    EXPECT_EQ(d * 100, Duration::min());
}

TEST_F(DurationTest, MultiplicationMinByNegativeOneSaturates) {
    auto d = Duration::min();
    EXPECT_EQ(d * -1, Duration::max());
}

// ==============================================================================
// Division by Scalar
// ==============================================================================

TEST_F(DurationTest, DivisionBasic) {
    auto d = Duration::from_picoseconds(100);
    EXPECT_EQ((d / 5).total_picoseconds(), 20);
}

TEST_F(DurationTest, DivisionTruncates) {
    auto d = Duration::from_picoseconds(103);
    EXPECT_EQ((d / 5).total_picoseconds(), 20); // 103/5 = 20.6, truncates to 20
}

TEST_F(DurationTest, DivisionByOne) {
    auto d = Duration::from_picoseconds(12345);
    EXPECT_EQ((d / 1).total_picoseconds(), 12345);
}

TEST_F(DurationTest, DivisionByNegativeOne) {
    auto d = Duration::from_picoseconds(100);
    EXPECT_EQ((d / -1).total_picoseconds(), -100);
}

TEST_F(DurationTest, DivisionCompoundAssignment) {
    auto d = Duration::from_picoseconds(300);
    d /= 3;
    EXPECT_EQ(d.total_picoseconds(), 100);
}

TEST_F(DurationTest, DivisionByZeroSaturatesToMax) {
    // Division by zero saturates to max (positive dividend)
    auto d = Duration::from_picoseconds(100);
    EXPECT_EQ(d / 0, Duration::max());
}

TEST_F(DurationTest, DivisionByZeroNegativeSaturatesToMin) {
    auto d = Duration::from_picoseconds(-100);
    EXPECT_EQ(d / 0, Duration::min());
}

// ==============================================================================
// Division of Durations
// ==============================================================================

TEST_F(DurationTest, DurationDivisionBasic) {
    auto a = Duration::from_picoseconds(1000);
    auto b = Duration::from_picoseconds(100);
    EXPECT_EQ(a / b, 10);
}

TEST_F(DurationTest, DurationDivisionTruncates) {
    auto a = Duration::from_picoseconds(1050);
    auto b = Duration::from_picoseconds(100);
    EXPECT_EQ(a / b, 10);
}

TEST_F(DurationTest, DurationDivisionByZeroSaturatesToMax) {
    auto a = Duration::from_picoseconds(100);
    auto b = Duration::zero();
    EXPECT_EQ(a / b, std::numeric_limits<int64_t>::max());
}

TEST_F(DurationTest, DurationDivisionByZeroNegativeDividendSaturatesToMin) {
    auto a = Duration::from_picoseconds(-100);
    auto b = Duration::zero();
    EXPECT_EQ(a / b, std::numeric_limits<int64_t>::min());
}

TEST_F(DurationTest, DurationDivisionRealWorld) {
    // How many 10MHz samples fit in 1 second?
    auto one_second = Duration::from_seconds(int64_t{1});
    auto sample_period = Duration::from_picoseconds(100'000); // 10 MHz = 100 ns
    EXPECT_EQ(one_second / sample_period, 10'000'000);
}

// ==============================================================================
// Comparison
// ==============================================================================

TEST_F(DurationTest, Equality) {
    auto a = Duration::from_picoseconds(100);
    auto b = Duration::from_picoseconds(100);
    auto c = Duration::from_picoseconds(101);

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST_F(DurationTest, Ordering) {
    auto a = Duration::from_picoseconds(100);
    auto b = Duration::from_picoseconds(200);

    EXPECT_LT(a, b);
    EXPECT_LE(a, b);
    EXPECT_LE(a, a);
    EXPECT_GT(b, a);
    EXPECT_GE(b, a);
    EXPECT_GE(a, a);
}

TEST_F(DurationTest, OrderingWithNegative) {
    auto neg = Duration::from_picoseconds(-100);
    auto zero = Duration::zero();
    auto pos = Duration::from_picoseconds(100);

    EXPECT_LT(neg, zero);
    EXPECT_LT(zero, pos);
    EXPECT_LT(neg, pos);
}

TEST_F(DurationTest, OrderingWithSeconds) {
    auto one_sec = Duration::from_seconds(int64_t{1});
    auto half_sec = Duration::from_picoseconds(static_cast<int64_t>(PICOS_PER_SEC / 2));

    EXPECT_GT(one_sec, half_sec);
}

// ==============================================================================
// Saturated Helper
// ==============================================================================

TEST_F(DurationTest, SaturatedDetectsMax) {
    EXPECT_TRUE(saturated(Duration::max()));
}

TEST_F(DurationTest, SaturatedDetectsMin) {
    EXPECT_TRUE(saturated(Duration::min()));
}

TEST_F(DurationTest, SaturatedFalseForNormalValues) {
    EXPECT_FALSE(saturated(Duration::zero()));
    EXPECT_FALSE(saturated(Duration::from_seconds(int64_t{1000})));
    EXPECT_FALSE(saturated(Duration::from_seconds(int64_t{-1000})));
}

TEST_F(DurationTest, SaturatedAfterOverflow) {
    auto d = Duration::max() + Duration::from_picoseconds(1);
    EXPECT_TRUE(saturated(d));
}

// ==============================================================================
// Real-World Scenarios
// ==============================================================================

TEST_F(DurationTest, RealWorldOneYear) {
    // 1 year = ~31.5 million seconds - well within 68-year range
    constexpr int64_t seconds_per_year = 365LL * 24 * 3600;
    auto d = Duration::from_seconds(seconds_per_year);
    EXPECT_EQ(d.seconds(), seconds_per_year);
    EXPECT_FALSE(saturated(d));
}

TEST_F(DurationTest, RealWorld50Years) {
    // 50 years - within range
    constexpr int64_t seconds_per_year = 365LL * 24 * 3600;
    auto d = Duration::from_seconds(50 * seconds_per_year);
    EXPECT_FALSE(saturated(d));
    EXPECT_NEAR(d.to_seconds(), 50.0 * seconds_per_year, 1.0);
}

TEST_F(DurationTest, RealWorldSampleCounting) {
    // Counting samples at 10 MHz for 1 hour
    // 10 MHz = 10^7 samples/sec, 1 hour = 3600 sec = 3.6e10 samples
    // Each sample is 100 ns = 100,000 ps
    auto sample_period = Duration::from_picoseconds(100'000);
    auto one_hour = Duration::from_seconds(int64_t{3600});

    int64_t sample_count = one_hour / sample_period;
    EXPECT_EQ(sample_count, 36'000'000'000LL);
}

// ==============================================================================
// from_samples Factory
// ==============================================================================

TEST_F(DurationTest, FromSamplesBasic) {
    // 10 samples at 100,000 ps each = 1,000,000 ps
    auto period = SamplePeriod::from_picoseconds(100'000);
    auto result = Duration::from_samples(10, period);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->total_picoseconds(), 1'000'000);
}

TEST_F(DurationTest, FromSamplesNegativeCount) {
    // Negative sample count is valid (represents going backward in time)
    auto period = SamplePeriod::from_picoseconds(100'000);
    auto result = Duration::from_samples(-10, period);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->total_picoseconds(), -1'000'000);
}

TEST_F(DurationTest, FromSamplesOverflow) {
    // Many samples at a large period should saturate, which from_samples detects
    auto period = SamplePeriod::from_picoseconds(1'000'000'000'000); // 1 second
    auto result = Duration::from_samples(std::numeric_limits<int64_t>::max(), period);

    // Should return nullopt when saturation is detected
    EXPECT_FALSE(result.has_value());
}

TEST_F(DurationTest, FromSamplesLargeButValid) {
    // 1 billion samples at 1 microsecond = 1000 seconds (fits easily)
    auto period = SamplePeriod::from_picoseconds(1'000'000); // 1 us
    auto result = Duration::from_samples(1'000'000'000, period);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->seconds(), 1000);
}

// ==============================================================================
// ShortDuration Tests
// ==============================================================================

TEST_F(DurationTest, ShortDurationDefaultConstruction) {
    ShortDuration d;
    EXPECT_EQ(d.picoseconds(), 0);
    EXPECT_TRUE(d.is_zero());
}

TEST_F(DurationTest, ShortDurationFromPicoseconds) {
    auto d = ShortDuration::from_picoseconds(1'000'000'000'000); // 1 second
    EXPECT_EQ(d.picoseconds(), 1'000'000'000'000);
    EXPECT_TRUE(d.is_positive());
}

TEST_F(DurationTest, ShortDurationNegative) {
    auto d = ShortDuration::from_picoseconds(-500'000'000'000); // -0.5 seconds
    EXPECT_EQ(d.picoseconds(), -500'000'000'000);
    EXPECT_TRUE(d.is_negative());
}

TEST_F(DurationTest, ShortDurationMaxMin) {
    EXPECT_EQ(ShortDuration::max().picoseconds(), std::numeric_limits<int64_t>::max());
    EXPECT_EQ(ShortDuration::min().picoseconds(), std::numeric_limits<int64_t>::min());
}

TEST_F(DurationTest, ShortDurationAddition) {
    auto a = ShortDuration::from_picoseconds(500'000'000'000);
    auto b = ShortDuration::from_picoseconds(700'000'000'000);
    auto result = a + b;
    EXPECT_EQ(result.picoseconds(), 1'200'000'000'000);
}

TEST_F(DurationTest, ShortDurationSubtraction) {
    auto a = ShortDuration::from_picoseconds(1'000'000'000'000);
    auto b = ShortDuration::from_picoseconds(300'000'000'000);
    auto result = a - b;
    EXPECT_EQ(result.picoseconds(), 700'000'000'000);
}

TEST_F(DurationTest, ShortDurationMultiplication) {
    auto d = ShortDuration::from_picoseconds(100'000'000'000);
    auto result = d * 5;
    EXPECT_EQ(result.picoseconds(), 500'000'000'000);
}

TEST_F(DurationTest, ShortDurationDivision) {
    auto d = ShortDuration::from_picoseconds(1'000'000'000'000);
    auto result = d / 4;
    EXPECT_EQ(result.picoseconds(), 250'000'000'000);
}

TEST_F(DurationTest, ShortDurationDivisionRatio) {
    auto a = ShortDuration::from_picoseconds(1'000'000'000'000);
    auto b = ShortDuration::from_picoseconds(100'000'000'000);
    EXPECT_EQ(a / b, 10);
}

TEST_F(DurationTest, ShortDurationAdditionOverflow) {
    auto a = ShortDuration::max();
    auto b = ShortDuration::from_picoseconds(1);
    auto result = a + b;
    EXPECT_TRUE(saturated(result));
    EXPECT_EQ(result, ShortDuration::max());
}

TEST_F(DurationTest, ShortDurationSubtractionUnderflow) {
    auto a = ShortDuration::min();
    auto b = ShortDuration::from_picoseconds(1);
    auto result = a - b;
    EXPECT_TRUE(saturated(result));
    EXPECT_EQ(result, ShortDuration::min());
}

TEST_F(DurationTest, ShortDurationMultiplicationOverflow) {
    auto d = ShortDuration::from_picoseconds(std::numeric_limits<int64_t>::max() / 2);
    auto result = d * 3;
    EXPECT_TRUE(saturated(result));
    EXPECT_EQ(result, ShortDuration::max());
}

TEST_F(DurationTest, ShortDurationDivisionByZero) {
    auto d = ShortDuration::from_picoseconds(1'000'000);
    auto result = d / 0;
    EXPECT_TRUE(saturated(result));
    EXPECT_EQ(result, ShortDuration::max());

    auto neg = ShortDuration::from_picoseconds(-1'000'000);
    auto neg_result = neg / 0;
    EXPECT_EQ(neg_result, ShortDuration::min());
}

TEST_F(DurationTest, ShortDurationDivisionRatioByZero) {
    auto a = ShortDuration::from_picoseconds(1'000'000);
    auto b = ShortDuration::zero();
    EXPECT_EQ(a / b, std::numeric_limits<int64_t>::max());
}

TEST_F(DurationTest, ShortDurationNegation) {
    auto d = ShortDuration::from_picoseconds(1'000'000);
    auto neg = -d;
    EXPECT_EQ(neg.picoseconds(), -1'000'000);
}

TEST_F(DurationTest, ShortDurationNegationMinSaturates) {
    auto result = -ShortDuration::min();
    EXPECT_EQ(result, ShortDuration::max()); // Saturate: -MIN would overflow
}

TEST_F(DurationTest, ShortDurationToDuration) {
    auto sd = ShortDuration::from_picoseconds(1'500'000'000'000); // 1.5 seconds
    auto d = sd.to_duration();
    EXPECT_EQ(d.seconds(), 1);
    EXPECT_EQ(d.picoseconds(), 500'000'000'000);
}

TEST_F(DurationTest, ShortDurationToDurationNegative) {
    auto sd = ShortDuration::from_picoseconds(-1'500'000'000'000); // -1.5 seconds
    auto d = sd.to_duration();
    EXPECT_EQ(d.seconds(), -2); // Floor semantics
    EXPECT_EQ(d.picoseconds(), 500'000'000'000);
}

TEST_F(DurationTest, ShortDurationFromDuration) {
    auto d = Duration::from_seconds(int64_t{10}) + Duration::from_picoseconds(500'000'000'000);
    auto sd = ShortDuration::from_duration(d);
    EXPECT_EQ(sd.picoseconds(), 10'500'000'000'000);
}

TEST_F(DurationTest, ShortDurationFromDurationSaturates) {
    // Duration::max() exceeds ShortDuration range, should saturate
    auto sd = ShortDuration::from_duration(Duration::max());
    EXPECT_EQ(sd, ShortDuration::max());

    auto sd_min = ShortDuration::from_duration(Duration::min());
    EXPECT_EQ(sd_min, ShortDuration::min());
}

TEST_F(DurationTest, DurationToShortDurationValid) {
    auto d = Duration::from_seconds(int64_t{1000});
    auto sd = d.to_short_duration();
    ASSERT_TRUE(sd.has_value());
    EXPECT_EQ(sd->picoseconds(), 1000'000'000'000'000);
}

TEST_F(DurationTest, DurationToShortDurationOutOfRange) {
    // Duration::max() (~68 years) exceeds ShortDuration's ~106 day range
    auto sd = Duration::max().to_short_duration();
    EXPECT_FALSE(sd.has_value());

    auto sd_min = Duration::min().to_short_duration();
    EXPECT_FALSE(sd_min.has_value());
}

TEST_F(DurationTest, DurationToShortDurationRoundTrip) {
    // Values within ShortDuration range should round-trip correctly
    auto original = Duration::from_seconds(int64_t{86400}); // 1 day
    auto sd = original.to_short_duration();
    ASSERT_TRUE(sd.has_value());

    auto back = sd->to_duration();
    EXPECT_EQ(back, original);
}

// ==============================================================================
// normalize() Tests (via from_picoseconds)
// ==============================================================================

TEST_F(DurationTest, NormalizeCarry) {
    // Large picoseconds value triggers carry
    auto d = Duration::from_picoseconds(2'500'000'000'000); // 2.5 seconds
    EXPECT_EQ(d.seconds(), 2);
    EXPECT_EQ(d.picoseconds(), 500'000'000'000);
}

TEST_F(DurationTest, NormalizeBorrow) {
    // Negative picoseconds triggers borrow
    auto d = Duration::from_picoseconds(-500'000'000'000); // -0.5 seconds
    EXPECT_EQ(d.seconds(), -1);                            // Floor semantics
    EXPECT_EQ(d.picoseconds(), 500'000'000'000);
}

TEST_F(DurationTest, NormalizeFloorSemantics) {
    // -1.5 seconds should be {-2, 500e9}
    auto d = Duration::from_picoseconds(-1'500'000'000'000);
    EXPECT_EQ(d.seconds(), -2);
    EXPECT_EQ(d.picoseconds(), 500'000'000'000);
}

TEST_F(DurationTest, NormalizeInt64Min) {
    // INT64_MIN picoseconds is ~-107 days, well within Duration's ±68 year range
    // This test verifies no UB occurs (the tricky -INT64_MIN case)
    auto d = Duration::from_picoseconds(std::numeric_limits<int64_t>::min());
    // Should NOT saturate - it's only ~107 days, not 68 years
    EXPECT_FALSE(saturated(d));
    // Verify it's a large negative value (~-107 days = ~-9.2 million seconds)
    EXPECT_TRUE(d.is_negative());
    EXPECT_LT(d.seconds(), -9'000'000); // Less than -9M seconds
}

TEST_F(DurationTest, NormalizeInt64Max) {
    // INT64_MAX should work (it's ~106 days)
    auto d = Duration::from_picoseconds(std::numeric_limits<int64_t>::max());
    // Should not saturate - this is within Duration's 68-year range
    EXPECT_FALSE(saturated(d));
}

TEST_F(DurationTest, NormalizeLargeNegative) {
    // Large negative value near boundary
    int64_t picos = -100 * 1'000'000'000'000LL; // -100 seconds
    auto d = Duration::from_picoseconds(picos);
    EXPECT_EQ(d.seconds(), -100);
    EXPECT_EQ(d.picoseconds(), 0);
}

// =============================================================================
// ShortDuration::from_samples tests
// =============================================================================

TEST_F(DurationTest, ShortDurationFromSamplesBasic) {
    // 1000 samples at 1 MHz (1 microsecond per sample) = 1 millisecond
    auto period = SamplePeriod::from_rate_hz(1e6);
    ASSERT_TRUE(period.has_value());

    auto d = ShortDuration::from_samples(1000, *period);
    // 1000 samples * 1e6 picos = 1e9 picos = 1 millisecond
    EXPECT_EQ(d.picoseconds(), 1'000'000'000LL);
}

TEST_F(DurationTest, ShortDurationFromSamplesNegative) {
    // Negative sample count = backward in time
    auto period = SamplePeriod::from_rate_hz(1e6);
    ASSERT_TRUE(period.has_value());

    auto d = ShortDuration::from_samples(-500, *period);
    EXPECT_EQ(d.picoseconds(), -500'000'000LL); // -500 microseconds
}

TEST_F(DurationTest, ShortDurationFromSamplesZero) {
    auto period = SamplePeriod::from_rate_hz(1e6);
    ASSERT_TRUE(period.has_value());

    auto d = ShortDuration::from_samples(0, *period);
    EXPECT_EQ(d.picoseconds(), 0);
    EXPECT_TRUE(d.is_zero());
}

TEST_F(DurationTest, ShortDurationFromSamplesOverflowPositive) {
    // Very large count that would overflow
    auto period = SamplePeriod::from_picoseconds(1'000'000'000'000ULL); // 1 second per sample
    auto d = ShortDuration::from_samples(std::numeric_limits<int64_t>::max(), period);

    // Should saturate to max
    EXPECT_EQ(d, ShortDuration::max());
}

TEST_F(DurationTest, ShortDurationFromSamplesOverflowNegative) {
    // Very large negative count that would overflow
    auto period = SamplePeriod::from_picoseconds(1'000'000'000'000ULL); // 1 second per sample
    auto d = ShortDuration::from_samples(std::numeric_limits<int64_t>::min(), period);

    // Should saturate to min
    EXPECT_EQ(d, ShortDuration::min());
}

TEST_F(DurationTest, ShortDurationFromSamplesHighRate) {
    // High sample rate: 100 GHz (10 ps per sample)
    auto period = SamplePeriod::from_picoseconds(10);
    auto d = ShortDuration::from_samples(1'000'000, period);

    // 1M samples * 10 picos = 10M picos = 10 microseconds
    EXPECT_EQ(d.picoseconds(), 10'000'000LL);
}
