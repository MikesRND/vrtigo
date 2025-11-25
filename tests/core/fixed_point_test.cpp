#include "vrtigo/detail/fixed_point.hpp"

#include <iostream>
#include <limits>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>

using vrtigo::detail::FixedPoint;

namespace {

// Small helper for approximate double comparisons
bool almost_equal(double a, double b, double eps = 1e-12) {
    return std::fabs(a - b) <= eps * (1.0 + std::max(std::fabs(a), std::fabs(b)));
}

} // namespace

// -----------------------------------------------------------------------------
// 1. Exact round-trip tests for small formats
//    (total_bits small enough that every value is exactly representable in double)
// -----------------------------------------------------------------------------

TEST(FixedPointTest, RoundtripUnsignedSmall) {
    using F = FixedPoint<4, 4, false>; // UQ4.4: total_bits = 8
    static_assert(F::total_bits == 8);

    for (std::uint64_t raw = 0; raw < (1ull << F::total_bits); ++raw) {
        std::uint64_t sanitized = F::sanitize_raw(raw);
        ASSERT_EQ(sanitized, raw);

        double v = F::to_double(raw);
        // Should be within representable range
        ASSERT_GE(v, F::min_value() - 1e-12);
        ASSERT_LE(v, F::max_value() + 1e-12);

        std::uint64_t raw2 = F::from_double(v);
        auto strict = F::from_double_strict(v);
        ASSERT_TRUE(strict.has_value());

        // For small total_bits, double holds all values exactly. Round-trip must match.
        ASSERT_EQ(raw2, sanitized);
        ASSERT_EQ(*strict, sanitized);
    }
}

TEST(FixedPointTest, RoundtripSignedSmall) {
    using F = FixedPoint<3, 4, true>; // Q3.4: total_bits = 7, range [-4, 3.9375]
    static_assert(F::total_bits == 7);

    for (std::uint64_t raw = 0; raw < (1ull << F::total_bits); ++raw) {
        std::uint64_t sanitized = F::sanitize_raw(raw);
        ASSERT_EQ(sanitized, raw);

        double v = F::to_double(raw);
        ASSERT_GE(v, F::min_value() - 1e-12);
        ASSERT_LE(v, F::max_value() + 1e-12);

        std::uint64_t raw2 = F::from_double(v);
        auto strict = F::from_double_strict(v);
        ASSERT_TRUE(strict.has_value());

        // Same rationale: small format, exact round-trip.
        ASSERT_EQ(raw2, sanitized);
        ASSERT_EQ(*strict, sanitized);
    }
}

// -----------------------------------------------------------------------------
// 2. Saturation behavior tests
// -----------------------------------------------------------------------------

TEST(FixedPointTest, SaturationUnsigned) {
    using F = FixedPoint<4, 4, false>;   // UQ4.4
    const double min_v = F::min_value(); // 0
    const double max_v = F::max_value(); // 15.9375

    // Inside range
    {
        double v = 5.25; // representable
        auto raw_sat = F::from_double(v);
        auto raw_strict = F::from_double_strict(v);
        ASSERT_TRUE(raw_strict.has_value());
        ASSERT_EQ(raw_sat, *raw_strict);
        double back = F::to_double(raw_sat);
        ASSERT_DOUBLE_EQ(back, v);
    }

    // Above max
    {
        double v = 20.0;                            // above max_v
        auto raw_sat = F::from_double(v);           // should saturate to max
        auto raw_strict = F::from_double_strict(v); // should return nullopt
        ASSERT_FALSE(raw_strict.has_value());

        double back = F::to_double(raw_sat);
        ASSERT_TRUE(almost_equal(back, max_v, 1e-10));
    }

    // Below min (unsigned, so 0)
    {
        double v = -5.0;                            // below min
        auto raw_sat = F::from_double(v);           // should saturate to 0
        auto raw_strict = F::from_double_strict(v); // should return nullopt
        ASSERT_FALSE(raw_strict.has_value());

        double back = F::to_double(raw_sat);
        ASSERT_DOUBLE_EQ(back, 0.0);
    }

    // NaN
    {
        double v = std::numeric_limits<double>::quiet_NaN();
        auto raw_sat = F::from_double(v);           // returns 0
        auto raw_strict = F::from_double_strict(v); // returns nullopt
        ASSERT_FALSE(raw_strict.has_value());
        ASSERT_EQ(raw_sat, 0u);
    }

    // Inf
    {
        double v = std::numeric_limits<double>::infinity();
        auto raw_sat = F::from_double(v);           // returns 0
        auto raw_strict = F::from_double_strict(v); // returns nullopt
        ASSERT_FALSE(raw_strict.has_value());
        ASSERT_EQ(raw_sat, 0u);
    }
}

TEST(FixedPointTest, SaturationSigned) {
    using F = FixedPoint<4, 4, true>;    // Q4.4: range [-8, 7.9375]
    const double min_v = F::min_value(); // -8
    const double max_v = F::max_value(); // 7.9375

    // Inside range
    {
        double v = -3.25;
        auto raw_sat = F::from_double(v);
        auto raw_strict = F::from_double_strict(v);
        ASSERT_TRUE(raw_strict.has_value());
        ASSERT_EQ(raw_sat, *raw_strict);
        double back = F::to_double(raw_sat);
        ASSERT_DOUBLE_EQ(back, v);
    }

    // Above max
    {
        double v = 10.0;
        auto raw_sat = F::from_double(v);           // should saturate to max
        auto raw_strict = F::from_double_strict(v); // should return nullopt
        ASSERT_FALSE(raw_strict.has_value());

        double back = F::to_double(raw_sat);
        ASSERT_TRUE(almost_equal(back, max_v, 1e-10));
    }

    // Below min
    {
        double v = -10.0;
        auto raw_sat = F::from_double(v);           // should saturate to min
        auto raw_strict = F::from_double_strict(v); // should return nullopt
        ASSERT_FALSE(raw_strict.has_value());

        double back = F::to_double(raw_sat);
        ASSERT_TRUE(almost_equal(back, min_v, 1e-10));
    }
}

// -----------------------------------------------------------------------------
// 3. Banker's rounding (round-to-nearest-even)
// -----------------------------------------------------------------------------

TEST(FixedPointTest, BankersRounding) {
    using F = FixedPoint<8, 8, false>; // UQ8.8

    // 10.5 in Q8.8: should round to 10 (even)
    {
        double v = 10.5 / 256.0; // 10.5 at the quantum level
        auto raw = F::from_double(v);
        double back = F::to_double(raw);
        // Should round to 10/256 (even)
        ASSERT_DOUBLE_EQ(back, 10.0 / 256.0);
    }

    // 11.5 in Q8.8: should round to 12 (even)
    {
        double v = 11.5 / 256.0; // 11.5 at the quantum level
        auto raw = F::from_double(v);
        double back = F::to_double(raw);
        // Should round to 12/256 (even)
        ASSERT_DOUBLE_EQ(back, 12.0 / 256.0);
    }
}

// -----------------------------------------------------------------------------
// 4. VITA 49.2 Q52.12 format tests
// -----------------------------------------------------------------------------

TEST(FixedPointTest, Q52_12_Format) {
    // Q52.12: 52 integer bits, 12 fractional bits, unsigned
    using Q52_12 = FixedPoint<52, 12, false>;
    // Scale factor: 2^12 = 4096

    // Test some typical values
    {
        double v = 10000000.0; // 10 MHz sample rate
        auto raw = Q52_12::from_double(v);
        double back = Q52_12::to_double(raw);
        ASSERT_TRUE(almost_equal(back, v, 1e-6));
    }

    // Test with fractional part
    {
        double v = 1234567.890625; // Has exact representation in Q52.12
        auto raw = Q52_12::from_double(v);
        double back = Q52_12::to_double(raw);
        ASSERT_DOUBLE_EQ(back, v);
    }

    // Test the specific conversion from current code
    {
        // Current code: value * 4096.0
        double hz = 10000000.0;
        uint64_t old_raw = static_cast<uint64_t>(hz * 4096.0 + 0.5);

        // New code
        uint64_t new_raw = Q52_12::from_double(hz);

        // Should be very close (banker's rounding vs round-half-up)
        ASSERT_LE(std::abs(static_cast<int64_t>(new_raw - old_raw)), 1);
    }
}

// -----------------------------------------------------------------------------
// 5. VITA 49.2 Q44.20 format tests
// -----------------------------------------------------------------------------

TEST(FixedPointTest, Q44_20_Format) {
    // Q44.20: 44 integer bits, 20 fractional bits, signed (two's complement)
    using Q44_20 = FixedPoint<44, 20, true>;
    // Scale factor: 2^20 = 1048576

    // Test positive frequency
    {
        double v = 2400000000.0; // 2.4 GHz
        auto raw = Q44_20::from_double(v);
        double back = Q44_20::to_double(raw);
        ASSERT_TRUE(almost_equal(back, v, 1e-6));
    }

    // Test negative frequency
    {
        double v = -10700000.0; // -10.7 MHz
        auto raw = Q44_20::from_double(v);
        double back = Q44_20::to_double(raw);
        ASSERT_TRUE(almost_equal(back, v, 1e-6));
    }

    // Test spec examples from VITA 49.2
    {
        // 0x0000000000100000 = 1 Hz
        uint64_t raw = 0x0000000000100000ULL;
        double v = Q44_20::to_double(raw);
        ASSERT_NEAR(v, 1.0, 1e-12);

        // 0xFFFFFFFFFFF00000 = -1 Hz (two's complement)
        raw = 0xFFFFFFFFFFF00000ULL;
        v = Q44_20::to_double(raw);
        ASSERT_NEAR(v, -1.0, 1e-12);

        // 0x0000000000000001 = 0.95 µHz (minimum positive)
        raw = 0x0000000000000001ULL;
        v = Q44_20::to_double(raw);
        ASSERT_NEAR(v, 9.5367431640625e-7, 1e-15);

        // 0xFFFFFFFFFFFFFFFF = -0.95 µHz (minimum magnitude negative)
        raw = 0xFFFFFFFFFFFFFFFFULL;
        v = Q44_20::to_double(raw);
        ASSERT_NEAR(v, -9.5367431640625e-7, 1e-15);
    }

    // Test the specific conversion from current code
    {
        double hz = 10700000.0; // 10.7 MHz

        // Current code pattern
        int64_t old_signed = static_cast<int64_t>(hz * 1048576.0 + 0.5);
        uint64_t old_raw = static_cast<uint64_t>(old_signed);

        // New code
        uint64_t new_raw = Q44_20::from_double(hz);

        // Should be very close
        ASSERT_LE(std::abs(static_cast<int64_t>(new_raw - old_raw)), 1);
    }
}

// -----------------------------------------------------------------------------
// 6. Sign extension tests (for formats < 64 bits)
// -----------------------------------------------------------------------------

TEST(FixedPointTest, SignExtension) {
    // Test a 16-bit signed format stored in uint64_t
    using Q8_8 = FixedPoint<8, 8, true>; // Total 16 bits

    // Negative value: -1.5
    {
        double v = -1.5;
        auto raw = Q8_8::from_double(v);

        // Raw should be 0xFFFFFFFFFFFFFE80 after sign extension
        // But stored in 16 bits would be 0xFE80
        ASSERT_EQ(raw & 0xFFFF, 0xFE80);

        double back = Q8_8::to_double(raw);
        ASSERT_DOUBLE_EQ(back, v);
    }

    // Test sign extension works correctly
    {
        // Create a negative value in 16 bits
        uint64_t raw = 0x8000; // -128 in Q8.8
        double v = Q8_8::to_double(raw);
        ASSERT_DOUBLE_EQ(v, -128.0);
    }
}

// -----------------------------------------------------------------------------
// 7. Edge cases and limits
// -----------------------------------------------------------------------------

TEST(FixedPointTest, EdgeCases) {
    // Test zero fractional bits
    using IntOnly = FixedPoint<16, 0, false>;
    {
        double v = 1234.0;
        auto raw = IntOnly::from_double(v);
        double back = IntOnly::to_double(raw);
        ASSERT_DOUBLE_EQ(back, v);
    }

    // Test zero integer bits (pure fractional)
    using FracOnly = FixedPoint<0, 16, false>;
    {
        double v = 0.75;
        auto raw = FracOnly::from_double(v);
        double back = FracOnly::to_double(raw);
        ASSERT_DOUBLE_EQ(back, v);
    }

    // Test 64-bit format (maximum size)
    using Full64 = FixedPoint<32, 32, true>;
    static_assert(Full64::total_bits == 64);
    {
        double v = 1234567.890625;
        auto raw = Full64::from_double(v);
        double back = Full64::to_double(raw);
        ASSERT_NEAR(back, v, 1e-6);
    }
}

// -----------------------------------------------------------------------------
// 8. Min/Max value tests
// -----------------------------------------------------------------------------

TEST(FixedPointTest, MinMaxValues) {
    // Unsigned format
    {
        using F = FixedPoint<8, 8, false>; // UQ8.8
        ASSERT_DOUBLE_EQ(F::min_value(), 0.0);
        ASSERT_NEAR(F::max_value(), 256.0 - 1.0 / 256.0, 1e-10);
    }

    // Signed format
    {
        using F = FixedPoint<8, 8, true>; // Q8.8
        ASSERT_DOUBLE_EQ(F::min_value(), -128.0);
        ASSERT_NEAR(F::max_value(), 128.0 - 1.0 / 256.0, 1e-10);
    }

    // Q44.20
    {
        using Q44_20 = FixedPoint<44, 20, true>;
        // -2^43 to 2^43 - 2^-20
        ASSERT_DOUBLE_EQ(Q44_20::min_value(), -8796093022208.0);
        ASSERT_NEAR(Q44_20::max_value(), 8796093022208.0 - 9.5367431640625e-7, 1.0);
    }

    // Q52.12
    {
        using Q52_12 = FixedPoint<52, 12, false>;
        // 0 to 2^52 - 2^-12
        ASSERT_DOUBLE_EQ(Q52_12::min_value(), 0.0);
        ASSERT_NEAR(Q52_12::max_value(), 4503599627370496.0 - 1.0 / 4096.0, 1.0);
    }
}

// -----------------------------------------------------------------------------
// 9. Storage type parameter tests
// -----------------------------------------------------------------------------

TEST(FixedPointTest, StorageType_16bit) {
    // Q9.7 with uint16_t storage (Reference Level format)
    using Q9_7_16 = FixedPoint<9, 7, true, uint16_t>;

    // Test round-trip for common dBm values
    double test_values[] = {-50.0, -10.0, 0.0, 10.0, 20.0};

    for (double v : test_values) {
        uint16_t raw = Q9_7_16::from_double(v);
        double back = Q9_7_16::to_double(raw);
        ASSERT_NEAR(back, v, 0.01); // Within 0.01 dBm
    }

    // Test bounds
    ASSERT_DOUBLE_EQ(Q9_7_16::min_value(), -256.0);
    ASSERT_NEAR(Q9_7_16::max_value(), 255.9921875, 0.001);

    // Test saturation
    uint16_t over = Q9_7_16::from_double(300.0);
    ASSERT_NEAR(Q9_7_16::to_double(over), 255.9921875, 0.001);

    uint16_t under = Q9_7_16::from_double(-300.0);
    ASSERT_NEAR(Q9_7_16::to_double(under), -256.0, 0.001);
}

TEST(FixedPointTest, StorageType_32bit) {
    // Q20.12 with uint32_t storage
    using Q20_12_32 = FixedPoint<20, 12, false, uint32_t>;

    // Test typical values
    double v1 = 1000000.0; // 1 MHz
    uint32_t raw1 = Q20_12_32::from_double(v1);
    double back1 = Q20_12_32::to_double(raw1);
    ASSERT_NEAR(back1, v1, 1.0);

    // Test fractional precision
    double v2 = 12345.6789;
    uint32_t raw2 = Q20_12_32::from_double(v2);
    double back2 = Q20_12_32::to_double(raw2);
    ASSERT_NEAR(back2, v2, 0.001);

    // Test bounds for 32-bit storage
    ASSERT_DOUBLE_EQ(Q20_12_32::min_value(), 0.0);
    ASSERT_NEAR(Q20_12_32::max_value(), 1048576.0 - 1.0 / 4096.0, 1.0);
}

TEST(FixedPointTest, StorageType_DefaultIs64bit) {
    // Verify default storage type is uint64_t for backward compatibility
    using DefaultStorage = FixedPoint<44, 20, true>;
    static_assert(std::is_same_v<DefaultStorage::storage_type, uint64_t>);
    static_assert(std::is_same_v<DefaultStorage::signed_storage_type, int64_t>);
}

TEST(FixedPointTest, StorageType_SignExtension) {
    // Test sign extension works correctly with 16-bit storage
    using Q8_8_16 = FixedPoint<8, 8, true, uint16_t>;

    // Negative value: -1.5
    double v = -1.5;
    uint16_t raw = Q8_8_16::from_double(v);
    double back = Q8_8_16::to_double(raw);
    ASSERT_DOUBLE_EQ(back, v);

    // Verify raw representation (16-bit two's complement)
    ASSERT_EQ(raw, 0xFE80); // -1.5 in Q8.8

    // Test with minimum value
    double min_val = Q8_8_16::min_value();
    uint16_t min_raw = Q8_8_16::from_double(min_val);
    double min_back = Q8_8_16::to_double(min_raw);
    ASSERT_DOUBLE_EQ(min_back, -128.0);
    ASSERT_EQ(min_raw, 0x8000);
}