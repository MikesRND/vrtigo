#include <chrono>
#include <thread>

#include <gtest/gtest.h>
#include <vrtigo.hpp>

using namespace vrtigo;

// Test fixture for Timestamp tests
class TimestampTest : public ::testing::Test {
protected:
    static constexpr uint32_t test_seconds = 1699000000;             // Example timestamp
    static constexpr uint64_t test_picoseconds = 500'000'000'000ULL; // 500 microseconds
};

// Construction tests
TEST_F(TimestampTest, DefaultConstruction) {
    UtcRealTimestamp ts;
    EXPECT_EQ(ts.tsi(), 0);
    EXPECT_EQ(ts.tsf(), 0);
}

TEST_F(TimestampTest, ComponentConstruction) {
    UtcRealTimestamp ts(test_seconds, test_picoseconds);
    EXPECT_EQ(ts.tsi(), test_seconds);
    EXPECT_EQ(ts.tsf(), test_picoseconds);
}

TEST_F(TimestampTest, NormalizationOnConstruction) {
    // 1.5 seconds worth of picoseconds
    uint64_t excess_picos = 1'500'000'000'000ULL;
    UtcRealTimestamp ts(100, excess_picos);

    EXPECT_EQ(ts.tsi(), 101);                // Should add 1 second
    EXPECT_EQ(ts.tsf(), 500'000'000'000ULL); // Remaining 500 microseconds
}

// Factory method tests
TEST_F(TimestampTest, FromUTCSeconds) {
    auto ts = UtcRealTimestamp::from_utc_seconds(test_seconds);
    EXPECT_EQ(ts.tsi(), test_seconds);
    EXPECT_EQ(ts.tsf(), 0);
}

TEST_F(TimestampTest, FromComponents) {
    auto ts = UtcRealTimestamp(test_seconds, test_picoseconds);
    EXPECT_EQ(ts.tsi(), test_seconds);
    EXPECT_EQ(ts.tsf(), test_picoseconds);
}

TEST_F(TimestampTest, Now) {
    auto before = std::chrono::system_clock::now();
    auto ts = UtcRealTimestamp::now();
    auto after = std::chrono::system_clock::now();

    // Convert bounds to seconds
    auto before_sec =
        std::chrono::duration_cast<std::chrono::seconds>(before.time_since_epoch()).count();
    auto after_sec =
        std::chrono::duration_cast<std::chrono::seconds>(after.time_since_epoch()).count();

    // Timestamp should be between before and after
    EXPECT_GE(ts.tsi(), static_cast<uint32_t>(before_sec));
    EXPECT_LE(ts.tsi(), static_cast<uint32_t>(after_sec));
}

// Chrono conversion tests
TEST_F(TimestampTest, FromChrono) {
    auto sys_time = std::chrono::system_clock::now();
    auto ts = UtcRealTimestamp::from_chrono(sys_time);

    // Convert back and compare (will lose picosecond precision)
    auto converted_back = ts.to_chrono();

    // Should be within 1 microsecond (chrono typically has nanosecond precision)
    auto diff = std::chrono::abs(sys_time - converted_back);
    EXPECT_LT(diff, std::chrono::microseconds(1));
}

TEST_F(TimestampTest, ToChrono) {
    UtcRealTimestamp ts(test_seconds, test_picoseconds);
    auto sys_time = ts.to_chrono();

    auto duration = sys_time.time_since_epoch();
    auto sec = std::chrono::duration_cast<std::chrono::seconds>(duration);
    auto nano = std::chrono::duration_cast<std::chrono::nanoseconds>(duration - sec);

    EXPECT_EQ(sec.count(), test_seconds);
    // 500 microseconds = 500,000 nanoseconds
    EXPECT_EQ(nano.count(), 500'000'000);
}

TEST_F(TimestampTest, ToTimeT) {
    UtcRealTimestamp ts(test_seconds, test_picoseconds);
    std::time_t t = ts.to_time_t();
    EXPECT_EQ(t, test_seconds);
}

// Comparison operator tests
TEST_F(TimestampTest, Equality) {
    UtcRealTimestamp ts1(100, 500);
    UtcRealTimestamp ts2(100, 500);
    UtcRealTimestamp ts3(100, 600);
    UtcRealTimestamp ts4(101, 500);

    EXPECT_EQ(ts1, ts2);
    EXPECT_NE(ts1, ts3);
    EXPECT_NE(ts1, ts4);
}

TEST_F(TimestampTest, LessThan) {
    UtcRealTimestamp ts1(100, 500);
    UtcRealTimestamp ts2(100, 600);
    UtcRealTimestamp ts3(101, 400);

    EXPECT_LT(ts1, ts2); // Same seconds, different picoseconds
    EXPECT_LT(ts1, ts3); // Different seconds
    EXPECT_LT(ts2, ts3);
}

TEST_F(TimestampTest, GreaterThan) {
    UtcRealTimestamp ts1(100, 600);
    UtcRealTimestamp ts2(100, 500);
    UtcRealTimestamp ts3(99, 700);

    EXPECT_GT(ts1, ts2); // Same seconds, different picoseconds
    EXPECT_GT(ts1, ts3); // Different seconds
}

// Arithmetic operation tests
TEST_F(TimestampTest, AdditionWithDuration) {
    UtcRealTimestamp ts(100, 500'000'000'000ULL); // 100 seconds + 500 microseconds

    // Add 1 millisecond (1,000,000 nanoseconds = 1,000,000,000 picoseconds)
    auto result = ts + std::chrono::milliseconds(1);

    EXPECT_EQ(result.tsi(), 100);
    EXPECT_EQ(result.tsf(), 501'000'000'000ULL); // 501 microseconds
}

TEST_F(TimestampTest, AdditionWithOverflow) {
    UtcRealTimestamp ts(100, 999'000'000'000ULL); // 100 seconds + 999 milliseconds

    // Add 2 milliseconds, should overflow to next second
    auto result = ts + std::chrono::milliseconds(2);

    EXPECT_EQ(result.tsi(), 101);
    EXPECT_EQ(result.tsf(), 1'000'000'000ULL); // 1 millisecond
}

TEST_F(TimestampTest, SubtractionWithDuration) {
    UtcRealTimestamp ts(100, 500'000'000'000ULL); // 100 seconds + 500 milliseconds

    // Subtract 100 microseconds (100,000,000 picoseconds)
    auto result = ts - std::chrono::microseconds(100);

    EXPECT_EQ(result.tsi(), 100);
    EXPECT_EQ(result.tsf(), 499'900'000'000ULL); // 499.9 milliseconds
}

TEST_F(TimestampTest, SubtractionWithBorrow) {
    UtcRealTimestamp ts(100,
                        100'000'000ULL); // 100 seconds + 100 microseconds (100,000,000 picoseconds)

    // Subtract 200 microseconds (200,000,000 picoseconds), should borrow from seconds
    auto result = ts - std::chrono::microseconds(200);

    EXPECT_EQ(result.tsi(), 99);
    EXPECT_EQ(result.tsf(), 999'900'000'000ULL); // 999.9 milliseconds
}

TEST_F(TimestampTest, SubtractionUnderflow) {
    UtcRealTimestamp ts(1, 0); // 1 second

    // Subtract 2 seconds, should clamp to zero
    auto result = ts - std::chrono::seconds(2);

    EXPECT_EQ(result.tsi(), 0);
    EXPECT_EQ(result.tsf(), 0);
}

TEST_F(TimestampTest, DifferenceBetweenTimestamps) {
    UtcRealTimestamp ts1(100, 500'000'000'000ULL); // 100 seconds + 0.5 seconds
    UtcRealTimestamp ts2(101, 200'000'000'000ULL); // 101 seconds + 0.2 seconds

    auto diff = ts2 - ts1;

    // Difference should be 0.7 seconds = 700,000,000 nanoseconds
    // (101.2 - 100.5 = 0.7 seconds)
    EXPECT_EQ(diff.count(), 700'000'000); // in nanoseconds
}

// Test large timestamp differences that would have caused overflow in old implementation
TEST_F(TimestampTest, LargeTimestampDifferences) {
    // Test 150 days difference (> 106 days which was the old overflow threshold)
    uint32_t days_150_seconds = 150 * 24 * 3600; // 12,960,000 seconds
    UtcRealTimestamp ts1(1000000000, 100'000'000'000ULL);
    UtcRealTimestamp ts2(1000000000 + days_150_seconds, 200'000'000'000ULL);

    auto diff = ts2 - ts1;

    // Expected: 150 days + 0.1 seconds in nanoseconds
    int64_t expected_nanos =
        static_cast<int64_t>(days_150_seconds) * 1'000'000'000LL + 100'000'000LL;
    EXPECT_EQ(diff.count(), expected_nanos);
}

TEST_F(TimestampTest, YearLongTimestampDifference) {
    // Test 1 year difference
    uint32_t year_seconds = 365 * 24 * 3600; // 31,536,000 seconds
    UtcRealTimestamp ts1(500000000, 0);
    UtcRealTimestamp ts2(500000000 + year_seconds, 0);

    auto diff = ts2 - ts1;

    // Expected: 365 days in nanoseconds
    int64_t expected_nanos = static_cast<int64_t>(year_seconds) * 1'000'000'000LL;
    EXPECT_EQ(diff.count(), expected_nanos);
}

TEST_F(TimestampTest, DecadeLongTimestampDifference) {
    // Test 10 year difference
    uint32_t decade_seconds = 10 * 365 * 24 * 3600;                     // ~315,360,000 seconds
    UtcRealTimestamp ts1(100000000, 999'999'999'999ULL);                // With max picoseconds
    UtcRealTimestamp ts2(100000000 + decade_seconds, 1'000'000'000ULL); // Small picoseconds

    auto diff = ts2 - ts1;

    // Expected: 10 years minus ~999ms in nanoseconds
    int64_t expected_nanos = static_cast<int64_t>(decade_seconds) * 1'000'000'000LL - 998'999'999LL;
    EXPECT_EQ(diff.count(), expected_nanos);
}

TEST_F(TimestampTest, NegativeLargeTimestampDifference) {
    // Test large negative difference (ts1 > ts2)
    uint32_t days_200_seconds = 200 * 24 * 3600; // 17,280,000 seconds
    UtcRealTimestamp ts1(2000000000, 500'000'000'000ULL);
    UtcRealTimestamp ts2(2000000000 - days_200_seconds, 400'000'000'000ULL);

    auto diff = ts2 - ts1;

    // Expected: -200 days - 0.1 seconds in nanoseconds
    int64_t expected_nanos =
        -static_cast<int64_t>(days_200_seconds) * 1'000'000'000LL - 100'000'000LL;
    EXPECT_EQ(diff.count(), expected_nanos);
}

TEST_F(TimestampTest, MaxSafeTimestampDifference) {
    // Test near maximum safe difference (~292 years)
    // Max safe seconds = INT64_MAX / 10^9 ≈ 9.2e9 seconds ≈ 292 years
    // Note: uint32_t max is ~136 years, so test with that limit instead

    UtcRealTimestamp ts1(0, 0);
    UtcRealTimestamp ts2(UINT32_MAX, 0); // Maximum possible seconds (~136 years)

    auto diff = ts2 - ts1;

    // Expected: UINT32_MAX seconds in nanoseconds
    int64_t expected_nanos = static_cast<int64_t>(UINT32_MAX) * 1'000'000'000LL;
    EXPECT_EQ(diff.count(), expected_nanos);
}

// Integration with SignalPacket tests
TEST_F(TimestampTest, PacketIntegration) {
    using PacketType = typed::SignalDataPacketBuilder<vrtigo::NoClassId, UtcRealTimestamp,
                                                      vrtigo::Trailer::none, 256>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};
    PacketType packet(buffer);

    // Set timestamp using unified interface
    UtcRealTimestamp ts(test_seconds, test_picoseconds);
    packet.set_timestamp(ts);

    // Read back
    auto read_ts = packet.timestamp();
    EXPECT_EQ(read_ts.tsi(), test_seconds);
    EXPECT_EQ(read_ts.tsf(), test_picoseconds);
}

TEST_F(TimestampTest, BuilderIntegration) {
    using PacketType = typed::SignalDataPacketBuilder<vrtigo::NoClassId, UtcRealTimestamp,
                                                      vrtigo::Trailer::none, 256>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    UtcRealTimestamp ts(test_seconds, test_picoseconds);

    PacketType packet(buffer);
    packet.set_stream_id(0x12345678);
    packet.set_timestamp(ts);

    auto read_ts = packet.timestamp();
    EXPECT_EQ(read_ts.tsi(), test_seconds);
    EXPECT_EQ(read_ts.tsf(), test_picoseconds);
}

// Test that non-UTC timestamp types are not implemented
TEST_F(TimestampTest, NonUTCTimestampsNotImplemented) {
    // GPS and other timestamp types are not implemented through the typed API
    // This is by design - users must handle these manually based on their
    // specific requirements (different epochs, conversions, etc.)

    // The primary template's static_assert ensures compile-time failure
    // if someone tries to instantiate an unsupported timestamp type:
    // Timestamp<tsi_type::gps, tsf_type::real_time> would fail to compile if instantiated

    // Only UTC+RealTime is fully supported with conversions
    UtcRealTimestamp utc_ts(1234567890, 500'000'000'000ULL);
    EXPECT_EQ(utc_ts.tsi(), 1234567890);
    EXPECT_EQ(utc_ts.tsf(), 500'000'000'000ULL);

    // For packets needing GPS or other timestamp types, use Timestamp<tsi, tsf>
    // as the template parameter to configure packet structure correctly
}

// Test that GPS timestamps can be used to configure packet structure
TEST_F(TimestampTest, GPSTimestampPacketStructure) {
    // GPS timestamps can be used to configure packet structure
    // even though the timestamp type itself is not fully implemented
    using GPSPacket = typed::SignalDataPacketBuilder<
        vrtigo::NoClassId, Timestamp<TsiType::gps, TsfType::real_time>, vrtigo::Trailer::none, 256>;

    // Verify the packet has timestamp support
    static_assert(GPSPacket::has_timestamp());

    alignas(4) std::array<uint8_t, GPSPacket::size_bytes()> buffer{};
    GPSPacket packet(buffer);

    // Use typed timestamp methods for GPS values
    uint32_t gps_seconds = 1234567890; // GPS seconds since Jan 6, 1980
    uint64_t gps_picoseconds = 500'000'000'000ULL;
    using GPSTimestamp = Timestamp<TsiType::gps, TsfType::real_time>;
    GPSTimestamp gps_ts(gps_seconds, gps_picoseconds);
    packet.set_timestamp(gps_ts);

    auto read_ts = packet.timestamp();
    EXPECT_EQ(read_ts.tsi(), gps_seconds);
    EXPECT_EQ(read_ts.tsf(), gps_picoseconds);

    // Verify header bits are correct for GPS
    uint32_t raw_header;
    std::memcpy(&raw_header, buffer.data(), 4);
    raw_header = detail::network_to_host32(raw_header);

    // TSI field (bits 23-22) should be 2 (GPS)
    EXPECT_EQ((raw_header >> 22) & 0x03, 2);
    // TSF field (bits 21-20) should be 2 (real_time)
    EXPECT_EQ((raw_header >> 20) & 0x03, 2);
}

// Negative duration arithmetic tests (testing the signed/unsigned fix)
TEST_F(TimestampTest, AddSmallNegativeDuration) {
    UtcRealTimestamp ts(100, 500'000'000'000ULL); // 100.5 seconds

    // Add -1 nanosecond (should subtract properly, not wrap to huge positive)
    auto result = ts + std::chrono::nanoseconds(-1);

    EXPECT_EQ(result.tsi(), 100);
    EXPECT_EQ(result.tsf(), 499'999'999'000ULL); // 1 nanosecond less
}

TEST_F(TimestampTest, SubtractNegativeDuration) {
    UtcRealTimestamp ts(100, 500'000'000'000ULL); // 100.5 seconds

    // Subtract -1 millisecond (double negative = addition)
    auto result = ts - std::chrono::milliseconds(-1);

    EXPECT_EQ(result.tsi(), 100);
    EXPECT_EQ(result.tsf(), 501'000'000'000ULL); // Added 1 millisecond
}

TEST_F(TimestampTest, AddLargeNegativeDuration) {
    UtcRealTimestamp ts(1000, 200'000'000'000ULL); // 1000.2 seconds

    // Add -10 seconds
    auto result = ts + std::chrono::seconds(-10);

    EXPECT_EQ(result.tsi(), 990);
    EXPECT_EQ(result.tsf(), 200'000'000'000ULL); // Picoseconds unchanged
}

TEST_F(TimestampTest, MixedPositiveNegativeOperations) {
    UtcRealTimestamp ts(500, 0); // 500 seconds exactly

    // Add 5 seconds then subtract 3 seconds
    ts += std::chrono::seconds(5);
    ts -= std::chrono::seconds(3);

    EXPECT_EQ(ts.tsi(), 502); // 500 + 5 - 3 = 502
    EXPECT_EQ(ts.tsf(), 0);

    // Now add negative duration
    ts += std::chrono::seconds(-2);

    EXPECT_EQ(ts.tsi(), 500); // Back to 500
    EXPECT_EQ(ts.tsf(), 0);
}

TEST_F(TimestampTest, NegativeDurationWithBorrow) {
    UtcRealTimestamp ts(100, 100'000'000ULL); // 100 seconds + 100 microseconds

    // Add -500 microseconds (should borrow from seconds)
    auto result = ts + std::chrono::microseconds(-500);

    EXPECT_EQ(result.tsi(), 99);
    EXPECT_EQ(result.tsf(), 999'600'000'000ULL); // Borrowed and subtracted
}

TEST_F(TimestampTest, LargeNegativeDurationNearZero) {
    UtcRealTimestamp ts(5, 0); // 5 seconds

    // Add -10 seconds (should clamp to zero, not underflow)
    auto result = ts + std::chrono::seconds(-10);

    EXPECT_EQ(result.tsi(), 0);
    EXPECT_EQ(result.tsf(), 0);
}

TEST_F(TimestampTest, CompoundAssignmentNegativeDuration) {
    UtcRealTimestamp ts(1000, 0);

    // Use compound assignment with negative duration
    ts += std::chrono::milliseconds(-500);

    EXPECT_EQ(ts.tsi(), 999);
    EXPECT_EQ(ts.tsf(), 500'000'000'000ULL); // 0.5 seconds

    // Another compound assignment
    ts -= std::chrono::milliseconds(-250); // Double negative = addition

    EXPECT_EQ(ts.tsi(), 999);
    EXPECT_EQ(ts.tsf(), 750'000'000'000ULL); // 0.75 seconds
}

TEST_F(TimestampTest, VerySmallNegativeDurations) {
    UtcRealTimestamp ts(100, 1000); // 100 seconds + 1000 picoseconds

    // Subtract 1 nanosecond (1000 picoseconds)
    ts -= std::chrono::nanoseconds(1);

    EXPECT_EQ(ts.tsi(), 100);
    EXPECT_EQ(ts.tsf(), 0); // Exactly zero picoseconds

    // Subtract another nanosecond (should borrow)
    ts -= std::chrono::nanoseconds(1);

    EXPECT_EQ(ts.tsi(), 99);
    EXPECT_EQ(ts.tsf(), 999'999'999'000ULL); // Borrowed from seconds
}

// Precision tests
TEST_F(TimestampTest, PicosecondPrecision) {
    // Test that we maintain picosecond precision
    UtcRealTimestamp ts(100, 123'456'789'012ULL); // Exact picoseconds

    EXPECT_EQ(ts.tsf(), 123'456'789'012ULL);

    // When converting to chrono and back, we lose sub-nanosecond precision
    auto sys_time = ts.to_chrono();
    auto ts2 = UtcRealTimestamp::from_chrono(sys_time);

    // Should preserve nanosecond precision (123,456,789 nanoseconds)
    EXPECT_EQ(ts2.tsi(), 100);
    EXPECT_EQ(ts2.tsf(), 123'456'789'000ULL); // Lost the 12 picoseconds
}

// Epoch boundary tests (testing the from_chrono bounds checking)
TEST_F(TimestampTest, PreEpochTimeClampedToZero) {
    // Create a time point before Unix epoch (1970-01-01)
    auto pre_epoch = std::chrono::system_clock::from_time_t(-1); // 1969-12-31 23:59:59

    auto ts = UtcRealTimestamp::from_chrono(pre_epoch);

    // Should clamp to zero (beginning of epoch)
    EXPECT_EQ(ts.tsi(), 0);
    EXPECT_EQ(ts.tsf(), 0);
}

TEST_F(TimestampTest, FarPreEpochTimeClampedToZero) {
    // Create a time point far before Unix epoch
    auto far_pre_epoch = std::chrono::system_clock::from_time_t(-31536000); // 1969-01-01

    auto ts = UtcRealTimestamp::from_chrono(far_pre_epoch);

    // Should clamp to zero
    EXPECT_EQ(ts.tsi(), 0);
    EXPECT_EQ(ts.tsf(), 0);
}

TEST_F(TimestampTest, PostMaxTimeClampedToMax) {
    // Create a time point after max uint32_t seconds (year ~2106)
    // UINT32_MAX seconds = 4294967295 = Feb 7, 2106 06:28:15 UTC
    auto max_time = std::chrono::system_clock::time_point(
        std::chrono::seconds(static_cast<int64_t>(UINT32_MAX) + 1));

    auto ts = UtcRealTimestamp::from_chrono(max_time);

    // Should clamp to max values
    EXPECT_EQ(ts.tsi(), UINT32_MAX);
    EXPECT_EQ(ts.tsf(), 999'999'999'999ULL);
}

TEST_F(TimestampTest, FarFutureTimeClampedToMax) {
    // Create a time point far in the future (year ~2200)
    auto far_future = std::chrono::system_clock::time_point(
        std::chrono::seconds(static_cast<int64_t>(UINT32_MAX) * 2));

    auto ts = UtcRealTimestamp::from_chrono(far_future);

    // Should clamp to max values
    EXPECT_EQ(ts.tsi(), UINT32_MAX);
    EXPECT_EQ(ts.tsf(), 999'999'999'999ULL);
}

TEST_F(TimestampTest, ExactMaxTime) {
    // Test exactly at UINT32_MAX seconds
    auto exact_max = std::chrono::system_clock::time_point(
        std::chrono::seconds(static_cast<int64_t>(UINT32_MAX)));

    auto ts = UtcRealTimestamp::from_chrono(exact_max);

    // Should be exactly max seconds with zero picoseconds
    EXPECT_EQ(ts.tsi(), UINT32_MAX);
    EXPECT_EQ(ts.tsf(), 0);
}

TEST_F(TimestampTest, NearEpochBoundary) {
    // Test time just after epoch start
    auto just_after_epoch = std::chrono::system_clock::from_time_t(1); // 1 second after epoch

    auto ts = UtcRealTimestamp::from_chrono(just_after_epoch);

    EXPECT_EQ(ts.tsi(), 1);
    EXPECT_EQ(ts.tsf(), 0);
}

TEST_F(TimestampTest, NormalRangeTime) {
    // Test a normal time in the valid range (year 2023)
    auto normal_time = std::chrono::system_clock::from_time_t(1672531200); // Jan 1, 2023

    auto ts = UtcRealTimestamp::from_chrono(normal_time);

    EXPECT_EQ(ts.tsi(), 1672531200);
    EXPECT_EQ(ts.tsf(), 0);
}

// Edge case tests
TEST_F(TimestampTest, MaxValues) {
    // Test with maximum uint32_t seconds (year ~2106)
    uint32_t max_seconds = UINT32_MAX;
    uint64_t max_valid_picos = 999'999'999'999ULL;

    UtcRealTimestamp ts(max_seconds, max_valid_picos);
    EXPECT_EQ(ts.tsi(), max_seconds);
    EXPECT_EQ(ts.tsf(), max_valid_picos);
}

TEST_F(TimestampTest, MultipleNormalizations) {
    // Test with multiple seconds worth of picoseconds
    uint64_t three_seconds_picos = 3'500'000'000'000ULL;

    UtcRealTimestamp ts(100, three_seconds_picos);
    EXPECT_EQ(ts.tsi(), 103);
    EXPECT_EQ(ts.tsf(), 500'000'000'000ULL);
}

// Overflow protection tests
TEST_F(TimestampTest, NormalizeOverflowProtection) {
    // Test that normalize() clamps when adding extra seconds would overflow
    uint64_t two_seconds_picos = 2'000'000'000'000ULL;

    // Timestamp near UINT32_MAX
    UtcRealTimestamp ts(UINT32_MAX - 1, two_seconds_picos);

    // Should clamp to max values instead of wrapping
    EXPECT_EQ(ts.tsi(), UINT32_MAX);
    EXPECT_EQ(ts.tsf(), 999'999'999'999ULL);
}

TEST_F(TimestampTest, NormalizeExtremeOverflow) {
    // Test with extreme picoseconds value that would add many seconds
    uint64_t huge_picos = 10'000'000'000'000ULL; // 10 seconds worth

    UtcRealTimestamp ts(UINT32_MAX - 2, huge_picos);

    // Should clamp to max values
    EXPECT_EQ(ts.tsi(), UINT32_MAX);
    EXPECT_EQ(ts.tsf(), 999'999'999'999ULL);
}

TEST_F(TimestampTest, ArithmeticWithNearMaxTimestamp) {
    // Test that arithmetic operations handle near-max timestamps safely
    UtcRealTimestamp ts(UINT32_MAX - 1, 500'000'000'000ULL);

    // Add 2 seconds - should clamp to max
    ts += std::chrono::seconds(2);

    EXPECT_EQ(ts.tsi(), UINT32_MAX);
    EXPECT_EQ(ts.tsf(), 999'999'999'999ULL);
}

TEST_F(TimestampTest, ArithmeticCausesNormalizationOverflow) {
    // Test arithmetic that causes normalization to trigger overflow protection
    UtcRealTimestamp ts(UINT32_MAX, 100'000'000'000ULL); // Already at max seconds

    // Add 1.5 seconds worth of nanoseconds - would cause overflow
    ts += std::chrono::nanoseconds(1'500'000'000);

    // Should clamp to max values
    EXPECT_EQ(ts.tsi(), UINT32_MAX);
    EXPECT_EQ(ts.tsf(), 999'999'999'999ULL);
}

// Tests for large duration arithmetic (> 106 days)
TEST_F(TimestampTest, AddVeryLargeDuration) {
    // Test adding a duration > 106 days
    UtcRealTimestamp ts(1000, 0);

    // Add 150 days worth of nanoseconds
    auto large_duration = std::chrono::nanoseconds(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::hours(150 * 24)).count());
    ts += large_duration;

    // Should have added 150 days = 12,960,000 seconds
    EXPECT_EQ(ts.tsi(), 1000 + 12'960'000);
    EXPECT_EQ(ts.tsf(), 0);
}

TEST_F(TimestampTest, SubtractVeryLargeDuration) {
    // Test subtracting a duration > 106 days
    UtcRealTimestamp ts(20'000'000, 0); // Start with a large timestamp

    // Subtract 150 days
    auto large_duration = std::chrono::nanoseconds(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::hours(150 * 24)).count());
    ts -= large_duration;

    // Should have subtracted 12,960,000 seconds
    EXPECT_EQ(ts.tsi(), 20'000'000 - 12'960'000);
    EXPECT_EQ(ts.tsf(), 0);
}

TEST_F(TimestampTest, SubtractNanosecondMin) {
    // Test subtracting nanoseconds::min() (special case for UB avoidance)
    UtcRealTimestamp ts(1'000'000'000, 0); // Start with a billion seconds

    // Subtract nanoseconds::min() (~-292 years)
    ts -= std::chrono::nanoseconds::min();

    // nanoseconds::min() is approximately -9,223,372,036,854,775,808 nanoseconds
    // Which is about +9.22e18 nanoseconds when negated (adding)
    // That's about 9,223,372,036 seconds (about 292 years)

    // We expect overflow clamping since this would exceed UINT32_MAX
    EXPECT_EQ(ts.tsi(), UINT32_MAX);
    EXPECT_EQ(ts.tsf(), 999'999'999'999ULL);
}

TEST_F(TimestampTest, AddNanosecondMax) {
    // Test adding nanoseconds::max() (largest positive duration)
    UtcRealTimestamp ts(1000, 0);

    // Add nanoseconds::max() (~292 years)
    ts += std::chrono::nanoseconds::max();

    // Should clamp to max values since this exceeds UINT32_MAX seconds
    EXPECT_EQ(ts.tsi(), UINT32_MAX);
    EXPECT_EQ(ts.tsf(), 999'999'999'999ULL);
}

TEST_F(TimestampTest, MultiMonthSpanArithmetic) {
    // Test arithmetic with multi-month spans (user-visible use case)
    UtcRealTimestamp ts1 = UtcRealTimestamp::from_utc_seconds(1700000000); // Nov 2023
    UtcRealTimestamp ts2 = UtcRealTimestamp::from_utc_seconds(1710000000); // Mar 2024

    // Calculate difference (about 115 days)
    auto diff = ts2 - ts1;

    // Should handle the large difference correctly
    auto days = std::chrono::duration_cast<std::chrono::hours>(diff).count() / 24;
    EXPECT_GT(days, 100); // Should be > 100 days
    EXPECT_LT(days, 120); // Should be < 120 days

    // Add the difference back to ts1
    auto ts3 = ts1 + diff;
    EXPECT_EQ(ts3.tsi(), ts2.tsi());
    EXPECT_EQ(ts3.tsf(), ts2.tsf());
}

TEST_F(TimestampTest, YearSpanArithmetic) {
    // Test arithmetic spanning a full year
    UtcRealTimestamp ts(1700000000, 0); // Nov 2023

    // Add 365 days
    auto year_duration = std::chrono::hours(365 * 24);
    ts += year_duration;

    // Should have added 31,536,000 seconds (365 days)
    EXPECT_EQ(ts.tsi(), 1700000000 + 31'536'000);
    EXPECT_EQ(ts.tsf(), 0);
}

// ==============================================================================
// Non-UTC Timestamp Instantiation Tests
// ==============================================================================

TEST_F(TimestampTest, GPSTimestampConstruction) {
    using GPSTimestamp = Timestamp<TsiType::gps, TsfType::real_time>;
    GPSTimestamp ts(1234567890, 500'000'000'000ULL);

    EXPECT_EQ(ts.tsi(), 1234567890);
    EXPECT_EQ(ts.tsf(), 500'000'000'000ULL);
    EXPECT_EQ(ts.tsi_kind(), TsiType::gps);
    EXPECT_EQ(ts.tsf_kind(), TsfType::real_time);
}

TEST_F(TimestampTest, TAITimestampConstruction) {
    using TAITimestamp = Timestamp<TsiType::other, TsfType::real_time>;
    TAITimestamp ts(1234567890, 500'000'000'000ULL);

    EXPECT_EQ(ts.tsi(), 1234567890);
    EXPECT_EQ(ts.tsf(), 500'000'000'000ULL);
    EXPECT_EQ(ts.tsi_kind(), TsiType::other);
    EXPECT_EQ(ts.tsf_kind(), TsfType::real_time);
}

TEST_F(TimestampTest, SampleCountTimestampConstruction) {
    using SampleCountTimestamp = Timestamp<TsiType::none, TsfType::sample_count>;
    SampleCountTimestamp ts(1000, 123456789);

    EXPECT_EQ(ts.tsi(), 1000);
    EXPECT_EQ(ts.tsf(), 123456789);
    EXPECT_EQ(ts.tsi_kind(), TsiType::none);
    EXPECT_EQ(ts.tsf_kind(), TsfType::sample_count);
}

TEST_F(TimestampTest, GPSTimestampComparison) {
    using GPSTimestamp = Timestamp<TsiType::gps, TsfType::real_time>;
    GPSTimestamp ts1(100, 500);
    GPSTimestamp ts2(100, 500);
    GPSTimestamp ts3(100, 600);
    GPSTimestamp ts4(101, 500);

    EXPECT_EQ(ts1, ts2);
    EXPECT_NE(ts1, ts3);
    EXPECT_LT(ts1, ts3);
    EXPECT_GT(ts4, ts1);
}

TEST_F(TimestampTest, NonUTCFromComponents) {
    using GPSTimestamp = Timestamp<TsiType::gps, TsfType::real_time>;
    auto gps = GPSTimestamp(12345, 678'000'000'000ULL);

    EXPECT_EQ(gps.tsi(), 12345);
    EXPECT_EQ(gps.tsf(), 678'000'000'000ULL);
}

// ==============================================================================
// Default Construction Tests for Non-UTC Types
// ==============================================================================

TEST_F(TimestampTest, GPSDefaultConstruction) {
    using GPSTimestamp = Timestamp<TsiType::gps, TsfType::real_time>;
    GPSTimestamp ts;

    EXPECT_EQ(ts.tsi(), 0);
    EXPECT_EQ(ts.tsf(), 0);
}

TEST_F(TimestampTest, TAIDefaultConstruction) {
    using TAITimestamp = Timestamp<TsiType::other, TsfType::real_time>;
    TAITimestamp ts;

    EXPECT_EQ(ts.tsi(), 0);
    EXPECT_EQ(ts.tsf(), 0);
}

TEST_F(TimestampTest, SampleCountDefaultConstruction) {
    using SampleCountTimestamp = Timestamp<TsiType::none, TsfType::sample_count>;
    SampleCountTimestamp ts;

    EXPECT_EQ(ts.tsi(), 0);
    EXPECT_EQ(ts.tsf(), 0);
}

// ==============================================================================
// Multi-Second Borrow Tests
// ==============================================================================

TEST_F(TimestampTest, MultiSecondBorrow) {
    UtcRealTimestamp ts(10, 100'000'000'000ULL); // 10s + 0.1s

    // Subtract 2.5 seconds worth of picoseconds
    ts -= std::chrono::nanoseconds(2'500'000'000); // 2.5 seconds

    EXPECT_EQ(ts.tsi(), 7);
    EXPECT_EQ(ts.tsf(), 600'000'000'000ULL); // 0.6 seconds (10.1 - 2.5 = 7.6)
}

TEST_F(TimestampTest, MultiSecondBorrowUnderflow) {
    UtcRealTimestamp ts(3, 200'000'000'000ULL);    // 3s + 0.2s
    ts -= std::chrono::nanoseconds(5'000'000'000); // 5 seconds

    EXPECT_EQ(ts.tsi(), 0); // Clamped to zero
    EXPECT_EQ(ts.tsf(), 0);
}

TEST_F(TimestampTest, ThreeSecondBorrow) {
    UtcRealTimestamp ts(5, 900'000'000'000ULL); // 5.9 seconds

    // Subtract 3.2 seconds requiring 3-second borrow
    ts -= std::chrono::nanoseconds(3'200'000'000);

    EXPECT_EQ(ts.tsi(), 2);
    EXPECT_EQ(ts.tsf(), 700'000'000'000ULL); // 0.7 seconds
}

// ==============================================================================
// Overflow Sentinel Tests
// ==============================================================================

TEST_F(TimestampTest, ArithmeticOverflowUsesSentinel) {
    UtcRealTimestamp ts(UINT32_MAX, 500'000'000'000ULL);
    ts += std::chrono::seconds(1);

    EXPECT_EQ(ts.tsi(), UINT32_MAX);
    EXPECT_EQ(ts.tsf(), UtcRealTimestamp::MAX_FRACTIONAL);
    EXPECT_EQ(ts.tsf(), 999'999'999'999ULL);
}

TEST_F(TimestampTest, NormalizationOverflowUsesSentinel) {
    // Constructor normalization should trigger overflow
    uint64_t huge_picos = 10'000'000'000'000ULL; // 10 seconds worth
    UtcRealTimestamp ts(UINT32_MAX - 2, huge_picos);

    EXPECT_EQ(ts.tsi(), UINT32_MAX);
    EXPECT_EQ(ts.tsf(), UtcRealTimestamp::MAX_FRACTIONAL);
}

TEST_F(TimestampTest, FromChronoOverflowUsesSentinel) {
    // Create time after year 2106 (> UINT32_MAX seconds)
    auto far_future = std::chrono::system_clock::time_point(
        std::chrono::seconds(static_cast<int64_t>(UINT32_MAX) + 1000));

    auto ts = UtcRealTimestamp::from_chrono(far_future);

    EXPECT_EQ(ts.tsi(), UINT32_MAX);
    EXPECT_EQ(ts.tsf(), UtcRealTimestamp::MAX_FRACTIONAL);
}

// ==============================================================================
// TSF-Aware Normalization Tests
// ==============================================================================

TEST_F(TimestampTest, SampleCountDoesNotNormalize) {
    using SampleCountTimestamp = Timestamp<TsiType::none, TsfType::sample_count>;

    // Sample count should NOT normalize even with huge fractional values
    uint64_t huge_sample_count = 5'000'000'000'000ULL; // Far exceeds 1 second worth
    SampleCountTimestamp ts(100, huge_sample_count);

    // Should NOT normalize - values stay as-is
    EXPECT_EQ(ts.tsi(), 100);
    EXPECT_EQ(ts.tsf(), huge_sample_count); // Unchanged!
}

TEST_F(TimestampTest, GPSRealTimeNormalizes) {
    using GPSTimestamp = Timestamp<TsiType::gps, TsfType::real_time>;

    // GPS with real_time TSF should normalize just like UTC
    uint64_t excess_picos = 2'500'000'000'000ULL; // 2.5 seconds
    GPSTimestamp ts(100, excess_picos);

    EXPECT_EQ(ts.tsi(), 102);                // Should add 2 seconds
    EXPECT_EQ(ts.tsf(), 500'000'000'000ULL); // Remaining 0.5 seconds
}

TEST_F(TimestampTest, FreeRunningDoesNotNormalize) {
    using FreeRunningTimestamp = Timestamp<TsiType::none, TsfType::free_running>;

    uint64_t large_fractional = 10'000'000'000'000ULL;
    FreeRunningTimestamp ts(50, large_fractional);

    // Should NOT normalize
    EXPECT_EQ(ts.tsi(), 50);
    EXPECT_EQ(ts.tsf(), large_fractional);
}

TEST_F(TimestampTest, TSFNoneDoesNotNormalize) {
    using NoTSFTimestamp = Timestamp<TsiType::utc, TsfType::none>;

    // Even with UTC TSI, TsfType::none should not normalize
    uint64_t arbitrary_value = 3'000'000'000'000ULL;
    NoTSFTimestamp ts(200, arbitrary_value);

    EXPECT_EQ(ts.tsi(), 200);
    EXPECT_EQ(ts.tsf(), arbitrary_value); // No normalization
}

// ==============================================================================
// Edge Case Tests
// ==============================================================================

TEST_F(TimestampTest, MaxFractionalConstantAccessible) {
    // Verify the constant is properly defined and accessible
    EXPECT_EQ(UtcRealTimestamp::MAX_FRACTIONAL, 999'999'999'999ULL);
    EXPECT_EQ(UtcRealTimestamp::MAX_FRACTIONAL, UtcRealTimestamp::PICOSECONDS_PER_SECOND - 1);
}

TEST_F(TimestampTest, NonUTCTypeTraits) {
    using GPSTimestamp = Timestamp<TsiType::gps, TsfType::real_time>;
    using TAITimestamp = Timestamp<TsiType::other, TsfType::real_time>;

    GPSTimestamp gps;
    TAITimestamp tai;

    EXPECT_EQ(gps.tsi_kind(), TsiType::gps);
    EXPECT_EQ(gps.tsf_kind(), TsfType::real_time);
    EXPECT_EQ(tai.tsi_kind(), TsiType::other);
    EXPECT_EQ(tai.tsf_kind(), TsfType::real_time);
}

// ==============================================================================
// Set Method Tests
// ==============================================================================

TEST_F(TimestampTest, SetBasicComponents) {
    UtcRealTimestamp ts;
    EXPECT_EQ(ts.tsi(), 0u);
    EXPECT_EQ(ts.tsf(), 0u);

    ts.set(1000u, 500'000'000'000ULL);
    EXPECT_EQ(ts.tsi(), 1000u);
    EXPECT_EQ(ts.tsf(), 500'000'000'000ULL);

    ts.set(2000u, 750'000'000'000ULL);
    EXPECT_EQ(ts.tsi(), 2000u);
    EXPECT_EQ(ts.tsf(), 750'000'000'000ULL);
}

TEST_F(TimestampTest, SetWithNormalization) {
    UtcRealTimestamp ts;

    // Set with fractional >= 1 second - should normalize
    ts.set(100u, 2'500'000'000'000ULL); // 2.5 extra seconds
    EXPECT_EQ(ts.tsi(), 102u);
    EXPECT_EQ(ts.tsf(), 500'000'000'000ULL);

    // Set with large fractional - should normalize multiple seconds
    ts.set(50u, 5'000'000'000'000ULL); // 5 extra seconds
    EXPECT_EQ(ts.tsi(), 55u);
    EXPECT_EQ(ts.tsf(), 0u);
}

TEST_F(TimestampTest, SetWithOverflowProtection) {
    UtcRealTimestamp ts;

    // Set seconds near max with excess fractional - should clamp
    ts.set(UINT32_MAX - 1, 3'000'000'000'000ULL); // Would overflow by 1 second
    EXPECT_EQ(ts.tsi(), UINT32_MAX);
    EXPECT_EQ(ts.tsf(), UtcRealTimestamp::MAX_FRACTIONAL);

    // Set at max with excess fractional - should clamp
    ts.set(UINT32_MAX, 1'000'000'000'000ULL);
    EXPECT_EQ(ts.tsi(), UINT32_MAX);
    EXPECT_EQ(ts.tsf(), UtcRealTimestamp::MAX_FRACTIONAL);
}

TEST_F(TimestampTest, SetWithSampleCountNoNormalization) {
    using SampleCountTimestamp = Timestamp<TsiType::utc, TsfType::sample_count>;
    SampleCountTimestamp ts;

    // For sample_count, fractional is a counter, not normalized
    ts.set(100u, 2'000'000'000'000ULL);
    EXPECT_EQ(ts.tsi(), 100u);
    EXPECT_EQ(ts.tsf(), 2'000'000'000'000ULL); // Should NOT normalize

    ts.set(50u, 5'000'000'000'000ULL);
    EXPECT_EQ(ts.tsi(), 50u);
    EXPECT_EQ(ts.tsf(), 5'000'000'000'000ULL); // Should NOT normalize
}

TEST_F(TimestampTest, SetWithGPSTimestamp) {
    using GPSTimestamp = Timestamp<TsiType::gps, TsfType::real_time>;
    GPSTimestamp ts;

    // GPS with real_time should normalize like UTC
    ts.set(1000u, 1'500'000'000'000ULL);
    EXPECT_EQ(ts.tsi(), 1001u);
    EXPECT_EQ(ts.tsf(), 500'000'000'000ULL);

    EXPECT_EQ(ts.tsi_kind(), TsiType::gps);
    EXPECT_EQ(ts.tsf_kind(), TsfType::real_time);
}

TEST_F(TimestampTest, SetWithFreeRunningNoNormalization) {
    using FreeRunningTimestamp = Timestamp<TsiType::utc, TsfType::free_running>;
    FreeRunningTimestamp ts;

    // free_running should NOT normalize
    ts.set(100u, 10'000'000'000'000ULL);
    EXPECT_EQ(ts.tsi(), 100u);
    EXPECT_EQ(ts.tsf(), 10'000'000'000'000ULL); // Should NOT normalize
}

TEST_F(TimestampTest, SetZeroValues) {
    UtcRealTimestamp ts(1000u, 500'000'000'000ULL);
    EXPECT_NE(ts.tsi(), 0u);
    EXPECT_NE(ts.tsf(), 0u);

    ts.set(0u, 0u);
    EXPECT_EQ(ts.tsi(), 0u);
    EXPECT_EQ(ts.tsf(), 0u);
}

TEST_F(TimestampTest, SetMaxValues) {
    UtcRealTimestamp ts;

    ts.set(UINT32_MAX, UtcRealTimestamp::MAX_FRACTIONAL);
    EXPECT_EQ(ts.tsi(), UINT32_MAX);
    EXPECT_EQ(ts.tsf(), UtcRealTimestamp::MAX_FRACTIONAL);
}