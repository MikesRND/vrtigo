#pragma once

#include "vrtigo/detail/time_math.hpp"

#include <limits>
#include <optional>

#include <cmath>
#include <cstdint>

namespace vrtigo {

// Forward declarations
class SamplePeriod;
class Duration;
class ShortDuration;

/**
 * Signed time interval with exact picosecond precision.
 *
 * ## Storage
 * 12-byte split representation: int32_t seconds + uint64_t picoseconds.
 * This matches VITA 49 wire format and enables ±68 year range.
 *
 * ## Range
 * Approximately ±68 years (INT32_MIN to INT32_MAX seconds).
 * This is a 234× improvement over the previous ±106 day range.
 *
 * ## Negative Value Representation (Floor Semantics)
 * Negative durations use floor representation with always-positive picoseconds:
 * - `-1.5 seconds` = `{seconds: -2, picoseconds: 500,000,000,000}`
 * - `-0.5 seconds` = `{seconds: -1, picoseconds: 500,000,000,000}`
 *
 * This ensures picoseconds() always returns a value in [0, 10^12).
 *
 * ## Overflow Policy
 * All arithmetic saturates to min()/max() on overflow:
 * - No exceptions are thrown
 * - No `expected<>` return types
 * - Use `saturated(duration)` helper to detect overflow when needed
 *
 * **Rationale:** With ±68 year range, overflow indicates programmer error
 * (not a recoverable condition). The complexity of dual code paths
 * (checked + unchecked) is not justified for truly exceptional conditions.
 *
 * ## Breaking Changes from Previous API
 * - `picoseconds()` now returns subsecond part only [0, 10^12), not total
 * - Use `total_picoseconds()` for the old behavior (saturates if > 106 days)
 * - Size increased from 8 to 12 bytes (ABI break)
 * - Removed: DurationError, Result, *_checked() methods
 *
 * This is a core library type: noexcept, no allocation.
 */
class Duration {
public:
    static constexpr uint64_t PICOSECONDS_PER_SECOND = detail::PICOS_PER_SEC;
    static constexpr uint64_t PICOSECONDS_PER_MILLISECOND = 1'000'000'000ULL;
    static constexpr uint64_t PICOSECONDS_PER_MICROSECOND = 1'000'000ULL;
    static constexpr uint64_t PICOSECONDS_PER_NANOSECOND = 1'000ULL;

    /// Maximum valid picoseconds value (one less than a full second)
    static constexpr uint64_t MAX_PICOSECONDS = detail::MAX_PICOS;

    // Named constants
    static constexpr Duration min() noexcept {
        return Duration(std::numeric_limits<int32_t>::min(), 0);
    }

    static constexpr Duration max() noexcept {
        return Duration(std::numeric_limits<int32_t>::max(), MAX_PICOSECONDS);
    }

    static constexpr Duration zero() noexcept { return Duration(0, 0); }

    // Default construction - zero duration
    constexpr Duration() noexcept = default;

    // Direct factories - saturate on overflow (consistent with arithmetic)
    static constexpr Duration from_picoseconds(int64_t ps) noexcept {
        auto [sec, picos] = detail::normalize(0, ps);
        return Duration(detail::clamp_to_duration(sec, picos), picos);
    }

    static constexpr Duration from_nanoseconds(int64_t ns) noexcept {
        // Check for overflow before multiplying (max safe: ~9.2e15 ns = ~106 days)
        constexpr int64_t MAX_SAFE_NS =
            std::numeric_limits<int64_t>::max() / static_cast<int64_t>(PICOSECONDS_PER_NANOSECOND);
        constexpr int64_t MIN_SAFE_NS =
            std::numeric_limits<int64_t>::min() / static_cast<int64_t>(PICOSECONDS_PER_NANOSECOND);
        if (ns > MAX_SAFE_NS)
            return max();
        if (ns < MIN_SAFE_NS)
            return min();
        return from_picoseconds(ns * static_cast<int64_t>(PICOSECONDS_PER_NANOSECOND));
    }

    static constexpr Duration from_microseconds(int64_t us) noexcept {
        constexpr int64_t MAX_SAFE_US =
            std::numeric_limits<int64_t>::max() / static_cast<int64_t>(PICOSECONDS_PER_MICROSECOND);
        constexpr int64_t MIN_SAFE_US =
            std::numeric_limits<int64_t>::min() / static_cast<int64_t>(PICOSECONDS_PER_MICROSECOND);
        if (us > MAX_SAFE_US)
            return max();
        if (us < MIN_SAFE_US)
            return min();
        return from_picoseconds(us * static_cast<int64_t>(PICOSECONDS_PER_MICROSECOND));
    }

    static constexpr Duration from_milliseconds(int64_t ms) noexcept {
        constexpr int64_t MAX_SAFE_MS =
            std::numeric_limits<int64_t>::max() / static_cast<int64_t>(PICOSECONDS_PER_MILLISECOND);
        constexpr int64_t MIN_SAFE_MS =
            std::numeric_limits<int64_t>::min() / static_cast<int64_t>(PICOSECONDS_PER_MILLISECOND);
        if (ms > MAX_SAFE_MS)
            return max();
        if (ms < MIN_SAFE_MS)
            return min();
        return from_picoseconds(ms * static_cast<int64_t>(PICOSECONDS_PER_MILLISECOND));
    }

    static constexpr Duration from_seconds(int64_t s) noexcept {
        uint64_t picos = 0;
        int32_t sec = detail::clamp_to_duration(s, picos);
        return Duration(sec, picos);
    }

    // Checked factory from double - returns nullopt on overflow or invalid input
    static std::optional<Duration> from_seconds(double s) noexcept {
        if (!std::isfinite(s)) {
            return std::nullopt;
        }

        // Split into seconds and fractional first
        double int_part;
        double frac_part = std::modf(s, &int_part);

        // Check that int_part fits in int32_t (prevents UB on cast)
        constexpr double max_sec = static_cast<double>(std::numeric_limits<int32_t>::max());
        constexpr double min_sec = static_cast<double>(std::numeric_limits<int32_t>::min());
        if (int_part > max_sec || int_part < min_sec) {
            return std::nullopt;
        }

        int32_t sec = static_cast<int32_t>(int_part);
        uint64_t picos;

        if (frac_part >= 0) {
            picos = static_cast<uint64_t>(frac_part * static_cast<double>(PICOSECONDS_PER_SECOND) +
                                          0.5);
        } else {
            // Negative fractional: adjust for floor semantics
            if (sec == std::numeric_limits<int32_t>::min()) {
                return std::nullopt; // sec-- would underflow
            }
            sec--;
            frac_part += 1.0;
            picos = static_cast<uint64_t>(frac_part * static_cast<double>(PICOSECONDS_PER_SECOND) +
                                          0.5);
        }

        // Normalize if picos >= PICOS_PER_SEC due to rounding
        if (picos >= PICOSECONDS_PER_SECOND) {
            if (sec == std::numeric_limits<int32_t>::max()) {
                return std::nullopt; // sec++ would overflow
            }
            picos -= PICOSECONDS_PER_SECOND;
            sec++;
        }

        return Duration(sec, picos);
    }

    // Checked factory from samples - returns nullopt on overflow
    static std::optional<Duration> from_samples(int64_t count, SamplePeriod period) noexcept;

    // Safe conversion to ShortDuration - returns nullopt if range exceeded (~106 days)
    std::optional<ShortDuration> to_short_duration() const noexcept;

    // Primary accessors - direct access to components
    constexpr int32_t seconds() const noexcept { return seconds_; }
    constexpr uint64_t picoseconds() const noexcept { return picoseconds_; }

    // Convenience accessor - total picoseconds (SATURATES if > ~106 days)
    // Use seconds() + picoseconds() for full ±68 year range
    constexpr int64_t total_picoseconds() const noexcept {
        // Check if total would overflow int64_t
        constexpr int64_t max_safe_sec =
            std::numeric_limits<int64_t>::max() / static_cast<int64_t>(PICOSECONDS_PER_SECOND);
        constexpr int64_t min_safe_sec =
            std::numeric_limits<int64_t>::min() / static_cast<int64_t>(PICOSECONDS_PER_SECOND);

        if (seconds_ > max_safe_sec) {
            return std::numeric_limits<int64_t>::max();
        }
        if (seconds_ < min_safe_sec) {
            return std::numeric_limits<int64_t>::min();
        }

        return static_cast<int64_t>(seconds_) * static_cast<int64_t>(PICOSECONDS_PER_SECOND) +
               static_cast<int64_t>(picoseconds_);
    }

    // Full precision conversion to double (always works, no saturation)
    constexpr double to_seconds() const noexcept {
        return static_cast<double>(seconds_) +
               static_cast<double>(picoseconds_) / static_cast<double>(PICOSECONDS_PER_SECOND);
    }

    // Predicates
    constexpr bool is_zero() const noexcept { return seconds_ == 0 && picoseconds_ == 0; }
    constexpr bool is_negative() const noexcept { return seconds_ < 0; }
    constexpr bool is_positive() const noexcept {
        return seconds_ > 0 || (seconds_ == 0 && picoseconds_ > 0);
    }

    // Absolute value (saturates for min())
    constexpr Duration abs() const noexcept {
        if (*this == min()) {
            return max(); // Saturate: abs(MIN) would overflow
        }
        if (is_negative()) {
            return -(*this);
        }
        return *this;
    }

    // Unary negation (saturates for min())
    constexpr Duration operator-() const noexcept {
        if (*this == min()) {
            return max(); // Saturate: -MIN would overflow
        }
        if (is_zero()) {
            return zero();
        }

        // Negate: flip sign and adjust for floor semantics
        // If we have {sec, picos} representing (sec + picos/10^12),
        // negation gives {-sec-1, 10^12 - picos} when picos > 0
        if (picoseconds_ == 0) {
            return Duration(-seconds_, 0);
        } else {
            return Duration(-seconds_ - 1, PICOSECONDS_PER_SECOND - picoseconds_);
        }
    }

    // Arithmetic operators (saturate on overflow)
    constexpr Duration& operator+=(Duration other) noexcept {
        auto [sec, picos] =
            detail::add_time(seconds_, picoseconds_, other.seconds_, other.picoseconds_);
        seconds_ = detail::clamp_to_duration(sec, picos);
        picoseconds_ = picos;
        return *this;
    }

    constexpr Duration& operator-=(Duration other) noexcept {
        auto [sec, picos] =
            detail::sub_time(seconds_, picoseconds_, other.seconds_, other.picoseconds_);
        seconds_ = detail::clamp_to_duration(sec, picos);
        picoseconds_ = picos;
        return *this;
    }

    constexpr Duration& operator*=(int64_t scalar) noexcept {
        auto [sec, picos] = detail::mul_time(seconds_, picoseconds_, scalar);
        seconds_ = detail::clamp_to_duration(sec, picos);
        picoseconds_ = picos;
        return *this;
    }

    constexpr Duration& operator/=(int64_t scalar) noexcept {
        auto [sec, picos] = detail::div_time(seconds_, picoseconds_, scalar);
        seconds_ = detail::clamp_to_duration(sec, picos);
        picoseconds_ = picos;
        return *this;
    }

    friend constexpr Duration operator+(Duration lhs, Duration rhs) noexcept {
        lhs += rhs;
        return lhs;
    }

    friend constexpr Duration operator-(Duration lhs, Duration rhs) noexcept {
        lhs -= rhs;
        return lhs;
    }

    friend constexpr Duration operator*(Duration d, int64_t scalar) noexcept {
        d *= scalar;
        return d;
    }

    friend constexpr Duration operator*(int64_t scalar, Duration d) noexcept {
        d *= scalar;
        return d;
    }

    friend constexpr Duration operator/(Duration d, int64_t scalar) noexcept {
        d /= scalar;
        return d;
    }

    // Division of durations yields a ratio
    // Uses double arithmetic to support full ±68 year range
    friend constexpr int64_t operator/(Duration lhs, Duration rhs) noexcept {
        if (rhs.is_zero()) {
            // Saturate based on dividend sign
            return lhs.is_negative() ? std::numeric_limits<int64_t>::min()
                                     : std::numeric_limits<int64_t>::max();
        }
        // Use to_seconds() for full range support (double has ~15 significant digits)
        double ratio = lhs.to_seconds() / rhs.to_seconds();
        if (ratio >= static_cast<double>(std::numeric_limits<int64_t>::max())) {
            return std::numeric_limits<int64_t>::max();
        }
        if (ratio <= static_cast<double>(std::numeric_limits<int64_t>::min())) {
            return std::numeric_limits<int64_t>::min();
        }
        return static_cast<int64_t>(ratio);
    }

    // Comparison
    constexpr auto operator<=>(const Duration& other) const noexcept {
        if (seconds_ != other.seconds_) {
            return seconds_ <=> other.seconds_;
        }
        return picoseconds_ <=> other.picoseconds_;
    }

    constexpr bool operator==(const Duration& other) const noexcept {
        return seconds_ == other.seconds_ && picoseconds_ == other.picoseconds_;
    }

private:
    int32_t seconds_{0};
    uint64_t picoseconds_{0}; // Always in [0, PICOSECONDS_PER_SECOND)

    constexpr Duration(int32_t sec, uint64_t picos) noexcept : seconds_(sec), picoseconds_(picos) {}
};

/**
 * Check if a Duration has saturated to min() or max().
 *
 * Use this after arithmetic operations to detect overflow:
 * ```cpp
 * Duration result = a + b;
 * if (saturated(result)) {
 *     // Handle overflow - result is at min() or max()
 * }
 * ```
 *
 * Note: This cannot distinguish between a legitimate max/min value and
 * overflow saturation. With ±68 year range, legitimate max/min is rare.
 */
constexpr bool saturated(const Duration& d) noexcept {
    return d == Duration::max() || d == Duration::min();
}

/**
 * Lightweight duration using single int64_t picoseconds.
 *
 * Range: ±106 days (vs Duration's ±68 years)
 * Size: 8 bytes (vs Duration's 12 bytes)
 *
 * Use cases:
 * - Internal accumulators in hot loops
 * - Exact division/ratio without split-representation complexity
 * - Short-span calculations where 106 days suffices
 *
 * Not a drop-in Duration replacement. Explicitly convert via to_duration().
 * Saturates at ±INT64_MAX picoseconds (~106 days).
 *
 * This is a core library type: noexcept, no allocation.
 */
class ShortDuration {
public:
    static constexpr uint64_t PICOS_PER_SEC = detail::PICOS_PER_SEC;

    static constexpr ShortDuration zero() noexcept { return ShortDuration(0); }
    static constexpr ShortDuration max() noexcept {
        return ShortDuration(std::numeric_limits<int64_t>::max());
    }
    static constexpr ShortDuration min() noexcept {
        return ShortDuration(std::numeric_limits<int64_t>::min());
    }

    // Default construction - zero duration
    constexpr ShortDuration() noexcept = default;

    static constexpr ShortDuration from_picoseconds(int64_t ps) noexcept {
        return ShortDuration(ps);
    }

    // Convert from Duration - saturates if Duration exceeds ±106 days
    static constexpr ShortDuration from_duration(Duration d) noexcept {
        return ShortDuration(d.total_picoseconds()); // total_picoseconds() already saturates
    }

    // Factory from samples - saturates on overflow (defined after SamplePeriod)
    static ShortDuration from_samples(int64_t count, SamplePeriod period) noexcept;

    constexpr int64_t picoseconds() const noexcept { return picos_; }

    // Convert to full Duration (decomposes to seconds+picos via normalize)
    constexpr Duration to_duration() const noexcept { return Duration::from_picoseconds(picos_); }

    // Predicates
    constexpr bool is_zero() const noexcept { return picos_ == 0; }
    constexpr bool is_negative() const noexcept { return picos_ < 0; }
    constexpr bool is_positive() const noexcept { return picos_ > 0; }

    // Arithmetic with saturation
    constexpr ShortDuration& operator+=(ShortDuration other) noexcept {
        // Check for overflow before adding
        if (other.picos_ > 0 && picos_ > std::numeric_limits<int64_t>::max() - other.picos_) {
            picos_ = std::numeric_limits<int64_t>::max();
        } else if (other.picos_ < 0 &&
                   picos_ < std::numeric_limits<int64_t>::min() - other.picos_) {
            picos_ = std::numeric_limits<int64_t>::min();
        } else {
            picos_ += other.picos_;
        }
        return *this;
    }

    constexpr ShortDuration& operator-=(ShortDuration other) noexcept {
        // Check for overflow before subtracting
        if (other.picos_ < 0 && picos_ > std::numeric_limits<int64_t>::max() + other.picos_) {
            picos_ = std::numeric_limits<int64_t>::max();
        } else if (other.picos_ > 0 &&
                   picos_ < std::numeric_limits<int64_t>::min() + other.picos_) {
            picos_ = std::numeric_limits<int64_t>::min();
        } else {
            picos_ -= other.picos_;
        }
        return *this;
    }

    constexpr ShortDuration& operator*=(int64_t scalar) noexcept {
        if (scalar == 0 || picos_ == 0) {
            picos_ = 0;
            return *this;
        }
        // Check for overflow
        bool result_negative = (picos_ < 0) != (scalar < 0);
        uint64_t abs_picos =
            picos_ < 0 ? 0ULL - static_cast<uint64_t>(picos_) : static_cast<uint64_t>(picos_);
        uint64_t abs_scalar =
            scalar < 0 ? 0ULL - static_cast<uint64_t>(scalar) : static_cast<uint64_t>(scalar);

        constexpr uint64_t MAX_SAFE = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
        if (abs_picos > MAX_SAFE / abs_scalar) {
            picos_ = result_negative ? std::numeric_limits<int64_t>::min()
                                     : std::numeric_limits<int64_t>::max();
        } else {
            uint64_t product = abs_picos * abs_scalar;
            picos_ =
                result_negative ? -static_cast<int64_t>(product) : static_cast<int64_t>(product);
        }
        return *this;
    }

    constexpr ShortDuration& operator/=(int64_t scalar) noexcept {
        if (scalar == 0) {
            picos_ = picos_ < 0 ? std::numeric_limits<int64_t>::min()
                                : std::numeric_limits<int64_t>::max();
            return *this;
        }
        picos_ /= scalar;
        return *this;
    }

    friend constexpr ShortDuration operator+(ShortDuration a, ShortDuration b) noexcept {
        a += b;
        return a;
    }

    friend constexpr ShortDuration operator-(ShortDuration a, ShortDuration b) noexcept {
        a -= b;
        return a;
    }

    friend constexpr ShortDuration operator*(ShortDuration d, int64_t s) noexcept {
        d *= s;
        return d;
    }

    friend constexpr ShortDuration operator*(int64_t s, ShortDuration d) noexcept {
        d *= s;
        return d;
    }

    friend constexpr ShortDuration operator/(ShortDuration d, int64_t s) noexcept {
        d /= s;
        return d;
    }

    // Division of ShortDurations yields a ratio - saturates on divide-by-zero
    friend constexpr int64_t operator/(ShortDuration lhs, ShortDuration rhs) noexcept {
        if (rhs.picos_ == 0) {
            return lhs.picos_ < 0 ? std::numeric_limits<int64_t>::min()
                                  : std::numeric_limits<int64_t>::max();
        }
        return lhs.picos_ / rhs.picos_;
    }

    // Unary negation (saturates for min())
    constexpr ShortDuration operator-() const noexcept {
        if (picos_ == std::numeric_limits<int64_t>::min()) {
            return max(); // Saturate: -MIN would overflow
        }
        return ShortDuration(-picos_);
    }

    // Comparison
    constexpr auto operator<=>(const ShortDuration&) const noexcept = default;
    constexpr bool operator==(const ShortDuration&) const noexcept = default;

private:
    int64_t picos_{0};

    constexpr explicit ShortDuration(int64_t ps) noexcept : picos_(ps) {}
};

/**
 * Check if a ShortDuration has saturated to min() or max().
 */
constexpr bool saturated(ShortDuration d) noexcept {
    return d == ShortDuration::max() || d == ShortDuration::min();
}

// Define Duration::to_short_duration after ShortDuration is complete
inline std::optional<ShortDuration> Duration::to_short_duration() const noexcept {
    // total_picoseconds() saturates for durations > ~106 days
    int64_t picos = total_picoseconds();
    // If saturated (including Duration::max()/min()), value doesn't fit
    if (picos == std::numeric_limits<int64_t>::max() ||
        picos == std::numeric_limits<int64_t>::min()) {
        return std::nullopt;
    }
    return ShortDuration::from_picoseconds(picos);
}

/**
 * Unsigned sample period with exact picosecond representation.
 *
 * Tracks whether the period is exactly representable (for integer/rational inputs)
 * or approximate (for floating-point inputs with significant rounding error).
 *
 * Factory rules:
 * - from_picoseconds: always exact, clamps 0 to 1
 * - from_rate_hz/from_seconds: exact if error < 1 femtosecond (floating-point noise)
 * - from_ratio: exact when representable, nullopt otherwise
 *
 * This is a core library type: noexcept, no allocation.
 */
class SamplePeriod {
public:
    static constexpr uint64_t PICOSECONDS_PER_SECOND = 1'000'000'000'000ULL;

    /// Sub-femtosecond errors are floating-point noise, not meaningful for RF timing
    static constexpr double EXACTNESS_TOLERANCE_PICOS = 1e-6; // 1 femtosecond

    // Direct factory - clamps zero to 1ps (consistent with saturating design)
    static constexpr SamplePeriod from_picoseconds(uint64_t ps) noexcept {
        return SamplePeriod(ps > 0 ? ps : 1, true, 0.0);
    }

    // Checked factories - return nullopt on invalid input
    static std::optional<SamplePeriod> from_rate_hz(double rate_hz) noexcept {
        if (!std::isfinite(rate_hz) || rate_hz <= 0.0) {
            return std::nullopt;
        }

        // Use long double for intermediate calculation
        long double ideal_picos =
            static_cast<long double>(PICOSECONDS_PER_SECOND) / static_cast<long double>(rate_hz);

        if (ideal_picos > static_cast<long double>(std::numeric_limits<uint64_t>::max())) {
            return std::nullopt;
        }

        // Round to nearest, half away from zero
        uint64_t picos = static_cast<uint64_t>(std::llround(ideal_picos));
        if (picos == 0) {
            return std::nullopt;
        }

        // Compute error for reporting
        double error = static_cast<double>(ideal_picos) - static_cast<double>(picos);

        // Treat sub-femtosecond errors as exact (floating-point noise)
        bool exact = std::fabs(error) < EXACTNESS_TOLERANCE_PICOS;
        if (exact) {
            error = 0.0;
        }

        return SamplePeriod(picos, exact, error);
    }

    static std::optional<SamplePeriod> from_seconds(double period_seconds) noexcept {
        if (!std::isfinite(period_seconds) || period_seconds <= 0.0) {
            return std::nullopt;
        }

        // Use long double for intermediate calculation
        long double ideal_picos = static_cast<long double>(period_seconds) *
                                  static_cast<long double>(PICOSECONDS_PER_SECOND);

        if (ideal_picos > static_cast<long double>(std::numeric_limits<uint64_t>::max())) {
            return std::nullopt;
        }

        // Round to nearest, half away from zero
        uint64_t picos = static_cast<uint64_t>(std::llround(ideal_picos));
        if (picos == 0) {
            return std::nullopt;
        }

        // Compute error for reporting
        double error = static_cast<double>(ideal_picos) - static_cast<double>(picos);

        // Treat sub-femtosecond errors as exact (floating-point noise)
        bool exact = std::fabs(error) < EXACTNESS_TOLERANCE_PICOS;
        if (exact) {
            error = 0.0;
        }

        return SamplePeriod(picos, exact, error);
    }

    // Rational factory - exact when 10^12 * den is divisible by num
    static std::optional<SamplePeriod> from_ratio(uint64_t num, uint64_t den) noexcept {
        if (num == 0 || den == 0) {
            return std::nullopt;
        }

        // Check for overflow in intermediate calculation
        // picos = PICOSECONDS_PER_SECOND * den / num
        // First check if PICOSECONDS_PER_SECOND * den would overflow
        if (den > std::numeric_limits<uint64_t>::max() / PICOSECONDS_PER_SECOND) {
            // Try to simplify by dividing first if possible
            // This handles cases where den is large but divisible by num
            if (PICOSECONDS_PER_SECOND % num == 0) {
                uint64_t factor = PICOSECONDS_PER_SECOND / num;
                // Check if factor * den would overflow
                if (den > std::numeric_limits<uint64_t>::max() / factor) {
                    return std::nullopt;
                }
                uint64_t picos = factor * den;
                return SamplePeriod(picos, true, 0.0);
            }
            return std::nullopt;
        }

        uint64_t numerator = PICOSECONDS_PER_SECOND * den;

        // Check if exactly divisible
        if (numerator % num != 0) {
            return std::nullopt;
        }

        uint64_t picos = numerator / num;
        if (picos == 0) {
            return std::nullopt;
        }

        return SamplePeriod(picos, true, 0.0);
    }

    // Accessors
    constexpr uint64_t picoseconds() const noexcept { return picos_; }

    double rate_hz() const noexcept {
        return static_cast<double>(PICOSECONDS_PER_SECOND) / static_cast<double>(picos_);
    }

    double seconds() const noexcept {
        return static_cast<double>(picos_) / static_cast<double>(PICOSECONDS_PER_SECOND);
    }

    // Exactness and error reporting
    constexpr bool is_exact() const noexcept { return exact_; }

    double error_picoseconds() const noexcept { return error_picos_; }

    double error_ppm() const noexcept {
        if (exact_) {
            return 0.0;
        }
        // For very small periods, this may return ±inf (documented sentinel)
        // For very large errors with precision loss, this may also return ±inf
        double ppm = (error_picos_ / static_cast<double>(picos_)) * 1e6;
        return ppm;
    }

    // Conversion to Duration
    constexpr Duration to_duration() const noexcept {
        // SamplePeriod picos is always positive and < UINT64_MAX
        // Duration can handle values up to ~68 years
        // UINT64_MAX picos ≈ 213 days, so this always fits
        if (picos_ >= PICOSECONDS_PER_SECOND) {
            uint64_t sec = picos_ / PICOSECONDS_PER_SECOND;
            uint64_t picos = picos_ % PICOSECONDS_PER_SECOND;
            if (sec > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
                return Duration::max();
            }
            return Duration::from_seconds(static_cast<int64_t>(sec)) +
                   Duration::from_picoseconds(static_cast<int64_t>(picos));
        }
        return Duration::from_picoseconds(static_cast<int64_t>(picos_));
    }

    // Comparison
    constexpr auto operator<=>(const SamplePeriod&) const noexcept = default;
    constexpr bool operator==(const SamplePeriod& other) const noexcept {
        return picos_ == other.picos_;
    }

private:
    uint64_t picos_;
    bool exact_;
    double error_picos_;

    constexpr SamplePeriod(uint64_t ps, bool exact, double error) noexcept
        : picos_(ps),
          exact_(exact),
          error_picos_(error) {}
};

// Define ShortDuration::from_samples after SamplePeriod is complete
inline ShortDuration ShortDuration::from_samples(int64_t count, SamplePeriod period) noexcept {
    // Check for overflow before multiplying
    // Period picos is always positive (uint64_t), count can be negative
    uint64_t period_picos = period.picoseconds();
    uint64_t abs_count =
        count < 0 ? 0ULL - static_cast<uint64_t>(count) : static_cast<uint64_t>(count);

    constexpr uint64_t MAX_SAFE = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());

    // Check for overflow in multiplication
    if (abs_count > 0 && period_picos > MAX_SAFE / abs_count) {
        return count < 0 ? min() : max();
    }

    uint64_t product = period_picos * abs_count;
    return count < 0 ? ShortDuration::from_picoseconds(-static_cast<int64_t>(product))
                     : ShortDuration::from_picoseconds(static_cast<int64_t>(product));
}

// Define Duration::from_samples after SamplePeriod is complete
inline std::optional<Duration> Duration::from_samples(int64_t count, SamplePeriod period) noexcept {
    // Convert period to Duration, then multiply
    Duration period_duration = period.to_duration();

    // Multiply by count - this saturates on overflow
    Duration result = period_duration * count;

    // Check if we saturated (indicating overflow)
    if (saturated(result) && count != 0 && !saturated(period_duration)) {
        return std::nullopt;
    }

    return result;
}

} // namespace vrtigo
