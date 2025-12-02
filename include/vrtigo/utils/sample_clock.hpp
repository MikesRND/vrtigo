#pragma once

#include "vrtigo/timestamp.hpp"
#include "vrtigo/utils/start_time.hpp"

#include <concepts>
#include <stdexcept>

#include <cmath>
#include <cstdint>

namespace vrtigo::utils {

/**
 * @brief Concept defining the common interface for all SampleClock specializations
 *
 * All SampleClock specializations must provide:
 * - time_point type alias
 * - now() to query current time without advancing
 * - tick() to advance by one sample
 * - tick(n) to advance by N samples
 * - reset() to reset to zero
 * - elapsed_samples() to query total samples elapsed
 */
template <typename T>
concept SampleClockLike = requires(T clock, uint64_t n) {
    typename T::time_point;
    { clock.now() } -> std::same_as<typename T::time_point>;
    { clock.tick() } -> std::same_as<typename T::time_point>;
    { clock.tick(n) } -> std::same_as<typename T::time_point>;
    { clock.reset() } -> std::same_as<void>;
    { clock.elapsed_samples() } -> std::same_as<uint64_t>;
};

/**
 * @brief Synthetic sample clock for generating timestamps at fixed sample intervals
 *
 * Template specializations exist for different TsfType values:
 * - TsfType::real_time: Generates picosecond-precision timestamps (implemented)
 * - TsfType::sample_count: Generates sample-numbered timestamps (not yet implemented)
 *
 * @tparam TSI Timestamp Integer type (default: TsiType::other)
 * @tparam TSF Timestamp Fractional type (default: TsfType::real_time)
 */
template <TsiType TSI = TsiType::other, TsfType TSF = TsfType::real_time>
class SampleClock;

/**
 * @brief SampleClock specialization for real-time timestamps (TsfType::real_time)
 *
 * Provides deterministic timestamp generation for sample-based systems. The input sample period
 * is rounded to the nearest picosecond once at construction and used consistently thereafter.
 * Use is_exact() to check whether rounding occurred.
 *
 * @warning Maximum runtime: ~213 days. The internal uint64_t picosecond accumulator overflows
 *          at UINT64_MAX picoseconds (~213.5 days). tick() operations throw overflow_error after
 *          this limit.
 *
 * @note Thread safety: const methods are safe for concurrent reads. Non-const methods require
 *       external synchronization.
 *
 * @note Default TSI is "other" to avoid implying UTC/GPS semantics. Callers can opt into utc
 *       TSI when needed for interop, but the clock remains synthetic and leap-second free.
 *
 * @tparam TSI Timestamp Integer type
 */
template <TsiType TSI>
class SampleClock<TSI, TsfType::real_time> {
public:
    using time_point = Timestamp<TSI, TsfType::real_time>;

    /**
     * @brief Construct a sample clock with specified period
     * @param sample_period_seconds Sample period in seconds (rounded to nearest picosecond)
     * @param start Starting timestamp (default: zero)
     * @throws std::invalid_argument if period is non-finite, zero, negative, or rounds to zero
     * @throws std::overflow_error if period overflows uint64_t picoseconds
     */
    explicit SampleClock(double sample_period_seconds, time_point start = {})
        : sample_period_picos_(normalize_period(sample_period_seconds)),
          period_error_picos_(compute_period_error(sample_period_seconds, sample_period_picos_)),
          period_is_exact_(period_error_picos_ == 0.0L),
          start_seconds_(start.tsi()),
          start_fractional_(start.tsf()) {}

    /**
     * @brief Construct a sample clock with deferred start time resolution
     * @param sample_period_seconds Sample period in seconds (rounded to nearest picosecond)
     * @param start_spec StartTime specification (resolved at construction)
     * @throws std::invalid_argument if period is non-finite, zero, negative, or rounds to zero
     * @throws std::overflow_error if period overflows uint64_t picoseconds
     *
     * @note Constrained to UTC and "other" TSI types to prevent silent GPS epoch mismatch.
     *       StartTime::now() captures UTC wall-clock time; using it with TsiType::gps would
     *       produce timestamps with incorrect epoch semantics (~18 second offset).
     *
     * Example:
     * @code
     *   SampleClock<TsiType::utc> clock(1e-6, StartTime::at_next_second()); // PPS aligned
     * @endcode
     */
    explicit SampleClock(double sample_period_seconds, StartTime start_spec)
        requires(TSI == TsiType::utc || TSI == TsiType::other)
        : SampleClock(sample_period_seconds, make_time_point_from_start(start_spec)) {}

    /// Get current time without advancing
    time_point now() const { return make_time_point(total_picos_); }

    /// Advance by 1 sample and return new time
    /// @throws std::overflow_error if accumulator exceeds ~213 days
    time_point tick() { return tick(1); }

    /// Advance by N samples and return new time
    /// @throws std::overflow_error if accumulator exceeds ~213 days or sample count too large
    time_point tick(uint64_t samples) {
        uint64_t delta_picos = checked_multiply(sample_period_picos_, samples);
        total_picos_ = checked_add(total_picos_, delta_picos);
        return make_time_point(total_picos_);
    }

    /// Advance by N samples without returning time
    /// @throws std::overflow_error if accumulator exceeds ~213 days or sample count too large
    void advance(uint64_t samples) {
        uint64_t delta_picos = checked_multiply(sample_period_picos_, samples);
        total_picos_ = checked_add(total_picos_, delta_picos);
    }

    void reset() noexcept { total_picos_ = 0; }

    void reset(time_point start) noexcept {
        total_picos_ = 0;
        start_seconds_ = start.tsi();
        start_fractional_ = start.tsf();
    }

    uint64_t sample_period_picoseconds() const noexcept { return sample_period_picos_; }

    double sample_period() const noexcept {
        return static_cast<double>(sample_period_picos_) /
               static_cast<double>(time_point::PICOSECONDS_PER_SECOND);
    }

    double sample_rate() const noexcept {
        // sample_period_picos_ is guaranteed > 0 after construction
        return static_cast<double>(time_point::PICOSECONDS_PER_SECOND) /
               static_cast<double>(sample_period_picos_);
    }

    /// Returns true if requested sample period is exactly representable in picoseconds
    bool is_exact() const noexcept { return period_is_exact_; }

    /// Returns the rounding error in picoseconds (requested - actual)
    long double error_picoseconds() const noexcept { return period_error_picos_; }

    /// Returns the rounding error in parts per million (ppm)
    /// This represents both period and frequency error (they're equal for small errors)
    double error_ppm() const noexcept {
        if (sample_period_picos_ == 0) {
            return 0.0;
        }
        return (static_cast<double>(period_error_picos_) /
                static_cast<double>(sample_period_picos_)) *
               1e6;
    }

    uint64_t elapsed_samples() const noexcept { return total_picos_ / sample_period_picos_; }

private:
    static time_point make_time_point_from_start(StartTime start_spec) noexcept {
        auto resolved = start_spec.resolve();
        return time_point(resolved.tsi(), resolved.tsf());
    }

    static uint64_t normalize_period(double seconds) {
        if (!std::isfinite(seconds)) {
            throw std::invalid_argument("sample period must be finite");
        }
        if (seconds <= 0.0) {
            throw std::invalid_argument("sample period must be positive");
        }

        long double scaled = static_cast<long double>(seconds) *
                             static_cast<long double>(time_point::PICOSECONDS_PER_SECOND);

        if (scaled > static_cast<long double>(UINT64_MAX)) {
            throw std::overflow_error("sample period in picoseconds overflows uint64_t");
        }

        uint64_t rounded = static_cast<uint64_t>(std::llround(scaled));
        if (rounded == 0) {
            throw std::invalid_argument("sample period rounds to zero picoseconds");
        }
        return rounded;
    }

    static long double compute_period_error(double seconds, uint64_t rounded) noexcept {
        if (!std::isfinite(seconds) || seconds <= 0.0) {
            return 0.0L;
        }

        long double scaled = static_cast<long double>(seconds) *
                             static_cast<long double>(time_point::PICOSECONDS_PER_SECOND);

        if (scaled > static_cast<long double>(UINT64_MAX)) {
            return 0.0L;
        }

        // Return signed error: positive means requested > actual (rounded down)
        long double error = scaled - static_cast<long double>(rounded);

        // Treat errors below 1 femtosecond as zero (floating-point noise)
        // Sub-femtosecond errors are physically meaningless for RF timing
        constexpr long double ROUNDING_TOLERANCE_PICOS = 1e-6L; // 1 femtosecond
        if (std::fabs(error) < ROUNDING_TOLERANCE_PICOS) {
            return 0.0L;
        }

        return error;
    }

    static uint64_t checked_add(uint64_t a, uint64_t b) {
        uint64_t sum = a + b;
        if (sum < a) {
            throw std::overflow_error("SampleClock tick overflow");
        }
        return sum;
    }

    static uint64_t checked_multiply(uint64_t a, uint64_t b) {
        if (a == 0 || b == 0) {
            return 0;
        }
        if (a > (UINT64_MAX / b)) {
            throw std::overflow_error("SampleClock tick overflow");
        }
        return a * b;
    }

    time_point make_time_point(uint64_t offset_picos) const noexcept {
        uint64_t fractional =
            start_fractional_ + (offset_picos % time_point::PICOSECONDS_PER_SECOND);
        uint64_t carry_seconds = fractional / time_point::PICOSECONDS_PER_SECOND;
        fractional %= time_point::PICOSECONDS_PER_SECOND;

        uint64_t seconds = static_cast<uint64_t>(start_seconds_) +
                           (offset_picos / time_point::PICOSECONDS_PER_SECOND) + carry_seconds;

        if (seconds > UINT32_MAX) {
            return time_point(UINT32_MAX, time_point::MAX_FRACTIONAL);
        }

        return time_point(static_cast<uint32_t>(seconds), fractional);
    }

    uint64_t sample_period_picos_{1};
    long double period_error_picos_{0.0L};
    bool period_is_exact_{true};
    uint64_t total_picos_{0};
    uint32_t start_seconds_{0};
    uint64_t start_fractional_{0};
};

/**
 * @brief SampleClock specialization for sample-count timestamps (TsfType::sample_count)
 *
 * @note This specialization is not yet implemented.
 *
 * @tparam TSI Timestamp Integer type
 */
template <TsiType TSI>
class SampleClock<TSI, TsfType::sample_count> {
    static_assert(TSI != TSI, "SampleClock with TsfType::sample_count is not yet implemented. "
                              "Only TsfType::real_time is currently supported.");
};

// Verify that specializations conform to the SampleClockLike concept
// Check the real_time specialization with a concrete instantiation
static_assert(SampleClockLike<SampleClock<TsiType::other, TsfType::real_time>>,
              "SampleClock<TSI, TsfType::real_time> must satisfy SampleClockLike concept");

} // namespace vrtigo::utils
