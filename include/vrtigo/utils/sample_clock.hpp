#pragma once

#include "vrtigo/duration.hpp"
#include "vrtigo/timestamp.hpp"
#include "vrtigo/utils/start_time.hpp"

#include <stdexcept>

#include <cmath>
#include <cstdint>

namespace vrtigo::utils {

/**
 * @brief Synthetic sample clock for generating timestamps at fixed sample intervals
 *
 * ## Overflow Policy
 * Uses saturation semantics - arithmetic saturates to maximum timestamp on overflow.
 * With the Duration-based accumulator, overflow occurs only after ~68 years of
 * continuous operation (vs previous ~213 day limit).
 *
 * **Detection:** Use `saturated(ts)` helper to check if returned timestamp hit max.
 * Alternatively, compare against previous timestamp - if `ts <= prev_ts`, the clock
 * has saturated.
 *
 * **Rationale:** With 68-year range, overflow indicates extraordinary circumstances
 * (programmer error or truly exceptional runtime). No `expected<>` return types,
 * no exceptions - callers needing safety must guard hot paths.
 *
 * Template specializations exist for different TsfType values:
 * - TsfType::real_time: Generates picosecond-precision timestamps (implemented)
 * - TsfType::sample_count: Generates sample-numbered timestamps (not yet implemented)
 *
 * @tparam TSI Timestamp Integer type (default: TsiType::utc)
 * @tparam TSF Timestamp Fractional type (default: TsfType::real_time)
 */
template <TsiType TSI = TsiType::utc, TsfType TSF = TsfType::real_time>
class SampleClock;

/**
 * @brief SampleClock specialization for real-time timestamps (TsfType::real_time)
 *
 * Provides deterministic timestamp generation for sample-based systems. The input sample period
 * is rounded to the nearest picosecond once at construction and used consistently thereafter.
 * Use period().is_exact() to check whether rounding occurred.
 *
 * @note Thread safety: const methods are safe for concurrent reads. Non-const methods require
 *       external synchronization.
 *
 * @note Default TSI is UTC for common interop. The clock remains synthetic and leap-second free.
 *
 * @tparam TSI Timestamp Integer type
 */
template <TsiType TSI>
class SampleClock<TSI, TsfType::real_time> {
public:
    using time_point = Timestamp<TSI, TsfType::real_time>;

    /**
     * @brief Construct a sample clock with specified period and start time
     * @param sample_period_seconds Sample period in seconds (rounded to nearest picosecond)
     * @param start_spec StartTime specification (resolved at construction, stored for reset)
     * @throws std::invalid_argument if period is non-finite, zero, negative, or rounds to zero
     *
     * @note Constrained to UTC and "other" TSI types to prevent silent GPS epoch mismatch.
     *       StartTime::now() captures UTC wall-clock time; using it with TsiType::gps would
     *       produce timestamps with incorrect epoch semantics (~18 second offset).
     *
     * Example:
     * @code
     *   SampleClock<TsiType::utc> clock(1e-6, StartTime::at_next_second()); // PPS aligned
     *   clock.reset();  // Re-resolves to next second boundary
     * @endcode
     */
    explicit SampleClock(double sample_period_seconds, StartTime start_spec = {})
        requires(TSI == TsiType::utc || TSI == TsiType::other)
        : period_(make_period(sample_period_seconds)),
          start_spec_(start_spec) {
        resolve_start();
    }

    /**
     * @brief Convenience constructor for absolute start time
     * @param sample_period_seconds Sample period in seconds
     * @param start Absolute start timestamp (stored as StartTime::absolute for reset)
     */
    explicit SampleClock(double sample_period_seconds, time_point start)
        requires(TSI == TsiType::utc || TSI == TsiType::other)
        : SampleClock(sample_period_seconds,
                      StartTime::absolute(UtcRealTimestamp(start.tsi(), start.tsf()))) {}

    /// Get current time without advancing
    time_point now() const noexcept { return start_ + accumulated_; }

    /**
     * Advance by 1 sample and return new time
     *
     * Returns the new timestamp directly. On overflow (after ~68 years), the timestamp
     * saturates to maximum value. Use `saturated(result)` to detect overflow if needed.
     */
    time_point tick() noexcept { return tick(1); }

    /**
     * Advance by N samples and return new time
     *
     * Returns the new timestamp directly. On overflow, the timestamp saturates to
     * maximum value. Use `saturated(result)` to detect overflow if needed.
     */
    time_point tick(uint64_t samples) noexcept {
        advance(samples);
        return now();
    }

    /**
     * Advance by N samples without returning time
     *
     * Saturates accumulated duration on overflow (after ~68 years).
     */
    void advance(uint64_t samples) noexcept {
        // Compute delta as Duration: period * samples
        // This saturates internally on multiplication overflow
        Duration delta = period_.to_duration() * static_cast<int64_t>(samples);
        accumulated_ += delta;
    }

    /// Reset elapsed samples and re-resolve start time from stored spec
    void reset() noexcept {
        accumulated_ = Duration::zero();
        resolve_start();
    }

    /// Access the sample period (use .picoseconds(), .seconds(), .rate_hz(), .is_exact(), etc.)
    const SamplePeriod& period() const noexcept { return period_; }

    /// Number of samples elapsed since construction or last reset
    uint64_t elapsed_samples() const noexcept {
        // Use Duration division which supports full ±68 year range via double arithmetic
        Duration period_dur = period_.to_duration();
        if (period_dur.is_zero())
            return 0;
        int64_t samples = accumulated_ / period_dur;
        return samples >= 0 ? static_cast<uint64_t>(samples) : 0;
    }

private:
    void resolve_start() noexcept {
        auto resolved = start_spec_.resolve();
        start_ = time_point(resolved.tsi(), resolved.tsf());
    }

    static SamplePeriod make_period(double seconds) {
        if (!std::isfinite(seconds)) {
            throw std::invalid_argument("sample period must be finite");
        }
        if (seconds <= 0.0) {
            throw std::invalid_argument("sample period must be positive");
        }
        auto period = SamplePeriod::from_seconds(seconds);
        if (!period) {
            throw std::invalid_argument("sample period rounds to zero or overflows");
        }
        return *period;
    }

    SamplePeriod period_{SamplePeriod::from_picoseconds(1)};
    Duration accumulated_{};
    StartTime start_spec_{};
    time_point start_{};
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

} // namespace vrtigo::utils
