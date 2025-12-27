#pragma once

#include "vrtigo/duration.hpp"
#include "vrtigo/timestamp.hpp"

#include <cstdint>

namespace vrtigo::utils {

/**
 * @brief Deferred time specification for SampleClock initialization
 *
 * Enables patterns like "start at next second boundary" or "start now + 100ms"
 * without requiring the caller to compute timestamps manually.
 *
 * StartTime produces UtcRealTimestamp because now() only makes sense for UTC
 * wall-clock access. Use absolute() to provide a pre-computed timestamp.
 *
 * @note Default-constructed StartTime equals StartTime::zero() (epoch 0,0).
 * @note resolve() may be called multiple times. Each call captures a fresh
 *       timestamp for now/next_second bases. Caller controls resolution timing.
 *
 * Example usage:
 * @code
 *   SampleClock clock(1e-6, StartTime::now());              // Start immediately
 *   SampleClock clock(1e-6, StartTime::now_plus(
 *       Duration::from_milliseconds(500)));                 // Start after setup time
 *   SampleClock clock(1e-6, StartTime::at_next_second());   // PPS alignment
 *   SampleClock clock(1e-6, StartTime::at_next_second_plus(
 *       Duration::from_milliseconds(100)));                 // PPS + offset
 * @endcode
 */
struct StartTime {
    /// Base time reference type
    enum class Base : uint8_t {
        now,         ///< Capture wall clock at resolve()
        next_second, ///< Round up to next second boundary, then apply offset
        absolute,    ///< Use provided timestamp exactly
        zero         ///< Epoch (0, 0)
    };

    /// Default construction equals zero() (epoch 0,0)
    constexpr StartTime() noexcept = default;

    // === Core factories ===

    /// Start from current wall-clock time
    static constexpr StartTime now() noexcept { return StartTime{Base::now, {}, {}}; }

    /// Start from current time + offset
    static constexpr StartTime now_plus(Duration offset) noexcept {
        return StartTime{Base::now, {}, offset};
    }

    /// Start from explicit timestamp (bypasses wall clock entirely)
    static constexpr StartTime absolute(UtcRealTimestamp time) noexcept {
        return StartTime{Base::absolute, time, {}};
    }

    /// Start from epoch (0, 0) - default state
    static constexpr StartTime zero() noexcept { return StartTime{Base::zero, {}, {}}; }

    // === Second-boundary factories (for PPS alignment) ===

    /// Start at next whole-second boundary (ceiling semantics)
    /// If already exactly on boundary, returns that boundary.
    static constexpr StartTime at_next_second() noexcept {
        return StartTime{Base::next_second, {}, {}};
    }

    /// Start at next whole-second boundary + offset
    /// Example: at_next_second_plus(Duration::from_milliseconds(100)) for "100ms after PPS edge"
    static constexpr StartTime at_next_second_plus(Duration offset) noexcept {
        return StartTime{Base::next_second, {}, offset};
    }

    /// Resolve to concrete timestamp
    /// @note May be called multiple times; captures fresh wall-clock for now/next_second bases
    UtcRealTimestamp resolve() const noexcept {
        switch (base_) {
            case Base::now:
                return UtcRealTimestamp::now() + offset_;

            case Base::next_second: {
                auto t = UtcRealTimestamp::now();
                // Ceiling: if fractional > 0, advance to next second
                if (t.tsf() > 0) {
                    uint32_t next_sec = t.tsi() < UINT32_MAX ? t.tsi() + 1 : UINT32_MAX;
                    t = UtcRealTimestamp(next_sec, 0);
                }
                return t + offset_;
            }

            case Base::absolute:
                return absolute_time_;

            case Base::zero:
            default:
                return UtcRealTimestamp{};
        }
    }

    /// Get the base time reference type
    constexpr Base base() const noexcept { return base_; }

    /// Get the offset (only meaningful for now and next_second bases)
    constexpr Duration offset() const noexcept { return offset_; }

private:
    constexpr StartTime(Base base, UtcRealTimestamp abs_time, Duration offset) noexcept
        : base_(base),
          absolute_time_(abs_time),
          offset_(offset) {}

    Base base_{Base::zero};
    UtcRealTimestamp absolute_time_{};
    Duration offset_{};
};

} // namespace vrtigo::utils
