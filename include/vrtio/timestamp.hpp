#pragma once

#include "vrtio/types.hpp"

#include <chrono>
#include <compare>

#include <cstdint>
#include <ctime>

namespace vrtio {

// Marker type for no timestamp
struct NoTimeStamp {};

// Alias for backwards compatibility
using Timestamp32Lsb = NoTimeStamp;

// Primary template - works for ALL timestamp combinations
template <TsiType TSI, TsfType TSF>
class TimeStamp {
    // Helper constant for readability and maintenance
    static constexpr bool is_utc_real_time = (TSI == TsiType::utc && TSF == TsfType::real_time);

public:
    // Constants (always available, compiler optimizes away if unused)
    static constexpr uint64_t PICOSECONDS_PER_SECOND = 1'000'000'000'000ULL;
    static constexpr uint64_t NANOSECONDS_PER_SECOND = 1'000'000'000ULL;
    static constexpr uint64_t PICOSECONDS_PER_NANOSECOND = 1'000ULL;
    static constexpr uint64_t MAX_FRACTIONAL = PICOSECONDS_PER_SECOND - 1;

    // Constructors - basic API available for all timestamp types
    constexpr TimeStamp() noexcept = default;

    constexpr TimeStamp(uint32_t sec, uint64_t frac) noexcept : seconds_(sec), fractional_(frac) {
        normalize();
    }

    // Factory methods - basic API
    static constexpr TimeStamp fromComponents(uint32_t sec, uint64_t frac) noexcept {
        return TimeStamp(sec, frac);
    }

    // Accessors - basic API available for all types
    constexpr uint32_t seconds() const noexcept { return seconds_; }
    constexpr uint64_t fractional() const noexcept { return fractional_; }

    constexpr TsiType tsiType() const noexcept { return TSI; }
    constexpr TsfType tsfType() const noexcept { return TSF; }

    // Comparison operators - basic API available for all types
    constexpr auto operator<=>(const TimeStamp& other) const noexcept = default;

    // UTC-specific factory methods
    static TimeStamp now() noexcept
        requires(is_utc_real_time)
    {
        auto now = std::chrono::system_clock::now();
        return fromChrono(now);
    }

    static constexpr TimeStamp fromUTCSeconds(uint32_t seconds) noexcept
        requires(is_utc_real_time)
    {
        return TimeStamp(seconds, 0);
    }

    static TimeStamp fromChrono(std::chrono::system_clock::time_point tp) noexcept
        requires(is_utc_real_time)
    {
        // Convert to duration since epoch
        auto duration = tp.time_since_epoch();

        // Extract seconds and nanoseconds
        auto secs = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(duration) - secs;

        // Convert to VRT format (uint32_t seconds, fractional picoseconds)
        // Handle pre-epoch (negative) and post-2106 (> UINT32_MAX) times
        int64_t epoch_seconds = secs.count();
        uint32_t vrt_seconds;
        uint64_t vrt_fractional;

        if (epoch_seconds < 0) {
            // Pre-epoch time (before 1970-01-01) - clamp to zero
            vrt_seconds = 0;
            vrt_fractional = 0;
        } else if (epoch_seconds > static_cast<int64_t>(UINT32_MAX)) {
            // Post-2106 time (after ~2106-02-07) - clamp to max
            vrt_seconds = UINT32_MAX;
            vrt_fractional = MAX_FRACTIONAL;
        } else {
            // Normal case: time fits in uint32_t
            vrt_seconds = static_cast<uint32_t>(epoch_seconds);
            // nanos.count() could be negative for times just after a second boundary
            // due to floating point rounding, but should be in range [0, 999999999]
            int64_t nanos_count = nanos.count();
            if (nanos_count < 0) {
                // Handle rare case of negative nanoseconds (shouldn't happen but be safe)
                vrt_fractional = 0;
            } else {
                vrt_fractional = static_cast<uint64_t>(nanos_count) * PICOSECONDS_PER_NANOSECOND;
            }
        }

        return TimeStamp(vrt_seconds, vrt_fractional);
    }

    // UTC-specific conversion methods
    std::chrono::system_clock::time_point toChrono() const noexcept
        requires(is_utc_real_time)
    {
        // Convert seconds to duration
        auto sec_duration = std::chrono::seconds(seconds_);

        // Convert fractional to nanoseconds (losing sub-nanosecond precision)
        auto nano_duration = std::chrono::nanoseconds(fractional_ / PICOSECONDS_PER_NANOSECOND);

        // Create time_point from epoch
        return std::chrono::system_clock::time_point(sec_duration + nano_duration);
    }

    std::time_t toTimeT() const noexcept
        requires(is_utc_real_time)
    {
        return static_cast<std::time_t>(seconds_);
    }

    // UTC-specific utility method
    constexpr uint64_t totalPicoseconds() const noexcept
        requires(is_utc_real_time)
    {
        // Check if multiplication would overflow
        // UINT64_MAX / PICOSECONDS_PER_SECOND ≈ 18,446,744 seconds (~213 days)
        constexpr uint32_t MAX_SAFE_SECONDS =
            static_cast<uint32_t>(UINT64_MAX / PICOSECONDS_PER_SECOND);

        if (seconds_ > MAX_SAFE_SECONDS) {
            // Would overflow - return saturated value
            return UINT64_MAX;
        }

        uint64_t total = static_cast<uint64_t>(seconds_) * PICOSECONDS_PER_SECOND;

        // Check if adding fractional would overflow
        if (fractional_ > (UINT64_MAX - total)) {
            return UINT64_MAX;
        }

        return total + fractional_;
    }

    // UTC-specific arithmetic operations
    TimeStamp& operator+=(std::chrono::nanoseconds duration) noexcept
        requires(is_utc_real_time)
    {
        // Decompose duration into seconds and nanosecond remainder to avoid overflow
        // This handles durations up to the full range of nanoseconds (~292 years)
        auto sec_duration = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto nano_remainder = duration - sec_duration;

        int64_t seconds_to_add = sec_duration.count();
        int64_t nanos_to_add = nano_remainder.count();

        // Handle seconds component
        if (seconds_to_add >= 0) {
            // Adding positive seconds
            uint64_t new_seconds =
                static_cast<uint64_t>(seconds_) + static_cast<uint64_t>(seconds_to_add);
            if (new_seconds > UINT32_MAX) {
                // Overflow - clamp to max
                seconds_ = UINT32_MAX;
                fractional_ = MAX_FRACTIONAL;
                return *this;
            }
            seconds_ = static_cast<uint32_t>(new_seconds);
        } else {
            // Subtracting seconds
            uint64_t secs_to_sub = static_cast<uint64_t>(-seconds_to_add);
            if (secs_to_sub > seconds_) {
                // Underflow - clamp to zero
                seconds_ = 0;
                fractional_ = 0;
                return *this;
            }
            seconds_ -= static_cast<uint32_t>(secs_to_sub);
        }

        // Handle nanosecond remainder (always < 1 second in magnitude)
        // Safe to multiply by 1000 since |nanos_to_add| < 10^9
        int64_t picos_to_add = nanos_to_add * static_cast<int64_t>(PICOSECONDS_PER_NANOSECOND);

        if (picos_to_add >= 0) {
            // Adding positive picoseconds
            fractional_ += static_cast<uint64_t>(picos_to_add);
            normalize(); // Handle carry to seconds if needed
        } else {
            // Subtracting picoseconds - handle multi-second borrow properly
            uint64_t picos_to_sub = static_cast<uint64_t>(-picos_to_add);

            if (fractional_ >= picos_to_sub) {
                // Simple case: no borrow needed
                fractional_ -= picos_to_sub;
            } else {
                // Need to borrow from seconds
                // Calculate how many full seconds we need to borrow
                uint64_t deficit = picos_to_sub - fractional_;
                uint32_t seconds_to_borrow = static_cast<uint32_t>(
                    (deficit + PICOSECONDS_PER_SECOND - 1) / PICOSECONDS_PER_SECOND);

                if (seconds_to_borrow > seconds_) {
                    // Would underflow - clamp to zero
                    seconds_ = 0;
                    fractional_ = 0;
                } else {
                    // Borrow the seconds and adjust fractional
                    seconds_ -= seconds_to_borrow;
                    fractional_ =
                        (seconds_to_borrow * PICOSECONDS_PER_SECOND + fractional_) - picos_to_sub;
                }
            }
        }

        return *this;
    }

    TimeStamp& operator-=(std::chrono::nanoseconds duration) noexcept
        requires(is_utc_real_time)
    {
        // Guard against nanoseconds::min() which cannot be negated
        if (duration == std::chrono::nanoseconds::min()) {
            // nanoseconds::min() is approximately -9.22e18
            // Subtracting it means: *this - (-9.22e18) = *this + 9.22e18
            // We can't negate min() directly, so we add max() then add 1
            // This gives us: *this + max() + 1 = *this + abs(min())
            *this += std::chrono::nanoseconds::max();
            return *this += std::chrono::nanoseconds(1);
        }

        // Normal case: negate and add
        return *this += -duration;
    }

    // Friend operators with inline definitions (UTC-specific)
    friend TimeStamp operator+(TimeStamp ts, std::chrono::nanoseconds duration) noexcept
        requires(is_utc_real_time)
    {
        ts += duration;
        return ts;
    }

    friend TimeStamp operator-(TimeStamp ts, std::chrono::nanoseconds duration) noexcept
        requires(is_utc_real_time)
    {
        ts -= duration;
        return ts;
    }

    friend std::chrono::nanoseconds operator-(const TimeStamp& lhs, const TimeStamp& rhs) noexcept
        requires(is_utc_real_time)
    {
        int64_t sec_diff = static_cast<int64_t>(lhs.seconds_) - static_cast<int64_t>(rhs.seconds_);
        int64_t frac_diff =
            static_cast<int64_t>(lhs.fractional_) - static_cast<int64_t>(rhs.fractional_);

        // Convert seconds to nanoseconds directly to avoid overflow
        // This implementation: sec_diff * 10^9 is safe for differences up to ~292 years
        int64_t total_nanos = sec_diff * static_cast<int64_t>(NANOSECONDS_PER_SECOND);

        // Add the fractional difference converted to nanoseconds
        // Since fractional_ is always < 1 second (for real_time), frac_diff is in range [-10^12,
        // +10^12] Dividing by 1000 gives range [-10^9, +10^9] which fits safely in int64_t
        total_nanos += frac_diff / static_cast<int64_t>(PICOSECONDS_PER_NANOSECOND);

        return std::chrono::nanoseconds(total_nanos);
    }

private:
    uint32_t seconds_{0};    // TSI component - default initialized to zero
    uint64_t fractional_{0}; // TSF component - default initialized to zero

    // TSF-aware normalization
    constexpr void normalize() noexcept {
        if constexpr (TSF == TsfType::real_time) {
            // Only normalize for real_time TSF
            if (fractional_ >= PICOSECONDS_PER_SECOND) {
                uint32_t extra_seconds =
                    static_cast<uint32_t>(fractional_ / PICOSECONDS_PER_SECOND);

                // Check for overflow before adding
                if (extra_seconds > (UINT32_MAX - seconds_)) {
                    // Would overflow - clamp to max
                    seconds_ = UINT32_MAX;
                    fractional_ = MAX_FRACTIONAL;
                } else {
                    seconds_ += extra_seconds;
                    fractional_ %= PICOSECONDS_PER_SECOND;
                }
            }
        }
        // No normalization for sample_count, free_running, none
    }
};

// Convenient type alias
using TimeStampUTC = TimeStamp<TsiType::utc, TsfType::real_time>;

} // namespace vrtio
