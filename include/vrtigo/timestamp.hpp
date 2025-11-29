#pragma once

#include "vrtigo/types.hpp"

#include <chrono>
#include <compare>
#include <optional>

#include <cstdint>
#include <ctime>

namespace vrtigo {

// Marker type for no timestamp
struct NoTimestamp {};

// Forward declaration for TimestampValue::as<>()
template <TsiType TSI, TsfType TSF>
class Timestamp;

// Forward declaration for TimestampValue friend
namespace detail {
class RuntimePacketBase;
}
namespace dynamic::detail {
class PacketBase;
}

/**
 * Type-erased timestamp value for runtime packet parsing
 *
 * TimestampValue is self-describing: it carries both the timestamp data
 * and format metadata (tsi_kind, tsf_kind). This enables uniform handling
 * of timestamps regardless of whether they came from compile-time or
 * runtime packets.
 *
 * Use as<TSI, TSF>() to narrow to a typed Timestamp<TSI, TSF> when you
 * need access to type-specific operations like to_chrono().
 */
class TimestampValue {
public:
    [[nodiscard]] constexpr uint32_t tsi() const noexcept { return tsi_; }
    [[nodiscard]] constexpr uint64_t tsf() const noexcept { return tsf_; }
    [[nodiscard]] constexpr TsiType tsi_kind() const noexcept { return tsi_kind_; }
    [[nodiscard]] constexpr TsfType tsf_kind() const noexcept { return tsf_kind_; }
    [[nodiscard]] constexpr bool has_tsi() const noexcept { return tsi_kind_ != TsiType::none; }
    [[nodiscard]] constexpr bool has_tsf() const noexcept { return tsf_kind_ != TsfType::none; }

    /**
     * Narrow to typed Timestamp if kinds match exactly
     *
     * @tparam TSI Expected integer timestamp type
     * @tparam TSF Expected fractional timestamp type
     * @return Typed timestamp if kinds match, nullopt otherwise
     */
    template <TsiType TSI, TsfType TSF>
    [[nodiscard]] constexpr std::optional<Timestamp<TSI, TSF>> as() const noexcept;

    constexpr auto operator<=>(const TimestampValue&) const noexcept = default;

private:
    // Construction only via Timestamp conversion or runtime packet classes
    template <TsiType, TsfType>
    friend class Timestamp;
    friend class detail::RuntimePacketBase;   // Base class for runtime packets (legacy)
    friend class dynamic::detail::PacketBase; // Base class for rt namespace packets

    constexpr TimestampValue(uint32_t tsi, uint64_t tsf, TsiType tsi_kind,
                             TsfType tsf_kind) noexcept
        : tsi_(tsi),
          tsf_(tsf),
          tsi_kind_(tsi_kind),
          tsf_kind_(tsf_kind) {}

    uint32_t tsi_{0};
    uint64_t tsf_{0};
    TsiType tsi_kind_{TsiType::none};
    TsfType tsf_kind_{TsfType::none};
};

// Primary template - works for ALL timestamp combinations
template <TsiType TSI, TsfType TSF>
class Timestamp {
    // Helper constant for readability and maintenance
    static constexpr bool is_utc_real_time = (TSI == TsiType::utc && TSF == TsfType::real_time);

public:
    // Constants (always available, compiler optimizes away if unused)
    static constexpr uint64_t PICOSECONDS_PER_SECOND = 1'000'000'000'000ULL;
    static constexpr uint64_t NANOSECONDS_PER_SECOND = 1'000'000'000ULL;
    static constexpr uint64_t PICOSECONDS_PER_NANOSECOND = 1'000ULL;
    static constexpr uint64_t MAX_FRACTIONAL = PICOSECONDS_PER_SECOND - 1;

    // Constructors - basic API available for all timestamp types
    constexpr Timestamp() noexcept = default;

    constexpr Timestamp(uint32_t sec, uint64_t frac) noexcept : seconds_(sec), fractional_(frac) {
        normalize();
    }

    // Accessors - basic API available for all types
    constexpr uint32_t tsi() const noexcept { return seconds_; }
    constexpr uint64_t tsf() const noexcept { return fractional_; }

    constexpr TsiType tsi_kind() const noexcept { return TSI; }
    constexpr TsfType tsf_kind() const noexcept { return TSF; }

    // Mutator - basic API available for all types
    constexpr void set(uint32_t tsi, uint64_t tsf) noexcept {
        seconds_ = tsi;
        fractional_ = tsf;
        normalize();
    }

    // Comparison operators - basic API available for all types
    // UTC-UTC and GPS-GPS comparisons OK.
    // Mixed-type comparisons not created yet.
    constexpr auto operator<=>(const Timestamp& other) const noexcept = default;

    // Conversion to type-erased TimestampValue
    constexpr operator TimestampValue() const noexcept {
        return TimestampValue{seconds_, fractional_, TSI, TSF};
    }

    // UTC-specific factory methods
    static Timestamp now() noexcept
        requires(is_utc_real_time)
    {
        auto now = std::chrono::system_clock::now();
        return from_chrono(now);
    }

    static constexpr Timestamp from_utc_seconds(uint32_t seconds) noexcept
        requires(is_utc_real_time)
    {
        return Timestamp(seconds, 0);
    }

    static Timestamp from_chrono(std::chrono::system_clock::time_point tp) noexcept
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

        return Timestamp(vrt_seconds, vrt_fractional);
    }

    // UTC-specific conversion methods
    std::chrono::system_clock::time_point to_chrono() const noexcept
        requires(is_utc_real_time)
    {
        // Convert seconds to duration
        auto sec_duration = std::chrono::seconds(seconds_);

        // Convert fractional to nanoseconds (losing sub-nanosecond precision)
        auto nano_duration = std::chrono::nanoseconds(fractional_ / PICOSECONDS_PER_NANOSECOND);

        // Create time_point from epoch
        return std::chrono::system_clock::time_point(sec_duration + nano_duration);
    }

    std::time_t to_time_t() const noexcept
        requires(is_utc_real_time)
    {
        return static_cast<std::time_t>(seconds_);
    }

    // UTC-specific arithmetic operations
    Timestamp& operator+=(std::chrono::nanoseconds duration) noexcept
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

    Timestamp& operator-=(std::chrono::nanoseconds duration) noexcept
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
    friend Timestamp operator+(Timestamp ts, std::chrono::nanoseconds duration) noexcept
        requires(is_utc_real_time)
    {
        ts += duration;
        return ts;
    }

    friend Timestamp operator-(Timestamp ts, std::chrono::nanoseconds duration) noexcept
        requires(is_utc_real_time)
    {
        ts -= duration;
        return ts;
    }

    friend std::chrono::nanoseconds operator-(const Timestamp& lhs, const Timestamp& rhs) noexcept
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

// Out-of-line definition of TimestampValue::as<>() - requires complete Timestamp type
template <TsiType TSI, TsfType TSF>
constexpr std::optional<Timestamp<TSI, TSF>> TimestampValue::as() const noexcept {
    if (tsi_kind_ != TSI || tsf_kind_ != TSF) {
        return std::nullopt;
    }
    return Timestamp<TSI, TSF>{tsi_, tsf_};
}

/**
 * Concept for types that provide timestamp accessors
 *
 * Satisfied by both Timestamp<TSI, TSF> and TimestampValue, enabling
 * generic code that works with either type.
 */
template <typename T>
concept TimestampLike = requires(const T& t) {
    { t.tsi() } -> std::convertible_to<uint32_t>;
    { t.tsf() } -> std::convertible_to<uint64_t>;
    { t.tsi_kind() } -> std::convertible_to<TsiType>;
    { t.tsf_kind() } -> std::convertible_to<TsfType>;
};

// Convenient type alias with TSF template parameter and default
template <TsfType TSF = TsfType::real_time>
using UtcTimestamp = Timestamp<TsiType::utc, TSF>;

// Most common case - UTC with real_time TSF
using UtcRealTimestamp = UtcTimestamp<>;

} // namespace vrtigo
