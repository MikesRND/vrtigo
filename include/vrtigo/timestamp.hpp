#pragma once

#include "vrtigo/detail/time_math.hpp"
#include "vrtigo/duration.hpp"
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
namespace dynamic::detail {
class PacketViewBase;
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
    friend class dynamic::detail::PacketViewBase; // Base class for dynamic packet views

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

/**
 * Typed timestamp with compile-time TSI/TSF specification.
 *
 * ## Storage
 * 12 bytes: uint32_t seconds + uint64_t fractional (picoseconds).
 * Matches VITA 49 wire format exactly.
 *
 * ## Overflow Policy
 * All arithmetic with Duration saturates on overflow/underflow:
 * - Addition saturates to max timestamp (UINT32_MAX, MAX_FRACTIONAL)
 * - Subtraction saturates to zero timestamp (0, 0)
 * - Use `saturated(timestamp)` helper to detect overflow when needed
 *
 * **Rationale:** Same as Duration - with large range, overflow indicates
 * programmer error. No `expected<>` return types, no exceptions.
 *
 * ## Breaking Changes from Previous API
 * - Removed: Result type alias, add_checked(), sub_checked(), diff_checked()
 * - Duration arithmetic now uses new 12-byte Duration with ±68 year range
 */
template <TsiType TSI, TsfType TSF>
class Timestamp {
    // Helper constant for readability and maintenance
    static constexpr bool is_utc_real_time = (TSI == TsiType::utc && TSF == TsfType::real_time);

public:
    // Constants (always available, compiler optimizes away if unused)
    static constexpr uint64_t PICOSECONDS_PER_SECOND = detail::PICOS_PER_SEC;
    static constexpr uint64_t NANOSECONDS_PER_SECOND = 1'000'000'000ULL;
    static constexpr uint64_t PICOSECONDS_PER_NANOSECOND = 1'000ULL;
    static constexpr uint64_t MAX_FRACTIONAL = detail::MAX_PICOS;

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

    // Arithmetic with Duration (saturates on overflow/underflow)
    Timestamp& operator+=(Duration d) noexcept
        requires(TSF == TsfType::real_time)
    {
        auto [sec, picos] = detail::add_time(static_cast<int64_t>(seconds_), fractional_,
                                             static_cast<int64_t>(d.seconds()), d.picoseconds());
        seconds_ = detail::clamp_to_timestamp(sec, picos);
        fractional_ = picos;
        return *this;
    }

    Timestamp& operator-=(Duration d) noexcept
        requires(TSF == TsfType::real_time)
    {
        auto [sec, picos] = detail::sub_time(static_cast<int64_t>(seconds_), fractional_,
                                             static_cast<int64_t>(d.seconds()), d.picoseconds());
        seconds_ = detail::clamp_to_timestamp(sec, picos);
        fractional_ = picos;
        return *this;
    }

    // Friend operators with Duration
    friend Timestamp operator+(Timestamp ts, Duration d) noexcept
        requires(TSF == TsfType::real_time)
    {
        ts += d;
        return ts;
    }

    friend Timestamp operator-(Timestamp ts, Duration d) noexcept
        requires(TSF == TsfType::real_time)
    {
        ts -= d;
        return ts;
    }

    // ShortDuration arithmetic - forwards to Duration overloads
    Timestamp& operator+=(ShortDuration d) noexcept
        requires(TSF == TsfType::real_time)
    {
        return *this += d.to_duration();
    }

    Timestamp& operator-=(ShortDuration d) noexcept
        requires(TSF == TsfType::real_time)
    {
        return *this -= d.to_duration();
    }

    friend Timestamp operator+(Timestamp ts, ShortDuration d) noexcept
        requires(TSF == TsfType::real_time)
    {
        ts += d;
        return ts;
    }

    friend Timestamp operator-(Timestamp ts, ShortDuration d) noexcept
        requires(TSF == TsfType::real_time)
    {
        ts -= d;
        return ts;
    }

    // Convenience helpers for common offset operations

    /// Offset timestamp by ShortDuration - more readable than operator+
    Timestamp offset(ShortDuration d) const noexcept
        requires(TSF == TsfType::real_time)
    {
        return *this + d;
    }

    /// Offset timestamp by sample count - combines from_samples + offset
    Timestamp offset_samples(int64_t count, SamplePeriod period) const noexcept
        requires(TSF == TsfType::real_time)
    {
        return *this + ShortDuration::from_samples(count, period);
    }

    // Timestamp difference returns Duration
    // With new ±68 year Duration range, saturation is rare
    friend Duration operator-(const Timestamp& lhs, const Timestamp& rhs) noexcept
        requires(TSF == TsfType::real_time)
    {
        auto [sec, picos] = detail::sub_time(static_cast<int64_t>(lhs.seconds_), lhs.fractional_,
                                             static_cast<int64_t>(rhs.seconds_), rhs.fractional_);

        // Clamp to Duration range (±68 years)
        uint64_t result_picos = picos;
        int32_t result_sec = detail::clamp_to_duration(sec, result_picos);

        // Construct Duration from seconds + picoseconds
        return Duration::from_seconds(static_cast<int64_t>(result_sec)) +
               Duration::from_picoseconds(static_cast<int64_t>(result_picos));
    }

private:
    uint32_t seconds_{0};    // TSI component - default initialized to zero
    uint64_t fractional_{0}; // TSF component - default initialized to zero

    // TSF-aware normalization using centralized detail::normalize()
    // Same path as arithmetic operators - ensures consistent behavior and shared bug fixes
    constexpr void normalize() noexcept {
        if constexpr (TSF == TsfType::real_time) {
            // Reuse centralized normalize (handles carry) + clamp_to_timestamp
            // Cast to signed for normalize, then clamp back to uint32_t
            // Note: fractional_ should be < 2^63 for any valid picoseconds input
            auto [sec, picos] = detail::normalize(static_cast<int64_t>(seconds_),
                                                  static_cast<int64_t>(fractional_));
            seconds_ = detail::clamp_to_timestamp(sec, picos);
            fractional_ = picos;
        }
        // No normalization for sample_count, free_running, none
    }
};

/**
 * Check if a Timestamp has saturated to max value.
 *
 * Use this after arithmetic operations to detect overflow:
 * ```cpp
 * auto ts = clock.tick();
 * if (saturated(ts)) {
 *     // Handle overflow - timestamp is at maximum
 * }
 * ```
 *
 * Note: This only checks for max (overflow). Underflow saturates to zero,
 * which is also a valid timestamp. For SampleClock, compare against previous
 * timestamp to detect stuck-at-zero condition.
 */
template <TsiType TSI, TsfType TSF>
constexpr bool saturated(const Timestamp<TSI, TSF>& ts) noexcept {
    return ts.tsi() == std::numeric_limits<uint32_t>::max() &&
           ts.tsf() == Timestamp<TSI, TSF>::MAX_FRACTIONAL;
}

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
