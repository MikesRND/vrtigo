// include/vrtigo/detail/time_math.hpp
#pragma once

#include <limits>
#include <utility>

#include <cstdint>

namespace vrtigo::detail {

/**
 * Centralized time arithmetic for Duration and Timestamp.
 *
 * Design rationale:
 * - Single source of truth for carry/borrow logic (no divergence between types)
 * - Uses int64_t intermediate for overflow detection before clamping
 * - Arithmetic is performed in wide integers, then clamped to storage range
 *
 * Overflow policy:
 * - All operations saturate to min/max on overflow (no exceptions, no expected<>)
 * - Callers needing overflow detection should use saturated() helper after operations
 * - With +/-68 year Duration range, overflow indicates programmer error
 */

/// Picoseconds per second (10^12)
inline constexpr uint64_t PICOS_PER_SEC = 1'000'000'000'000ULL;

/// Maximum valid picoseconds value (one less than a full second)
inline constexpr uint64_t MAX_PICOS = PICOS_PER_SEC - 1;

/**
 * Normalize (seconds, picoseconds) to canonical form with floor semantics.
 *
 * Handles:
 * - Picoseconds overflow (≥10^12) → carry to seconds
 * - Picoseconds underflow (<0) → borrow from seconds
 * - Floor semantics for negatives: -1.5s → {-2, 500e9}
 *
 * This is the single source of truth for all normalization logic.
 * Used by add_time, sub_time, mul_time, div_time, and Duration::from_picoseconds.
 *
 * @param sec Seconds (may need adjustment)
 * @param picos Picoseconds (may be out of range, including INT64_MIN)
 * @return Pair of (normalized_sec, normalized_picos) where picos ∈ [0, 10^12)
 */
constexpr auto normalize(int64_t sec, int64_t picos) noexcept -> std::pair<int64_t, uint64_t> {
    // Handle carry (picos >= 10^12)
    if (picos >= static_cast<int64_t>(PICOS_PER_SEC)) {
        int64_t carry = picos / static_cast<int64_t>(PICOS_PER_SEC);
        sec += carry;
        picos %= static_cast<int64_t>(PICOS_PER_SEC);
        return {sec, static_cast<uint64_t>(picos)};
    }

    // Handle borrow (picos < 0) - use unsigned math to avoid UB on INT64_MIN
    if (picos < 0) {
        // Convert to unsigned absolute value safely (0 - cast avoids -INT64_MIN UB)
        uint64_t abs_picos = 0ULL - static_cast<uint64_t>(picos);
        uint64_t borrow = (abs_picos - 1) / PICOS_PER_SEC + 1;
        sec -= static_cast<int64_t>(borrow);
        uint64_t result_picos = borrow * PICOS_PER_SEC - abs_picos;
        return {sec, result_picos};
    }

    // Already normalized
    return {sec, static_cast<uint64_t>(picos)};
}

/**
 * Add two time values represented as (seconds, picoseconds).
 *
 * Returns (seconds, picoseconds) with picoseconds normalized to [0, 10^12).
 * Seconds may overflow int64_t range - caller must clamp afterward.
 *
 * @param sec_a First value's seconds component
 * @param picos_a First value's picoseconds component [0, 10^12)
 * @param sec_b Second value's seconds component
 * @param picos_b Second value's picoseconds component [0, 10^12)
 * @return Pair of (seconds, picoseconds) - NOT clamped to storage range
 */
constexpr auto add_time(int64_t sec_a, uint64_t picos_a, int64_t sec_b,
                        uint64_t picos_b) noexcept -> std::pair<int64_t, uint64_t> {
    return normalize(sec_a + sec_b, static_cast<int64_t>(picos_a) + static_cast<int64_t>(picos_b));
}

/**
 * Subtract two time values: (sec_a, picos_a) - (sec_b, picos_b).
 *
 * Returns (seconds, picoseconds) with picoseconds normalized to [0, 10^12).
 * Seconds may underflow int64_t range - caller must clamp afterward.
 *
 * @param sec_a Minuend's seconds component
 * @param picos_a Minuend's picoseconds component [0, 10^12)
 * @param sec_b Subtrahend's seconds component
 * @param picos_b Subtrahend's picoseconds component [0, 10^12)
 * @return Pair of (seconds, picoseconds) - NOT clamped to storage range
 */
constexpr auto sub_time(int64_t sec_a, uint64_t picos_a, int64_t sec_b,
                        uint64_t picos_b) noexcept -> std::pair<int64_t, uint64_t> {
    return normalize(sec_a - sec_b, static_cast<int64_t>(picos_a) - static_cast<int64_t>(picos_b));
}

/**
 * Clamp seconds to int32_t range for Duration storage.
 *
 * On overflow/underflow, also sets picoseconds to boundary value.
 * This ensures Duration::max() and Duration::min() are well-defined sentinels.
 *
 * @param sec Seconds value (may exceed int32_t range)
 * @param picos Reference to picoseconds (modified on saturation)
 * @return Clamped seconds value in int32_t range
 */
constexpr int32_t clamp_to_duration(int64_t sec, uint64_t& picos) noexcept {
    if (sec > std::numeric_limits<int32_t>::max()) {
        picos = MAX_PICOS;
        return std::numeric_limits<int32_t>::max();
    }
    if (sec < std::numeric_limits<int32_t>::min()) {
        picos = 0;
        return std::numeric_limits<int32_t>::min();
    }
    return static_cast<int32_t>(sec);
}

/**
 * Clamp seconds to uint32_t range for Timestamp storage.
 *
 * On overflow/underflow, also sets picoseconds to boundary value.
 * This ensures max timestamp is a well-defined sentinel.
 *
 * @param sec Seconds value (may exceed uint32_t range or be negative)
 * @param picos Reference to picoseconds (modified on saturation)
 * @return Clamped seconds value in uint32_t range
 */
constexpr uint32_t clamp_to_timestamp(int64_t sec, uint64_t& picos) noexcept {
    if (sec > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        picos = MAX_PICOS;
        return std::numeric_limits<uint32_t>::max();
    }
    if (sec < 0) {
        picos = 0;
        return 0;
    }
    return static_cast<uint32_t>(sec);
}

/**
 * Multiply a time value by a scalar.
 *
 * Used for Duration * scalar operations.
 * Returns (seconds, picoseconds) - caller must clamp afterward.
 *
 * @param sec Seconds component
 * @param picos Picoseconds component [0, 10^12)
 * @param scalar Multiplier (can be negative)
 * @return Pair of (seconds, picoseconds) - may need clamping
 */
constexpr auto mul_time(int64_t sec, uint64_t picos,
                        int64_t scalar) noexcept -> std::pair<int64_t, uint64_t> {
    if (scalar == 0) {
        return {0, 0};
    }

    // Handle sign separately for cleaner arithmetic
    // Use 0 - cast to avoid UB when value == INT64_MIN (negation overflow)
    bool negative = (sec < 0) != (scalar < 0);
    uint64_t abs_sec = sec < 0 ? 0ULL - static_cast<uint64_t>(sec) : static_cast<uint64_t>(sec);
    uint64_t abs_scalar =
        scalar < 0 ? 0ULL - static_cast<uint64_t>(scalar) : static_cast<uint64_t>(scalar);

    // For negative durations in floor representation:
    // If sec < 0, the total value is (sec * PICOS + picos) where sec is negative
    // We need to handle this carefully

    // Convert to total picoseconds conceptually, but avoid overflow by working in parts
    // sec_result = abs_sec * abs_scalar + (picos * abs_scalar) / PICOS_PER_SEC
    // picos_result = (picos * abs_scalar) % PICOS_PER_SEC

    // Check for potential overflow in multiplication
    constexpr uint64_t MAX_SAFE_SEC = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());

    // Simple overflow check - if either factor is huge, saturate
    if (abs_sec > MAX_SAFE_SEC / abs_scalar && abs_scalar > 0) {
        // Overflow - saturate based on sign
        uint64_t sat_picos = negative ? 0 : MAX_PICOS;
        return {negative ? std::numeric_limits<int64_t>::min()
                         : std::numeric_limits<int64_t>::max(),
                sat_picos};
    }

    // Multiply picoseconds first
    __uint128_t picos_product = static_cast<__uint128_t>(picos) * abs_scalar;
    uint64_t extra_sec = static_cast<uint64_t>(picos_product / PICOS_PER_SEC);
    uint64_t result_picos = static_cast<uint64_t>(picos_product % PICOS_PER_SEC);

    // Multiply seconds
    uint64_t sec_product = abs_sec * abs_scalar + extra_sec;

    // Apply sign - use normalize() for floor semantics on negative results
    if (negative) {
        return normalize(-static_cast<int64_t>(sec_product), -static_cast<int64_t>(result_picos));
    } else {
        return {static_cast<int64_t>(sec_product), result_picos};
    }
}

/**
 * Divide a time value by a scalar.
 *
 * Used for Duration / scalar operations.
 * Returns (seconds, picoseconds) - already normalized.
 *
 * @param sec Seconds component
 * @param picos Picoseconds component [0, 10^12)
 * @param scalar Divisor (must not be zero)
 * @return Pair of (seconds, picoseconds)
 */
constexpr auto div_time(int64_t sec, uint64_t picos,
                        int64_t scalar) noexcept -> std::pair<int64_t, uint64_t> {
    if (scalar == 0) {
        // Division by zero - saturate based on sign of numerator
        if (sec > 0 || (sec == 0 && picos > 0)) {
            return {std::numeric_limits<int64_t>::max(), MAX_PICOS};
        } else if (sec < 0) {
            return {std::numeric_limits<int64_t>::min(), 0};
        } else {
            return {0, 0};
        }
    }

    // Convert to total picoseconds for division (may lose precision for huge values)
    // For values within ~106 days, this is exact
    bool negative = (sec < 0) != (scalar < 0);

    // Get absolute total picoseconds
    __int128_t total_picos;
    if (sec >= 0) {
        total_picos = static_cast<__int128_t>(sec) * static_cast<__int128_t>(PICOS_PER_SEC) +
                      static_cast<__int128_t>(picos);
    } else {
        // Floor representation: sec is negative, picos is positive offset
        // Total = sec * PICOS + picos (where sec < 0)
        total_picos = static_cast<__int128_t>(sec) * static_cast<__int128_t>(PICOS_PER_SEC) +
                      static_cast<__int128_t>(picos);
    }

    // Use 0 - cast to avoid UB when scalar == INT64_MIN
    uint64_t abs_scalar =
        scalar < 0 ? 0ULL - static_cast<uint64_t>(scalar) : static_cast<uint64_t>(scalar);
    __int128_t abs_total = total_picos < 0 ? -total_picos : total_picos;

    __int128_t quotient = abs_total / abs_scalar;

    // Convert back to seconds + picoseconds
    int64_t result_sec = static_cast<int64_t>(quotient / static_cast<__int128_t>(PICOS_PER_SEC));
    uint64_t result_picos =
        static_cast<uint64_t>(quotient % static_cast<__int128_t>(PICOS_PER_SEC));

    // Apply sign - use normalize() for floor semantics on negative results
    if (negative) {
        return normalize(-result_sec, -static_cast<int64_t>(result_picos));
    } else {
        return {result_sec, result_picos};
    }
}

} // namespace vrtigo::detail
