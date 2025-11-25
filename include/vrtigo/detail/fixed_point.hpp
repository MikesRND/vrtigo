// include/vrtigo/detail/fixed_point.hpp
#pragma once

#include <concepts>
#include <optional>
#include <type_traits>

#include <cmath>
#include <cstdint>

namespace vrtigo::detail {

/**
 * FixedPoint<IntBits, FracBits, Signed, StorageType>
 *
 * - Storage: Configurable word size (uint16_t, uint32_t, or uint64_t) containing a fixed-point
 * value.
 * - Layout:
 *   - total_bits = IntBits + FracBits (must fit within StorageType)
 *   - For Signed == true:
 *       * Two's complement with total_bits significant bits.
 *       * Range (in real values):
 *           [ -2^(IntBits - 1), 2^(IntBits - 1) - 2^(-FracBits) ]
 *   - For Signed == false:
 *       * Unsigned with total_bits significant bits.
 *       * Range (in real values):
 *           [ 0, 2^IntBits - 2^(-FracBits) ]
 *
 * Rounding rule for double -> fixed:
 *   - Round to nearest, ties to even (banker's rounding),
 *     same as the default IEEE 754 rounding mode.
 *
 * Conversion behavior:
 *   - to_double(raw): interpret `raw` as fixed-point (masked to total_bits)
 *                     and return a double.
 *   - from_double(v): saturating conversion from double to fixed-point.
 *                     Out-of-range values are clamped to representable range.
 *   - from_double_strict(v): like from_double but returns std::nullopt if
 *                            value is out of range instead of clamping.
 */
template <int IntBits, int FracBits, bool Signed = false, typename StorageType = std::uint64_t>
struct FixedPoint {
    static_assert(std::unsigned_integral<StorageType>,
                  "StorageType must be an unsigned integral type");
    static_assert(sizeof(StorageType) >= 2, "StorageType must be at least 16 bits");
    // Allow FracBits up to the full storage width.
    static_assert(FracBits >= 0 && FracBits <= static_cast<int>(sizeof(StorageType) * 8),
                  "Fractional bits must be in range [0, storage_bits]");
    static_assert(IntBits >= 0 && IntBits <= static_cast<int>(sizeof(StorageType) * 8),
                  "Integer bits must be in range [0, storage_bits]");
    static_assert(IntBits + FracBits >= 1, "Total bits must be at least 1");
    static_assert(IntBits + FracBits <= static_cast<int>(sizeof(StorageType) * 8),
                  "Total bits must not exceed storage size");
    static_assert(!Signed || IntBits >= 1,
                  "Signed formats require at least 1 integer bit (including sign)");

    using storage_type = StorageType;
    using signed_storage_type = std::make_signed_t<StorageType>;

    static constexpr int int_bits = IntBits;
    static constexpr int frac_bits = FracBits;
    static constexpr bool is_signed = Signed;
    static constexpr int total_bits = IntBits + FracBits;
    static constexpr int storage_bits = static_cast<int>(sizeof(StorageType) * 8);

private:
    // constexpr 2^n in long double, no integer shifts involved.
    static constexpr long double pow2_long_double(int n) noexcept {
        return (n == 0) ? 1.0L : 2.0L * pow2_long_double(n - 1);
    }

public:
    // 2^FracBits as long double for better intermediate precision.
    // Safe even when FracBits == storage_bits.
    static constexpr long double scale_ld = pow2_long_double(FracBits);

    // Mask with total_bits low bits set.
    static constexpr storage_type full_mask =
        (total_bits == storage_bits) ? ~storage_type{0} : (storage_type{1} << total_bits) - 1;

    // For signed formats: sign bit and sign-extend mask.
    static constexpr storage_type sign_bit_mask =
        Signed ? (storage_type{1} << (total_bits - 1)) : storage_type{0};

    static constexpr storage_type sign_extend_mask =
        (Signed && total_bits < storage_bits) ? (~storage_type{0} << total_bits) : storage_type{0};

    // ---------------------------------------------------------------------
    // Public API
    // ---------------------------------------------------------------------

    // Convert a raw fixed-point word to double.
    static double to_double(storage_type raw) noexcept {
        storage_type v = raw & full_mask;

        if constexpr (Signed) {
            // Sign extend if width < storage_bits and sign bit set.
            if (total_bits < storage_bits && (v & sign_bit_mask)) {
                v |= sign_extend_mask;
            }
            const auto s = static_cast<signed_storage_type>(
                v); // impl-defined for full-width negatives; accepted for our targets
            const long double x = static_cast<long double>(s);
            return static_cast<double>(x / scale_ld);
        } else {
            const long double x = static_cast<long double>(v);
            return static_cast<double>(x / scale_ld);
        }
    }

    // Saturating conversion from double to fixed-point.
    // Out-of-range values are clamped to representable range.
    static storage_type from_double(double value) noexcept {
        if (!std::isfinite(value)) {
            // Current contract: non-finite maps to 0 in saturating mode.
            return storage_type{0};
        }

        long double x = static_cast<long double>(value);

        if constexpr (Signed) {
            // Range: [ -2^(IntBits-1), 2^(IntBits-1) - 2^(-FracBits) ]
            const long double min_val = -std::ldexp(1.0L, IntBits - 1);
            const long double max_val = std::ldexp(1.0L, IntBits - 1) - 1.0L / scale_ld;

            if (x < min_val)
                x = min_val;
            if (x > max_val)
                x = max_val;
        } else {
            // Range: [ 0, 2^IntBits - 2^(-FracBits) ]
            const long double min_val = 0.0L;
            const long double max_val = std::ldexp(1.0L, IntBits) - 1.0L / scale_ld;

            if (x < min_val)
                x = min_val;
            if (x > max_val)
                x = max_val;
        }

        const long double scaled = x * scale_ld;
        const long double q = round_to_nearest_even_ld(scaled);

        if constexpr (Signed) {
            const auto as_int = static_cast<signed_storage_type>(q);
            return static_cast<storage_type>(as_int) & full_mask;
        } else {
            const auto as_uint = static_cast<storage_type>(q);
            return as_uint & full_mask;
        }
    }

    // Strict conversion: returns nullopt instead of saturating on overflow.
    [[nodiscard]]
    static std::optional<storage_type> from_double_strict(double value) noexcept {
        if (!std::isfinite(value)) {
            return std::nullopt;
        }

        long double x = static_cast<long double>(value);

        if constexpr (Signed) {
            const long double min_val = -std::ldexp(1.0L, IntBits - 1);
            const long double max_val = std::ldexp(1.0L, IntBits - 1) - 1.0L / scale_ld;

            if (x < min_val || x > max_val) {
                return std::nullopt;
            }
        } else {
            const long double min_val = 0.0L;
            const long double max_val = std::ldexp(1.0L, IntBits) - 1.0L / scale_ld;

            if (x < min_val || x > max_val) {
                return std::nullopt;
            }
        }

        const long double scaled = x * scale_ld;
        const long double q = round_to_nearest_even_ld(scaled);

        if constexpr (Signed) {
            const auto as_int = static_cast<signed_storage_type>(q);
            return static_cast<storage_type>(as_int) & full_mask;
        } else {
            const auto as_uint = static_cast<storage_type>(q);
            return as_uint & full_mask;
        }
    }

    // Minimum representable real value as double.
    static double min_value() noexcept {
        if constexpr (Signed) {
            const long double min_val = -std::ldexp(1.0L, IntBits - 1);
            return static_cast<double>(min_val);
        } else {
            return 0.0;
        }
    }

    // Maximum representable real value as double.
    static double max_value() noexcept {
        if constexpr (Signed) {
            const long double max_val = std::ldexp(1.0L, IntBits - 1) - 1.0L / scale_ld;
            return static_cast<double>(max_val);
        } else {
            const long double max_val = std::ldexp(1.0L, IntBits) - 1.0L / scale_ld;
            return static_cast<double>(max_val);
        }
    }

    // Mask and sanitize a raw storage value to the valid bit-width.
    static constexpr storage_type mask() noexcept { return full_mask; }

    static constexpr storage_type sanitize_raw(storage_type raw) noexcept {
        return raw & full_mask;
    }

private:
    // Round-to-nearest, ties-to-even, in long double.
    // This matches the default IEEE 754 rounding mode.
    static long double round_to_nearest_even_ld(long double v) noexcept {
        const long double floor_val = std::floor(v);
        const long double frac = v - floor_val; // in [0, 1)

        if (frac < 0.5L) {
            return floor_val;
        } else if (frac > 0.5L) {
            return floor_val + 1.0L;
        } else {
            // Exactly at .5, choose the even integer.
            if constexpr (Signed) {
                const auto i = static_cast<signed_storage_type>(floor_val);
                return (i & 1) == 0 ? floor_val : floor_val + 1.0L;
            } else {
                const auto i = static_cast<storage_type>(floor_val);
                return (i & storage_type{1}) == 0 ? floor_val : floor_val + 1.0L;
            }
        }
    }
};

} // namespace vrtigo::detail
