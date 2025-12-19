#pragma once

// Include guard for extra ODR safety
#ifndef VRTIGO_GPU_POD_COMPLEX_ADAPTER_HPP
    #define VRTIGO_GPU_POD_COMPLEX_ADAPTER_HPP

    #include <complex>
    #include <span>
    #include <type_traits>
    #include <vector>

    #include <cstddef>
    #include <cstring>
    #include <vrtigo/gpu/complex_traits.hpp>

namespace vrtigo::gpu {

/**
 * Compile-time check for layout compatibility between POD complex and std::complex.
 *
 * Zero-copy reinterpret_cast is only safe when:
 *   - sizeof(PodT) == sizeof(std::complex<T>): Sizes match exactly
 *   - alignof(PodT) <= alignof(std::complex<T>): POD alignment not stricter
 *   - std::is_trivially_copyable_v<PodT>: Safe to memcpy
 *
 * IMPORTANT: The C++ standard does NOT guarantee std::complex<T> has {real, imag}
 * layout. It only guarantees array-style access via reinterpret_cast<T*>(&c)[0..1].
 * In practice, all major implementations use {real, imag} layout, but this is
 * implementation-defined. Always use is_layout_compatible_v to verify.
 *
 * Usage with static_assert for early failure:
 *
 *   static_assert(is_layout_compatible_v<Complex16, int16_t>,
 *       "Complex16 must be layout-compatible with std::complex<int16_t> for zero-copy");
 *
 * @tparam PodT POD complex type (must satisfy ComplexType concept)
 * @tparam T Component type of std::complex<T>
 */
template <typename PodT, typename T>
    requires ComplexType<PodT>
inline constexpr bool is_layout_compatible_v =
    sizeof(PodT) == sizeof(std::complex<T>) && alignof(PodT) <= alignof(std::complex<T>) &&
    std::is_trivially_copyable_v<PodT>;

// ============================================================================
// std::complex<T> -> POD complex (adapt_to_pod)
// ============================================================================

/**
 * Zero-copy adaptation from std::complex<T> span to POD complex span.
 *
 * Returns a reinterpreted view of the source data. No allocation, no copy.
 * Only available when layout compatibility is verified at compile time.
 *
 * @tparam PodT Target POD complex type (must satisfy ComplexType)
 * @tparam T Component type of source std::complex<T>
 * @param src Source span of std::complex<T>
 * @return Span of PodT viewing the same memory
 *
 * @note The returned span is valid only as long as src's underlying data exists.
 */
template <typename PodT, typename T>
    requires ComplexType<PodT> && is_layout_compatible_v<PodT, T>
std::span<PodT> adapt_to_pod(std::span<std::complex<T>> src) noexcept {
    return {reinterpret_cast<PodT*>(src.data()), src.size()};
}

/**
 * Zero-copy adaptation from const std::complex<T> span to const POD complex span.
 *
 * Const-correct version of adapt_to_pod for read-only access.
 *
 * @tparam PodT Target POD complex type (must satisfy ComplexType)
 * @tparam T Component type of source std::complex<T>
 * @param src Source span of const std::complex<T>
 * @return Span of const PodT viewing the same memory
 */
template <typename PodT, typename T>
    requires ComplexType<PodT> && is_layout_compatible_v<PodT, T>
std::span<const PodT> adapt_to_pod(std::span<const std::complex<T>> src) noexcept {
    return {reinterpret_cast<const PodT*>(src.data()), src.size()};
}

/**
 * Copy-based adaptation from std::complex<T> span to POD complex vector.
 *
 * Use when layout compatibility cannot be verified or when a separate
 * copy is desired. Returns a new vector with converted values.
 *
 * @tparam PodT Target POD complex type (must satisfy ComplexType)
 * @tparam T Component type of source std::complex<T>
 * @param src Source span of std::complex<T>
 * @return Vector of PodT with copied and converted values
 */
template <typename PodT, typename T>
    requires ComplexType<PodT> && (!is_layout_compatible_v<PodT, T>)
std::vector<PodT> adapt_to_pod_copy(std::span<const std::complex<T>> src) {
    std::vector<PodT> result(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        result[i] = complex_traits<PodT>::make(
            static_cast<typename complex_traits<PodT>::value_type>(src[i].real()),
            static_cast<typename complex_traits<PodT>::value_type>(src[i].imag()));
    }
    return result;
}

/**
 * Copy-based adaptation (explicit version for when layout IS compatible).
 *
 * Sometimes a copy is desired even when zero-copy is available
 * (e.g., to decouple lifetimes or for thread safety).
 *
 * @tparam PodT Target POD complex type (must satisfy ComplexType)
 * @tparam T Component type of source std::complex<T>
 * @param src Source span of std::complex<T>
 * @return Vector of PodT with copied values
 */
template <typename PodT, typename T>
    requires ComplexType<PodT> && is_layout_compatible_v<PodT, T>
std::vector<PodT> adapt_to_pod_copy(std::span<const std::complex<T>> src) {
    std::vector<PodT> result(src.size());
    // Layout compatible: can use memcpy for efficiency
    std::memcpy(result.data(), src.data(), src.size() * sizeof(PodT));
    return result;
}

// ============================================================================
// POD complex -> std::complex<T> (adapt_from_pod)
// ============================================================================

/**
 * Zero-copy adaptation from POD complex span to std::complex<T> span.
 *
 * Returns a reinterpreted view of the source data. No allocation, no copy.
 * Only available when layout compatibility is verified at compile time.
 *
 * @tparam T Component type of target std::complex<T>
 * @tparam PodT Source POD complex type (must satisfy ComplexType)
 * @param src Source span of PodT
 * @return Span of std::complex<T> viewing the same memory
 *
 * @note The returned span is valid only as long as src's underlying data exists.
 */
template <typename T, typename PodT>
    requires ComplexType<PodT> && is_layout_compatible_v<PodT, T>
std::span<std::complex<T>> adapt_from_pod(std::span<PodT> src) noexcept {
    return {reinterpret_cast<std::complex<T>*>(src.data()), src.size()};
}

/**
 * Zero-copy adaptation from const POD complex span to const std::complex<T> span.
 *
 * Const-correct version of adapt_from_pod for read-only access.
 *
 * @tparam T Component type of target std::complex<T>
 * @tparam PodT Source POD complex type (must satisfy ComplexType)
 * @param src Source span of const PodT
 * @return Span of const std::complex<T> viewing the same memory
 */
template <typename T, typename PodT>
    requires ComplexType<PodT> && is_layout_compatible_v<PodT, T>
std::span<const std::complex<T>> adapt_from_pod(std::span<const PodT> src) noexcept {
    return {reinterpret_cast<const std::complex<T>*>(src.data()), src.size()};
}

/**
 * Copy-based adaptation from POD complex span to std::complex<T> vector.
 *
 * Use when layout compatibility cannot be verified or when a separate
 * copy is desired. Returns a new vector with converted values.
 *
 * @tparam T Component type of target std::complex<T>
 * @tparam PodT Source POD complex type (must satisfy ComplexType)
 * @param src Source span of PodT
 * @return Vector of std::complex<T> with copied and converted values
 */
template <typename T, typename PodT>
    requires ComplexType<PodT> && (!is_layout_compatible_v<PodT, T>)
std::vector<std::complex<T>> adapt_from_pod_copy(std::span<const PodT> src) {
    std::vector<std::complex<T>> result(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        result[i] = std::complex<T>(static_cast<T>(complex_traits<PodT>::real(src[i])),
                                    static_cast<T>(complex_traits<PodT>::imag(src[i])));
    }
    return result;
}

/**
 * Copy-based adaptation (explicit version for when layout IS compatible).
 *
 * Sometimes a copy is desired even when zero-copy is available
 * (e.g., to decouple lifetimes or for thread safety).
 *
 * @tparam T Component type of target std::complex<T>
 * @tparam PodT Source POD complex type (must satisfy ComplexType)
 * @param src Source span of PodT
 * @return Vector of std::complex<T> with copied values
 */
template <typename T, typename PodT>
    requires ComplexType<PodT> && is_layout_compatible_v<PodT, T>
std::vector<std::complex<T>> adapt_from_pod_copy(std::span<const PodT> src) {
    std::vector<std::complex<T>> result(src.size());
    // Layout compatible: can use memcpy for efficiency
    std::memcpy(result.data(), src.data(), src.size() * sizeof(std::complex<T>));
    return result;
}

} // namespace vrtigo::gpu

#endif // VRTIGO_GPU_POD_COMPLEX_ADAPTER_HPP
