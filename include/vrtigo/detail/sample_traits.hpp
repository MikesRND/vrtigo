#pragma once

#include <complex>
#include <concepts>

#include <cstddef>
#include <cstdint>

namespace vrtigo::detail {

/**
 * Sample type traits for VRT payload interpretation.
 *
 * Provides compile-time size information for supported sample types.
 * Primary template is empty (invalid) - only specializations are valid.
 *
 * Supported types:
 *   - int8_t, int16_t, int32_t (signed integers)
 *   - float, double (IEEE 754)
 *   - std::complex<T> for any of the above
 *
 * NOT supported (will fail ValidSampleType concept):
 *   - Unsigned integers
 *   - VRT bit-packed formats
 *   - IEEE half-precision
 *   - Fixed-point with fractional bits
 */
template <typename T>
struct SampleTraits {};

// Scalar specializations

template <>
struct SampleTraits<int8_t> {
    static constexpr size_t component_size = 1;
    static constexpr size_t sample_size = 1;
};

template <>
struct SampleTraits<int16_t> {
    static constexpr size_t component_size = 2;
    static constexpr size_t sample_size = 2;
};

template <>
struct SampleTraits<int32_t> {
    static constexpr size_t component_size = 4;
    static constexpr size_t sample_size = 4;
};

template <>
struct SampleTraits<float> {
    static constexpr size_t component_size = 4;
    static constexpr size_t sample_size = 4;
};

template <>
struct SampleTraits<double> {
    static constexpr size_t component_size = 8;
    static constexpr size_t sample_size = 8;
};

// Complex specialization - reuses component traits
template <typename T>
    requires requires { SampleTraits<T>::component_size; }
struct SampleTraits<std::complex<T>> {
    static constexpr size_t component_size = SampleTraits<T>::component_size;
    static constexpr size_t sample_size = component_size * 2;
};

} // namespace vrtigo::detail

namespace vrtigo {

/**
 * Concept for valid sample types.
 *
 * Satisfied by types with valid SampleTraits specialization:
 *   - int8_t, int16_t, int32_t
 *   - float, double
 *   - std::complex<T> for any of the above
 */
template <typename T>
concept ValidSampleType = requires {
    { detail::SampleTraits<T>::component_size } -> std::convertible_to<size_t>;
    { detail::SampleTraits<T>::sample_size } -> std::convertible_to<size_t>;
};

} // namespace vrtigo
