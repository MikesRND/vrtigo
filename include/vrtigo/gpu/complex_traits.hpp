#pragma once

// Include guard for extra ODR safety
#ifndef VRTIGO_GPU_COMPLEX_TRAITS_HPP
    #define VRTIGO_GPU_COMPLEX_TRAITS_HPP

    #include "vrtigo/gpu/detail/cuda_macros.hpp"

    #include <concepts>
    #include <type_traits>

namespace vrtigo::gpu {

// Forward declaration of Complex<T> for built-in trait specialization
template <typename T>
struct Complex;

/**
 * Primary complex_traits template - specialize for your POD complex types.
 *
 * GPU complex types (cuFloatComplex, thrust::complex<T>, etc.) don't work
 * with std::complex<T> on device. This trait system provides a uniform
 * interface for extracting real/imag components and constructing complex
 * values.
 *
 * Users must specialize this template for their complex types:
 *
 *   template<> struct vrtigo::gpu::complex_traits<cuFloatComplex> {
 *       using value_type = float;
 *       VRTIGO_HOST_DEVICE static float real(const cuFloatComplex& c) { return c.x; }
 *       VRTIGO_HOST_DEVICE static float imag(const cuFloatComplex& c) { return c.y; }
 *       VRTIGO_HOST_DEVICE static cuFloatComplex make(float r, float i) { return {r, i}; }
 *   };
 *
 *   template<typename T>
 *   struct vrtigo::gpu::complex_traits<thrust::complex<T>> {
 *       using value_type = T;
 *       VRTIGO_HOST_DEVICE static T real(const thrust::complex<T>& c) { return c.real(); }
 *       VRTIGO_HOST_DEVICE static T imag(const thrust::complex<T>& c) { return c.imag(); }
 *       VRTIGO_HOST_DEVICE static thrust::complex<T> make(T r, T i) { return {r, i}; }
 *   };
 */
template <typename T>
struct complex_traits {
    // Primary template is empty - only specializations are valid.
    // Must provide:
    //   using value_type = ...;
    //   static value_type real(const T&);
    //   static value_type imag(const T&);
    //   static T make(value_type r, value_type i);
};

/**
 * Built-in specialization for vrtigo::gpu::Complex<T>.
 *
 * This is the default POD complex type provided by vrtigo. Users can use
 * their own POD complex types by specializing complex_traits.
 */
template <typename T>
struct complex_traits<Complex<T>> {
    using value_type = T;

    VRTIGO_HOST_DEVICE
    static constexpr T real(const Complex<T>& c) noexcept { return c.re; }

    VRTIGO_HOST_DEVICE
    static constexpr T imag(const Complex<T>& c) noexcept { return c.im; }

    VRTIGO_HOST_DEVICE
    static constexpr Complex<T> make(T r, T i) noexcept { return Complex<T>{r, i}; }
};

/**
 * Concept for types with valid complex_traits specialization.
 *
 * A type satisfies ComplexType if complex_traits<T> provides:
 *   - value_type typedef
 *   - real(const T&) returning convertible to value_type
 *   - imag(const T&) returning convertible to value_type
 *   - make(value_type, value_type) returning T
 */
template <typename T>
concept ComplexType = requires(const T t, typename complex_traits<T>::value_type v) {
    typename complex_traits<T>::value_type;
    { complex_traits<T>::real(t) } -> std::convertible_to<typename complex_traits<T>::value_type>;
    { complex_traits<T>::imag(t) } -> std::convertible_to<typename complex_traits<T>::value_type>;
    { complex_traits<T>::make(v, v) } -> std::same_as<T>;
};

/**
 * Concept for GPU-compatible sample types.
 *
 * Includes all scalar types supported by VRT plus any ComplexType.
 * Used by raw_sample_io.hpp and sample_span_device.hpp.
 */
template <typename T>
concept GpuSampleType =
    std::same_as<T, int8_t> || std::same_as<T, int16_t> || std::same_as<T, int32_t> ||
    std::same_as<T, float> || std::same_as<T, double> || ComplexType<T>;

} // namespace vrtigo::gpu

#endif // VRTIGO_GPU_COMPLEX_TRAITS_HPP
