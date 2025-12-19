#pragma once

// =============================================================================
// ODR SAFETY WARNING
// =============================================================================
//
// This header extends vrtigo::detail::SampleTraits for POD complex types.
// Include order matters:
//
//   #include <vrtigo/gpu/sample_traits_ext.hpp>  // MUST come BEFORE sample_span.hpp
//   #include <vrtigo/sample_span.hpp>
//
// Including both core and extension headers in different orders across
// translation units WILL cause ODR violations and undefined behavior.
//
// CORRECT:
//   // file_a.cpp and file_b.cpp both do:
//   #include <vrtigo/gpu/sample_traits_ext.hpp>
//   #include <vrtigo/sample_span.hpp>
//
// INCORRECT (ODR violation):
//   // file_a.cpp:
//   #include <vrtigo/sample_span.hpp>  // No extension
//   // file_b.cpp:
//   #include <vrtigo/gpu/sample_traits_ext.hpp>
//   #include <vrtigo/sample_span.hpp>  // With extension
//
// Use consistently across your entire project.
// =============================================================================

#ifndef VRTIGO_GPU_SAMPLE_TRAITS_EXT_HPP
    #define VRTIGO_GPU_SAMPLE_TRAITS_EXT_HPP

    #include <cstddef>
    #include <vrtigo/detail/sample_traits.hpp>
    #include <vrtigo/gpu/complex.hpp>

namespace vrtigo::detail {

/**
 * SampleTraits extension for vrtigo::gpu::Complex<T> POD complex types.
 *
 * This specialization enables gpu::Complex<T> to work with SampleSpan
 * and SampleSpanView. The component_type is extracted from the
 * complex_traits specialization.
 *
 * Example: After including this header, SampleSpan<Complex16> just works:
 *
 *   using Complex16 = vrtigo::gpu::Complex<int16_t>;
 *   SampleSpan<Complex16> span(payload);
 *   span.set(0, Complex16{1, 2});
 *   Complex16 sample = span[0];
 *
 * Note: This specialization only covers gpu::Complex<T>. For other POD
 * complex types (e.g., cuFloatComplex), users must add their own
 * SampleTraits specialization.
 */
template <typename T>
struct SampleTraits<gpu::Complex<T>> {
    using component_type = T;
    static constexpr size_t component_size = sizeof(T);
    static constexpr size_t sample_size = component_size * 2;
};

} // namespace vrtigo::detail

namespace vrtigo::gpu {

/**
 * Free function accessor for real component.
 *
 * Provides uniform access to real component for any ComplexType.
 * Works with both POD complex types and (via specialization) std::complex.
 *
 * @param c Complex value
 * @return Real component
 */
template <ComplexType T>
auto real(const T& c) {
    return complex_traits<T>::real(c);
}

/**
 * Free function accessor for imaginary component.
 *
 * Provides uniform access to imaginary component for any ComplexType.
 * Works with both POD complex types and (via specialization) std::complex.
 *
 * @param c Complex value
 * @return Imaginary component
 */
template <ComplexType T>
auto imag(const T& c) {
    return complex_traits<T>::imag(c);
}

} // namespace vrtigo::gpu

#endif // VRTIGO_GPU_SAMPLE_TRAITS_EXT_HPP
