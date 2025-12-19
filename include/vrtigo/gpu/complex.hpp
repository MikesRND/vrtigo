#pragma once

// Include guard for extra ODR safety
#ifndef VRTIGO_GPU_COMPLEX_HPP
    #define VRTIGO_GPU_COMPLEX_HPP

    #include "vrtigo/gpu/complex_traits.hpp"

    #include <complex>

    #include <cstdint>

namespace vrtigo::gpu {

/**
 * POD complex number type for GPU-compatible code.
 *
 * std::complex<T> is not usable in CUDA device code because it has
 * non-trivial constructors and member functions not marked __device__.
 * This struct provides a simple, POD alternative that works on both
 * host and device.
 *
 * Features:
 *   - Trivially copyable (POD layout)
 *   - __host__ __device__ constructors and arithmetic operators
 *   - Host-only: implicit conversions to/from std::complex<T>
 *   - Satisfies ComplexType concept via complex_traits specialization
 *
 * Layout:
 *   - Two contiguous T values: real followed by imag
 *   - Compatible with array-style complex representation
 *
 * @tparam T Component scalar type (e.g., float, double, int16_t)
 */
template <typename T>
struct Complex {
    using value_type = T;

    // Data members (use re/im to avoid conflict with accessor methods)
    T re;
    T im;

    // Accessor methods for std::complex compatibility
    VRTIGO_HOST_DEVICE
    constexpr T real() const noexcept { return re; }

    VRTIGO_HOST_DEVICE
    constexpr T imag() const noexcept { return im; }

    // Default constructor - zero-initialized
    VRTIGO_HOST_DEVICE
    constexpr Complex() noexcept : re{}, im{} {}

    // Construct from real and imaginary parts
    VRTIGO_HOST_DEVICE
    constexpr Complex(T r, T i) noexcept : re{r}, im{i} {}

    // Construct from real part only (imaginary = 0)
    VRTIGO_HOST_DEVICE
    constexpr explicit Complex(T r) noexcept : re{r}, im{} {}

    // Copy and move are defaulted (trivially copyable)
    Complex(const Complex&) = default;
    Complex(Complex&&) = default;
    Complex& operator=(const Complex&) = default;
    Complex& operator=(Complex&&) = default;

    // Host-only: implicit conversion from std::complex<T>
    // Note: These are always defined to maintain ODR compliance across
    // host/device compilation passes. They're callable from host code only.
    VRTIGO_HOST
    constexpr Complex(const std::complex<T>& c) noexcept : re{c.real()}, im{c.imag()} {}

    // Host-only: implicit conversion to std::complex<T>
    VRTIGO_HOST
    constexpr operator std::complex<T>() const noexcept { return std::complex<T>{re, im}; }

    // Arithmetic operators

    VRTIGO_HOST_DEVICE
    constexpr Complex operator+(const Complex& rhs) const noexcept {
        return Complex{static_cast<T>(re + rhs.re), static_cast<T>(im + rhs.im)};
    }

    VRTIGO_HOST_DEVICE
    constexpr Complex operator-(const Complex& rhs) const noexcept {
        return Complex{static_cast<T>(re - rhs.re), static_cast<T>(im - rhs.im)};
    }

    VRTIGO_HOST_DEVICE
    constexpr Complex operator*(const Complex& rhs) const noexcept {
        // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        return Complex{static_cast<T>(re * rhs.re - im * rhs.im),
                       static_cast<T>(re * rhs.im + im * rhs.re)};
    }

    VRTIGO_HOST_DEVICE
    constexpr Complex operator/(const Complex& rhs) const noexcept {
        // (a + bi)/(c + di) = ((ac + bd) + (bc - ad)i) / (c^2 + d^2)
        T denom = static_cast<T>(rhs.re * rhs.re + rhs.im * rhs.im);
        return Complex{static_cast<T>((re * rhs.re + im * rhs.im) / denom),
                       static_cast<T>((im * rhs.re - re * rhs.im) / denom)};
    }

    // Compound assignment operators

    VRTIGO_HOST_DEVICE
    constexpr Complex& operator+=(const Complex& rhs) noexcept {
        re = static_cast<T>(re + rhs.re);
        im = static_cast<T>(im + rhs.im);
        return *this;
    }

    VRTIGO_HOST_DEVICE
    constexpr Complex& operator-=(const Complex& rhs) noexcept {
        re = static_cast<T>(re - rhs.re);
        im = static_cast<T>(im - rhs.im);
        return *this;
    }

    VRTIGO_HOST_DEVICE
    constexpr Complex& operator*=(const Complex& rhs) noexcept {
        *this = *this * rhs;
        return *this;
    }

    VRTIGO_HOST_DEVICE
    constexpr Complex& operator/=(const Complex& rhs) noexcept {
        *this = *this / rhs;
        return *this;
    }

    // Unary operators

    VRTIGO_HOST_DEVICE
    constexpr Complex operator-() const noexcept {
        return Complex{static_cast<T>(-re), static_cast<T>(-im)};
    }

    VRTIGO_HOST_DEVICE
    constexpr Complex operator+() const noexcept { return *this; }

    // Comparison operators

    VRTIGO_HOST_DEVICE
    constexpr bool operator==(const Complex& rhs) const noexcept {
        return re == rhs.re && im == rhs.im;
    }

    VRTIGO_HOST_DEVICE
    constexpr bool operator!=(const Complex& rhs) const noexcept { return !(*this == rhs); }
};

// Scalar multiplication (scalar * complex)
template <typename T>
VRTIGO_HOST_DEVICE constexpr Complex<T> operator*(T lhs, const Complex<T>& rhs) noexcept {
    return Complex<T>{static_cast<T>(lhs * rhs.re), static_cast<T>(lhs * rhs.im)};
}

// Scalar multiplication (complex * scalar)
template <typename T>
VRTIGO_HOST_DEVICE constexpr Complex<T> operator*(const Complex<T>& lhs, T rhs) noexcept {
    return Complex<T>{static_cast<T>(lhs.re * rhs), static_cast<T>(lhs.im * rhs)};
}

// Scalar division (complex / scalar)
template <typename T>
VRTIGO_HOST_DEVICE constexpr Complex<T> operator/(const Complex<T>& lhs, T rhs) noexcept {
    return Complex<T>{static_cast<T>(lhs.re / rhs), static_cast<T>(lhs.im / rhs)};
}

/**
 * Type aliases for common VRT-supported complex sample types.
 *
 * These correspond to the complex variants of VRT sample formats:
 *   Complex8  - 8-bit signed integer I/Q (2 bytes per sample)
 *   Complex16 - 16-bit signed integer I/Q (4 bytes per sample)
 *   Complex32 - 32-bit signed integer I/Q (8 bytes per sample)
 *   ComplexF  - 32-bit float I/Q (8 bytes per sample)
 *   ComplexD  - 64-bit double I/Q (16 bytes per sample)
 */
using Complex8 = Complex<int8_t>;
using Complex16 = Complex<int16_t>;
using Complex32 = Complex<int32_t>;
using ComplexF = Complex<float>;
using ComplexD = Complex<double>;

} // namespace vrtigo::gpu

#endif // VRTIGO_GPU_COMPLEX_HPP
