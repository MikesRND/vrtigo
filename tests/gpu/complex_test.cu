/**
 * @file complex_test.cu
 * @brief Unit tests for vrtigo GPU complex types
 *
 * Tests Complex<T> construction, arithmetic, and complex_traits access.
 * Tests host/device conversions and POD complex adapter functionality.
 *
 * Requires: CUDA runtime, Google Test
 */

// VRTIGO_ENABLE_CUDA should be defined via compiler flag (-DVRTIGO_ENABLE_CUDA)
#ifndef VRTIGO_ENABLE_CUDA
#define VRTIGO_ENABLE_CUDA
#endif

#include <vrtigo/vrtigo_gpu.hpp>

#include <gtest/gtest.h>

#include <complex>
#include <cmath>
#include <vector>

using namespace vrtigo::gpu;

// ============================================================================
// Host-side Complex<T> tests
// ============================================================================

class ComplexHostTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset CUDA device state
        cudaDeviceSynchronize();
    }
};

TEST_F(ComplexHostTest, DefaultConstruction) {
    Complex16 c;
    EXPECT_EQ(c.re, 0);
    EXPECT_EQ(c.im, 0);

    ComplexF cf;
    EXPECT_FLOAT_EQ(cf.re, 0.0f);
    EXPECT_FLOAT_EQ(cf.im, 0.0f);
}

TEST_F(ComplexHostTest, ValueConstruction) {
    Complex16 c(10, 20);
    EXPECT_EQ(c.re, 10);
    EXPECT_EQ(c.im, 20);

    ComplexF cf(1.5f, 2.5f);
    EXPECT_FLOAT_EQ(cf.re, 1.5f);
    EXPECT_FLOAT_EQ(cf.im, 2.5f);
}

TEST_F(ComplexHostTest, RealOnlyConstruction) {
    Complex16 c(42);
    EXPECT_EQ(c.re, 42);
    EXPECT_EQ(c.im, 0);
}

TEST_F(ComplexHostTest, StdComplexConversion) {
    // POD to std::complex
    Complex16 pod(10, 20);
    std::complex<int16_t> std_c = pod;
    EXPECT_EQ(std_c.real(), 10);
    EXPECT_EQ(std_c.imag(), 20);

    // std::complex to POD
    std::complex<int16_t> std_c2(30, 40);
    Complex16 pod2 = std_c2;
    EXPECT_EQ(pod2.re, 30);
    EXPECT_EQ(pod2.im, 40);

    // Float version
    ComplexF pod_f(1.5f, 2.5f);
    std::complex<float> std_cf = pod_f;
    EXPECT_FLOAT_EQ(std_cf.real(), 1.5f);
    EXPECT_FLOAT_EQ(std_cf.imag(), 2.5f);
}

TEST_F(ComplexHostTest, Addition) {
    Complex16 a(10, 20);
    Complex16 b(3, 4);
    Complex16 result = a + b;

    EXPECT_EQ(result.re, 13);
    EXPECT_EQ(result.im, 24);
}

TEST_F(ComplexHostTest, Subtraction) {
    Complex16 a(10, 20);
    Complex16 b(3, 4);
    Complex16 result = a - b;

    EXPECT_EQ(result.re, 7);
    EXPECT_EQ(result.im, 16);
}

TEST_F(ComplexHostTest, Multiplication) {
    ComplexF a(3.0f, 4.0f);
    ComplexF b(2.0f, 1.0f);
    // (3 + 4i)(2 + i) = 6 + 3i + 8i + 4i^2 = 6 + 11i - 4 = 2 + 11i
    ComplexF result = a * b;

    EXPECT_FLOAT_EQ(result.re, 2.0f);
    EXPECT_FLOAT_EQ(result.im, 11.0f);
}

TEST_F(ComplexHostTest, Division) {
    ComplexF a(3.0f, 4.0f);
    ComplexF b(2.0f, 1.0f);
    // (3 + 4i)/(2 + i) = (3 + 4i)(2 - i)/(2^2 + 1^2)
    //                  = (6 - 3i + 8i - 4i^2)/5
    //                  = (6 + 5i + 4)/5 = (10 + 5i)/5 = 2 + i
    ComplexF result = a / b;

    EXPECT_FLOAT_EQ(result.re, 2.0f);
    EXPECT_FLOAT_EQ(result.im, 1.0f);
}

TEST_F(ComplexHostTest, CompoundAssignment) {
    ComplexF c(1.0f, 2.0f);

    c += ComplexF(3.0f, 4.0f);
    EXPECT_FLOAT_EQ(c.re, 4.0f);
    EXPECT_FLOAT_EQ(c.im, 6.0f);

    c -= ComplexF(1.0f, 1.0f);
    EXPECT_FLOAT_EQ(c.re, 3.0f);
    EXPECT_FLOAT_EQ(c.im, 5.0f);

    c *= ComplexF(2.0f, 0.0f);
    EXPECT_FLOAT_EQ(c.re, 6.0f);
    EXPECT_FLOAT_EQ(c.im, 10.0f);
}

TEST_F(ComplexHostTest, UnaryOperators) {
    Complex16 c(10, 20);

    Complex16 neg = -c;
    EXPECT_EQ(neg.re, -10);
    EXPECT_EQ(neg.im, -20);

    Complex16 pos = +c;
    EXPECT_EQ(pos.re, 10);
    EXPECT_EQ(pos.im, 20);
}

TEST_F(ComplexHostTest, Comparison) {
    Complex16 a(10, 20);
    Complex16 b(10, 20);
    Complex16 c(10, 21);

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
    EXPECT_FALSE(a != b);
    EXPECT_TRUE(a != c);
}

TEST_F(ComplexHostTest, ScalarMultiplication) {
    ComplexF c(3.0f, 4.0f);

    ComplexF r1 = c * 2.0f;
    EXPECT_FLOAT_EQ(r1.re, 6.0f);
    EXPECT_FLOAT_EQ(r1.im, 8.0f);

    ComplexF r2 = 2.0f * c;
    EXPECT_FLOAT_EQ(r2.re, 6.0f);
    EXPECT_FLOAT_EQ(r2.im, 8.0f);

    ComplexF r3 = c / 2.0f;
    EXPECT_FLOAT_EQ(r3.re, 1.5f);
    EXPECT_FLOAT_EQ(r3.im, 2.0f);
}

// ============================================================================
// complex_traits tests
// ============================================================================

TEST_F(ComplexHostTest, ComplexTraitsAccess) {
    Complex16 c(42, 84);

    EXPECT_EQ(complex_traits<Complex16>::real(c), 42);
    EXPECT_EQ(complex_traits<Complex16>::imag(c), 84);

    Complex16 made = complex_traits<Complex16>::make(100, 200);
    EXPECT_EQ(made.re, 100);
    EXPECT_EQ(made.im, 200);
}

TEST_F(ComplexHostTest, FreeFunctionAccessors) {
    Complex16 c(42, 84);

    EXPECT_EQ(real(c), 42);
    EXPECT_EQ(imag(c), 84);
}

TEST_F(ComplexHostTest, ComplexTypeConcept) {
    // Static assertions already in compile_check.cu, but verify at runtime
    EXPECT_TRUE((ComplexType<Complex8>));
    EXPECT_TRUE((ComplexType<Complex16>));
    EXPECT_TRUE((ComplexType<Complex32>));
    EXPECT_TRUE((ComplexType<ComplexF>));
    EXPECT_TRUE((ComplexType<ComplexD>));
}

TEST_F(ComplexHostTest, GpuSampleTypeConcept) {
    // Scalars
    EXPECT_TRUE((GpuSampleType<int8_t>));
    EXPECT_TRUE((GpuSampleType<int16_t>));
    EXPECT_TRUE((GpuSampleType<int32_t>));
    EXPECT_TRUE((GpuSampleType<float>));
    EXPECT_TRUE((GpuSampleType<double>));

    // Complex
    EXPECT_TRUE((GpuSampleType<Complex16>));
    EXPECT_TRUE((GpuSampleType<ComplexF>));
}

// ============================================================================
// POD complex adapter tests
// ============================================================================

TEST_F(ComplexHostTest, LayoutCompatibilityCheck) {
    // These should be layout-compatible
    EXPECT_TRUE((is_layout_compatible_v<Complex16, int16_t>));
    EXPECT_TRUE((is_layout_compatible_v<Complex32, int32_t>));
    EXPECT_TRUE((is_layout_compatible_v<ComplexF, float>));
    EXPECT_TRUE((is_layout_compatible_v<ComplexD, double>));

    // Verify sizes match
    EXPECT_EQ(sizeof(Complex16), sizeof(std::complex<int16_t>));
    EXPECT_EQ(sizeof(ComplexF), sizeof(std::complex<float>));
}

TEST_F(ComplexHostTest, AdaptToPodZeroCopy) {
    std::complex<int16_t> std_arr[4] = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
    std::span<std::complex<int16_t>> std_span(std_arr, 4);

    auto pod_span = adapt_to_pod<Complex16>(std_span);

    // Should be same size
    EXPECT_EQ(pod_span.size(), 4u);

    // Should point to same memory (zero-copy)
    EXPECT_EQ(reinterpret_cast<void*>(pod_span.data()),
              reinterpret_cast<void*>(std_span.data()));

    // Values should match
    EXPECT_EQ(pod_span[0].re, 1);
    EXPECT_EQ(pod_span[0].im, 2);
    EXPECT_EQ(pod_span[3].re, 7);
    EXPECT_EQ(pod_span[3].im, 8);
}

TEST_F(ComplexHostTest, AdaptFromPodZeroCopy) {
    Complex16 pod_arr[4] = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
    std::span<Complex16> pod_span(pod_arr, 4);

    auto std_span = adapt_from_pod<int16_t>(pod_span);

    // Should be same size
    EXPECT_EQ(std_span.size(), 4u);

    // Should point to same memory (zero-copy)
    EXPECT_EQ(reinterpret_cast<void*>(std_span.data()),
              reinterpret_cast<void*>(pod_span.data()));

    // Values should match
    EXPECT_EQ(std_span[0].real(), 1);
    EXPECT_EQ(std_span[0].imag(), 2);
}

TEST_F(ComplexHostTest, AdaptToPodCopy) {
    std::complex<int16_t> std_arr[4] = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
    std::span<const std::complex<int16_t>> std_span(std_arr, 4);

    auto pod_vec = adapt_to_pod_copy<Complex16>(std_span);

    // Should be a copy (different memory)
    EXPECT_NE(reinterpret_cast<void*>(pod_vec.data()),
              reinterpret_cast<const void*>(std_span.data()));

    // Values should match
    EXPECT_EQ(pod_vec.size(), 4u);
    EXPECT_EQ(pod_vec[0].re, 1);
    EXPECT_EQ(pod_vec[0].im, 2);
    EXPECT_EQ(pod_vec[3].re, 7);
    EXPECT_EQ(pod_vec[3].im, 8);
}

TEST_F(ComplexHostTest, AdaptFromPodCopy) {
    Complex16 pod_arr[4] = {{10, 20}, {30, 40}, {50, 60}, {70, 80}};
    std::span<const Complex16> pod_span(pod_arr, 4);

    auto std_vec = adapt_from_pod_copy<int16_t>(pod_span);

    // Should be a copy
    EXPECT_NE(reinterpret_cast<void*>(std_vec.data()),
              reinterpret_cast<const void*>(pod_span.data()));

    // Values should match
    EXPECT_EQ(std_vec.size(), 4u);
    EXPECT_EQ(std_vec[0].real(), 10);
    EXPECT_EQ(std_vec[0].imag(), 20);
}

// ============================================================================
// Device-side Complex tests (require GPU)
// ============================================================================

__global__ void test_complex_construction_kernel(Complex16* results) {
    // Default construction
    Complex16 c0;
    results[0] = c0;

    // Value construction
    Complex16 c1(10, 20);
    results[1] = c1;

    // Real-only construction
    Complex16 c2(42);
    results[2] = c2;
}

TEST_F(ComplexHostTest, DeviceConstruction) {
    Complex16* d_results;
    cudaMalloc(&d_results, 3 * sizeof(Complex16));

    test_complex_construction_kernel<<<1, 1>>>(d_results);
    cudaDeviceSynchronize();

    std::vector<Complex16> h_results(3);
    cudaMemcpy(h_results.data(), d_results, 3 * sizeof(Complex16), cudaMemcpyDeviceToHost);
    cudaFree(d_results);

    EXPECT_EQ(h_results[0].re, 0);
    EXPECT_EQ(h_results[0].im, 0);

    EXPECT_EQ(h_results[1].re, 10);
    EXPECT_EQ(h_results[1].im, 20);

    EXPECT_EQ(h_results[2].re, 42);
    EXPECT_EQ(h_results[2].im, 0);
}

__global__ void test_complex_arithmetic_kernel(ComplexF* results) {
    ComplexF a(3.0f, 4.0f);
    ComplexF b(2.0f, 1.0f);

    results[0] = a + b;  // 5 + 5i
    results[1] = a - b;  // 1 + 3i
    results[2] = a * b;  // (3*2 - 4*1) + (3*1 + 4*2)i = 2 + 11i
    results[3] = a / b;  // 2 + i
}

TEST_F(ComplexHostTest, DeviceArithmetic) {
    ComplexF* d_results;
    cudaMalloc(&d_results, 4 * sizeof(ComplexF));

    test_complex_arithmetic_kernel<<<1, 1>>>(d_results);
    cudaDeviceSynchronize();

    std::vector<ComplexF> h_results(4);
    cudaMemcpy(h_results.data(), d_results, 4 * sizeof(ComplexF), cudaMemcpyDeviceToHost);
    cudaFree(d_results);

    // Addition
    EXPECT_FLOAT_EQ(h_results[0].re, 5.0f);
    EXPECT_FLOAT_EQ(h_results[0].im, 5.0f);

    // Subtraction
    EXPECT_FLOAT_EQ(h_results[1].re, 1.0f);
    EXPECT_FLOAT_EQ(h_results[1].im, 3.0f);

    // Multiplication
    EXPECT_FLOAT_EQ(h_results[2].re, 2.0f);
    EXPECT_FLOAT_EQ(h_results[2].im, 11.0f);

    // Division
    EXPECT_FLOAT_EQ(h_results[3].re, 2.0f);
    EXPECT_FLOAT_EQ(h_results[3].im, 1.0f);
}

__global__ void test_complex_traits_kernel(Complex16* results) {
    Complex16 c(42, 84);

    // Access via traits
    int16_t r = complex_traits<Complex16>::real(c);
    int16_t i = complex_traits<Complex16>::imag(c);

    // Make via traits
    results[0] = complex_traits<Complex16>::make(r * 2, i * 2);
}

TEST_F(ComplexHostTest, DeviceComplexTraits) {
    Complex16* d_results;
    cudaMalloc(&d_results, sizeof(Complex16));

    test_complex_traits_kernel<<<1, 1>>>(d_results);
    cudaDeviceSynchronize();

    Complex16 h_result;
    cudaMemcpy(&h_result, d_results, sizeof(Complex16), cudaMemcpyDeviceToHost);
    cudaFree(d_results);

    EXPECT_EQ(h_result.re, 84);
    EXPECT_EQ(h_result.im, 168);
}

// ============================================================================
// Memory buffer tests (DeviceBuffer, PinnedBuffer)
// ============================================================================

TEST_F(ComplexHostTest, DeviceBufferBasic) {
    DeviceBuffer<Complex16> buf(100);

    EXPECT_TRUE(buf.valid());
    EXPECT_EQ(buf.size(), 100u);
    EXPECT_EQ(buf.size_bytes(), 100 * sizeof(Complex16));
    EXPECT_NE(buf.data(), nullptr);
}

TEST_F(ComplexHostTest, DeviceBufferUploadDownload) {
    constexpr size_t count = 64;

    // Create host data
    std::vector<Complex16> h_src(count);
    for (size_t i = 0; i < count; ++i) {
        h_src[i] = Complex16(static_cast<int16_t>(i), static_cast<int16_t>(i * 2));
    }

    // Upload to device
    DeviceBuffer<Complex16> d_buf(count);
    ASSERT_TRUE(d_buf.valid());
    EXPECT_EQ(d_buf.upload(h_src.data(), count), cudaSuccess);

    // Download from device
    std::vector<Complex16> h_dst(count);
    EXPECT_EQ(d_buf.download(h_dst.data(), count), cudaSuccess);
    cudaDeviceSynchronize();

    // Verify
    for (size_t i = 0; i < count; ++i) {
        EXPECT_EQ(h_dst[i].re, static_cast<int16_t>(i));
        EXPECT_EQ(h_dst[i].im, static_cast<int16_t>(i * 2));
    }
}

TEST_F(ComplexHostTest, PinnedBufferBasic) {
    PinnedBuffer<ComplexF> buf(100);

    EXPECT_TRUE(buf.valid());
    EXPECT_EQ(buf.size(), 100u);
    EXPECT_NE(buf.data(), nullptr);

    // Array access
    buf[0] = ComplexF(1.0f, 2.0f);
    EXPECT_FLOAT_EQ(buf[0].re, 1.0f);
    EXPECT_FLOAT_EQ(buf[0].im, 2.0f);

    // Span access
    auto s = buf.span();
    EXPECT_EQ(s.size(), 100u);
}

TEST_F(ComplexHostTest, DeviceBufferMoveSemantics) {
    DeviceBuffer<Complex16> buf1(100);
    Complex16* ptr1 = buf1.data();

    // Move construct
    DeviceBuffer<Complex16> buf2(std::move(buf1));
    EXPECT_EQ(buf2.data(), ptr1);
    EXPECT_EQ(buf2.size(), 100u);
    EXPECT_EQ(buf1.data(), nullptr);
    EXPECT_EQ(buf1.size(), 0u);

    // Move assign
    DeviceBuffer<Complex16> buf3;
    buf3 = std::move(buf2);
    EXPECT_EQ(buf3.data(), ptr1);
    EXPECT_EQ(buf2.data(), nullptr);
}

// ============================================================================
// Type aliases tests
// ============================================================================

TEST_F(ComplexHostTest, TypeAliases) {
    // Verify type alias sizes
    EXPECT_EQ(sizeof(Complex8), 2u);   // 2 x int8_t
    EXPECT_EQ(sizeof(Complex16), 4u);  // 2 x int16_t
    EXPECT_EQ(sizeof(Complex32), 8u);  // 2 x int32_t
    EXPECT_EQ(sizeof(ComplexF), 8u);   // 2 x float
    EXPECT_EQ(sizeof(ComplexD), 16u);  // 2 x double
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
