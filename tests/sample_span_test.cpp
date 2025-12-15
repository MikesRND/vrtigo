#include <array>
#include <complex>
#include <vector>

#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <vrtigo/sample_span.hpp>

using namespace vrtigo;

// Helper to write a value in network byte order (big-endian)
template <typename T>
void write_network_order(uint8_t* buffer, T value) {
    if constexpr (sizeof(T) == 2) {
        uint16_t raw;
        std::memcpy(&raw, &value, sizeof(raw));
        raw = detail::host_to_network16(raw);
        std::memcpy(buffer, &raw, sizeof(raw));
    } else if constexpr (sizeof(T) == 4) {
        uint32_t raw;
        std::memcpy(&raw, &value, sizeof(raw));
        raw = detail::host_to_network32(raw);
        std::memcpy(buffer, &raw, sizeof(raw));
    } else if constexpr (sizeof(T) == 8) {
        uint64_t raw;
        std::memcpy(&raw, &value, sizeof(raw));
        raw = detail::host_to_network64(raw);
        std::memcpy(buffer, &raw, sizeof(raw));
    } else {
        std::memcpy(buffer, &value, sizeof(T));
    }
}

// =============================================================================
// SampleTraits Tests
// =============================================================================

TEST(SampleTraitsTest, ScalarSizes) {
    EXPECT_EQ(detail::SampleTraits<int8_t>::component_size, 1);
    EXPECT_EQ(detail::SampleTraits<int8_t>::sample_size, 1);

    EXPECT_EQ(detail::SampleTraits<int16_t>::component_size, 2);
    EXPECT_EQ(detail::SampleTraits<int16_t>::sample_size, 2);

    EXPECT_EQ(detail::SampleTraits<int32_t>::component_size, 4);
    EXPECT_EQ(detail::SampleTraits<int32_t>::sample_size, 4);

    EXPECT_EQ(detail::SampleTraits<float>::component_size, 4);
    EXPECT_EQ(detail::SampleTraits<float>::sample_size, 4);

    EXPECT_EQ(detail::SampleTraits<double>::component_size, 8);
    EXPECT_EQ(detail::SampleTraits<double>::sample_size, 8);
}

TEST(SampleTraitsTest, ComplexSizes) {
    EXPECT_EQ(detail::SampleTraits<std::complex<int8_t>>::component_size, 1);
    EXPECT_EQ(detail::SampleTraits<std::complex<int8_t>>::sample_size, 2);

    EXPECT_EQ(detail::SampleTraits<std::complex<int16_t>>::component_size, 2);
    EXPECT_EQ(detail::SampleTraits<std::complex<int16_t>>::sample_size, 4);

    EXPECT_EQ(detail::SampleTraits<std::complex<int32_t>>::component_size, 4);
    EXPECT_EQ(detail::SampleTraits<std::complex<int32_t>>::sample_size, 8);

    EXPECT_EQ(detail::SampleTraits<std::complex<float>>::component_size, 4);
    EXPECT_EQ(detail::SampleTraits<std::complex<float>>::sample_size, 8);

    EXPECT_EQ(detail::SampleTraits<std::complex<double>>::component_size, 8);
    EXPECT_EQ(detail::SampleTraits<std::complex<double>>::sample_size, 16);
}

TEST(SampleTraitsTest, ValidSampleTypeConcept) {
    // These should all satisfy the concept
    static_assert(ValidSampleType<int8_t>);
    static_assert(ValidSampleType<int16_t>);
    static_assert(ValidSampleType<int32_t>);
    static_assert(ValidSampleType<float>);
    static_assert(ValidSampleType<double>);
    static_assert(ValidSampleType<std::complex<int8_t>>);
    static_assert(ValidSampleType<std::complex<int16_t>>);
    static_assert(ValidSampleType<std::complex<int32_t>>);
    static_assert(ValidSampleType<std::complex<float>>);
    static_assert(ValidSampleType<std::complex<double>>);

    // These should NOT satisfy the concept (compilation would fail if used)
    static_assert(!ValidSampleType<uint8_t>);
    static_assert(!ValidSampleType<uint16_t>);
    static_assert(!ValidSampleType<uint32_t>);
    static_assert(!ValidSampleType<char>);
    // Note: int/long may satisfy concept on some platforms if they match int32_t
}

// =============================================================================
// SampleSpanView Tests - int16_t
// =============================================================================

TEST(SampleSpanViewTest, Int16Count) {
    std::array<uint8_t, 10> buffer{};
    SampleSpanView<int16_t> view(buffer);

    // 10 bytes / 2 bytes per sample = 5 samples
    EXPECT_EQ(view.count(), 5);
    EXPECT_EQ(view.size_bytes(), 10);
}

TEST(SampleSpanViewTest, Int16PartialSample) {
    std::array<uint8_t, 7> buffer{}; // 3.5 samples
    SampleSpanView<int16_t> view(buffer);

    // Should floor to 3 samples
    EXPECT_EQ(view.count(), 3);
}

TEST(SampleSpanViewTest, Int16ReadSingle) {
    std::array<uint8_t, 4> buffer{};
    int16_t val1 = 0x1234;
    int16_t val2 = -100;
    write_network_order(buffer.data(), val1);
    write_network_order(buffer.data() + 2, val2);

    SampleSpanView<int16_t> view(buffer);
    EXPECT_EQ(view[0], val1);
    EXPECT_EQ(view[1], val2);
}

TEST(SampleSpanViewTest, Int16BulkRead) {
    std::array<uint8_t, 8> buffer{};
    int16_t values[] = {100, 200, -300, 400};
    for (int i = 0; i < 4; ++i) {
        write_network_order(buffer.data() + i * 2, values[i]);
    }

    SampleSpanView<int16_t> view(buffer);
    std::array<int16_t, 4> dest{};
    size_t copied = view.read(dest);

    EXPECT_EQ(copied, 4);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(dest[i], values[i]);
    }
}

TEST(SampleSpanViewTest, Int16BulkReadWithOffset) {
    std::array<uint8_t, 8> buffer{};
    int16_t values[] = {100, 200, -300, 400};
    for (int i = 0; i < 4; ++i) {
        write_network_order(buffer.data() + i * 2, values[i]);
    }

    SampleSpanView<int16_t> view(buffer);
    std::array<int16_t, 2> dest{};
    size_t copied = view.read(dest, 2); // Start at index 2

    EXPECT_EQ(copied, 2);
    EXPECT_EQ(dest[0], -300);
    EXPECT_EQ(dest[1], 400);
}

TEST(SampleSpanViewTest, Int16BulkReadOffsetPastEnd) {
    std::array<uint8_t, 8> buffer{};
    SampleSpanView<int16_t> view(buffer);

    std::array<int16_t, 2> dest{};
    size_t copied = view.read(dest, 10); // Past end

    EXPECT_EQ(copied, 0);
}

// =============================================================================
// SampleSpanView Tests - float
// =============================================================================

TEST(SampleSpanViewTest, FloatCount) {
    std::array<uint8_t, 16> buffer{};
    SampleSpanView<float> view(buffer);

    EXPECT_EQ(view.count(), 4);
}

TEST(SampleSpanViewTest, FloatReadSingle) {
    std::array<uint8_t, 8> buffer{};
    float val1 = 1.5f;
    float val2 = -3.14159f;
    write_network_order(buffer.data(), val1);
    write_network_order(buffer.data() + 4, val2);

    SampleSpanView<float> view(buffer);
    EXPECT_FLOAT_EQ(view[0], val1);
    EXPECT_FLOAT_EQ(view[1], val2);
}

// =============================================================================
// SampleSpanView Tests - std::complex<int16_t>
// =============================================================================

TEST(SampleSpanViewTest, ComplexInt16Count) {
    std::array<uint8_t, 16> buffer{};
    SampleSpanView<std::complex<int16_t>> view(buffer);

    // 16 bytes / 4 bytes per complex sample = 4 samples
    EXPECT_EQ(view.count(), 4);
}

TEST(SampleSpanViewTest, ComplexInt16ReadSingle) {
    std::array<uint8_t, 8> buffer{};
    // Sample 0: I=100, Q=200
    write_network_order<int16_t>(buffer.data(), 100);
    write_network_order<int16_t>(buffer.data() + 2, 200);
    // Sample 1: I=-50, Q=-75
    write_network_order<int16_t>(buffer.data() + 4, -50);
    write_network_order<int16_t>(buffer.data() + 6, -75);

    SampleSpanView<std::complex<int16_t>> view(buffer);

    auto s0 = view[0];
    EXPECT_EQ(s0.real(), 100);
    EXPECT_EQ(s0.imag(), 200);

    auto s1 = view[1];
    EXPECT_EQ(s1.real(), -50);
    EXPECT_EQ(s1.imag(), -75);
}

// =============================================================================
// SampleSpanView Tests - std::complex<float>
// =============================================================================

TEST(SampleSpanViewTest, ComplexFloatReadSingle) {
    std::array<uint8_t, 16> buffer{};
    // Sample 0: I=1.5, Q=2.5
    write_network_order<float>(buffer.data(), 1.5f);
    write_network_order<float>(buffer.data() + 4, 2.5f);
    // Sample 1: I=-3.0, Q=4.0
    write_network_order<float>(buffer.data() + 8, -3.0f);
    write_network_order<float>(buffer.data() + 12, 4.0f);

    SampleSpanView<std::complex<float>> view(buffer);

    auto s0 = view[0];
    EXPECT_FLOAT_EQ(s0.real(), 1.5f);
    EXPECT_FLOAT_EQ(s0.imag(), 2.5f);

    auto s1 = view[1];
    EXPECT_FLOAT_EQ(s1.real(), -3.0f);
    EXPECT_FLOAT_EQ(s1.imag(), 4.0f);
}

// =============================================================================
// SampleSpan (Mutable) Tests
// =============================================================================

TEST(SampleSpanTest, Int16SetSingle) {
    std::array<uint8_t, 4> buffer{};
    SampleSpan<int16_t> span(buffer);

    span.set(0, 0x1234);
    span.set(1, -500);

    // Verify by reading back
    EXPECT_EQ(span[0], 0x1234);
    EXPECT_EQ(span[1], -500);
}

TEST(SampleSpanTest, FloatSetSingle) {
    std::array<uint8_t, 8> buffer{};
    SampleSpan<float> span(buffer);

    span.set(0, 3.14159f);
    span.set(1, -2.71828f);

    EXPECT_FLOAT_EQ(span[0], 3.14159f);
    EXPECT_FLOAT_EQ(span[1], -2.71828f);
}

TEST(SampleSpanTest, ComplexFloatSetSingle) {
    std::array<uint8_t, 16> buffer{};
    SampleSpan<std::complex<float>> span(buffer);

    span.set(0, {1.0f, 2.0f});
    span.set(1, {-3.0f, 4.0f});

    auto s0 = span[0];
    EXPECT_FLOAT_EQ(s0.real(), 1.0f);
    EXPECT_FLOAT_EQ(s0.imag(), 2.0f);

    auto s1 = span[1];
    EXPECT_FLOAT_EQ(s1.real(), -3.0f);
    EXPECT_FLOAT_EQ(s1.imag(), 4.0f);
}

TEST(SampleSpanTest, BulkWrite) {
    std::array<uint8_t, 16> buffer{};
    SampleSpan<int32_t> span(buffer);

    std::array<int32_t, 4> src = {1000, 2000, -3000, 4000};
    size_t written = span.write(src);

    EXPECT_EQ(written, 4);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(span[i], src[i]);
    }
}

TEST(SampleSpanTest, BulkWriteWithOffset) {
    std::array<uint8_t, 16> buffer{};
    SampleSpan<int32_t> span(buffer);

    // Fill with zeros first
    span.set(0, 0);
    span.set(1, 0);
    span.set(2, 0);
    span.set(3, 0);

    std::array<int32_t, 2> src = {1234, 5678};
    size_t written = span.write(src, 1); // Start at index 1

    EXPECT_EQ(written, 2);
    EXPECT_EQ(span[0], 0);
    EXPECT_EQ(span[1], 1234);
    EXPECT_EQ(span[2], 5678);
    EXPECT_EQ(span[3], 0);
}

TEST(SampleSpanTest, BulkWriteOffsetPastEnd) {
    std::array<uint8_t, 8> buffer{};
    SampleSpan<int32_t> span(buffer);

    std::array<int32_t, 2> src = {1, 2};
    size_t written = span.write(src, 10); // Past end

    EXPECT_EQ(written, 0);
}

TEST(SampleSpanTest, BulkWritePartial) {
    std::array<uint8_t, 8> buffer{}; // Room for 2 int32_t
    SampleSpan<int32_t> span(buffer);

    std::array<int32_t, 4> src = {1, 2, 3, 4}; // More than fits
    size_t written = span.write(src);

    EXPECT_EQ(written, 2);
    EXPECT_EQ(span[0], 1);
    EXPECT_EQ(span[1], 2);
}

// =============================================================================
// int8_t Tests (no byte swapping)
// =============================================================================

TEST(SampleSpanTest, Int8NoSwap) {
    std::array<uint8_t, 4> buffer = {0x12, 0x34, 0x56, 0x78};
    SampleSpanView<int8_t> view(buffer);

    EXPECT_EQ(view.count(), 4);
    EXPECT_EQ(view[0], 0x12);
    EXPECT_EQ(view[1], 0x34);
    EXPECT_EQ(view[2], 0x56);
    EXPECT_EQ(view[3], 0x78);
}

TEST(SampleSpanTest, ComplexInt8) {
    std::array<uint8_t, 4> buffer = {10, 20, 30, 40};
    SampleSpanView<std::complex<int8_t>> view(buffer);

    EXPECT_EQ(view.count(), 2);

    auto s0 = view[0];
    EXPECT_EQ(s0.real(), 10);
    EXPECT_EQ(s0.imag(), 20);

    auto s1 = view[1];
    EXPECT_EQ(s1.real(), 30);
    EXPECT_EQ(s1.imag(), 40);
}

// =============================================================================
// Double Tests
// =============================================================================

TEST(SampleSpanTest, DoubleReadWrite) {
    std::array<uint8_t, 16> buffer{};
    SampleSpan<double> span(buffer);

    span.set(0, 3.141592653589793);
    span.set(1, -2.718281828459045);

    EXPECT_DOUBLE_EQ(span[0], 3.141592653589793);
    EXPECT_DOUBLE_EQ(span[1], -2.718281828459045);
}

TEST(SampleSpanTest, ComplexDoubleReadWrite) {
    std::array<uint8_t, 32> buffer{};
    SampleSpan<std::complex<double>> span(buffer);

    span.set(0, {1.1, 2.2});
    span.set(1, {-3.3, 4.4});

    auto s0 = span[0];
    EXPECT_DOUBLE_EQ(s0.real(), 1.1);
    EXPECT_DOUBLE_EQ(s0.imag(), 2.2);

    auto s1 = span[1];
    EXPECT_DOUBLE_EQ(s1.real(), -3.3);
    EXPECT_DOUBLE_EQ(s1.imag(), 4.4);
}

// =============================================================================
// Empty Buffer Tests
// =============================================================================

TEST(SampleSpanTest, EmptyBuffer) {
    std::span<uint8_t> empty;
    SampleSpanView<int16_t> view(empty);

    EXPECT_EQ(view.count(), 0);
    EXPECT_EQ(view.size_bytes(), 0);

    std::array<int16_t, 4> dest{};
    EXPECT_EQ(view.read(dest), 0);
}

// =============================================================================
// Roundtrip Tests
// =============================================================================

TEST(SampleSpanTest, RoundtripInt16) {
    std::array<uint8_t, 20> buffer{};
    SampleSpan<int16_t> span(buffer);

    // Write values
    std::array<int16_t, 10> original = {0, 1, -1, 32767, -32768, 100, -100, 1000, -1000, 12345};
    span.write(original);

    // Read back
    std::array<int16_t, 10> recovered{};
    span.read(recovered);

    for (size_t i = 0; i < 10; ++i) {
        EXPECT_EQ(recovered[i], original[i]) << "Mismatch at index " << i;
    }
}

TEST(SampleSpanTest, RoundtripComplexFloat) {
    std::array<uint8_t, 32> buffer{};
    SampleSpan<std::complex<float>> span(buffer);

    std::array<std::complex<float>, 4> original = {
        std::complex<float>{1.0f, 2.0f}, std::complex<float>{-3.0f, 4.0f},
        std::complex<float>{0.0f, 0.0f}, std::complex<float>{-1.5f, -2.5f}};
    span.write(original);

    std::array<std::complex<float>, 4> recovered{};
    span.read(recovered);

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(recovered[i].real(), original[i].real());
        EXPECT_FLOAT_EQ(recovered[i].imag(), original[i].imag());
    }
}
