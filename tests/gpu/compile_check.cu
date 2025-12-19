/**
 * @file compile_check.cu
 * @brief CUDA compilation syntax check for vrtigo GPU headers
 *
 * This file includes all vrtigo GPU headers and instantiates template
 * functions to verify they compile correctly with nvcc. No runtime
 * is required - this is purely a syntax check.
 *
 * Build with: nvcc -c --std=c++20 compile_check.cu -o /dev/null
 * Or via CMake: cmake --build . --target gpu_compile_check
 */

// VRTIGO_ENABLE_CUDA should be defined via compiler flag (-DVRTIGO_ENABLE_CUDA)
#ifndef VRTIGO_ENABLE_CUDA
#define VRTIGO_ENABLE_CUDA
#endif

// Include all GPU headers
#include <vrtigo/vrtigo_gpu.hpp>

// Include core vrtigo headers after GPU extensions for proper SampleTraits setup
#include <vrtigo/sample_span.hpp>

#include <cstdint>

// ============================================================================
// Template instantiation to verify compilation (must be at namespace scope)
// ============================================================================

// Instantiate Complex<T> for all supported types
template struct vrtigo::gpu::Complex<int8_t>;
template struct vrtigo::gpu::Complex<int16_t>;
template struct vrtigo::gpu::Complex<int32_t>;
template struct vrtigo::gpu::Complex<float>;
template struct vrtigo::gpu::Complex<double>;

// Verify complex_traits work for our Complex types
static_assert(vrtigo::gpu::ComplexType<vrtigo::gpu::Complex8>);
static_assert(vrtigo::gpu::ComplexType<vrtigo::gpu::Complex16>);
static_assert(vrtigo::gpu::ComplexType<vrtigo::gpu::Complex32>);
static_assert(vrtigo::gpu::ComplexType<vrtigo::gpu::ComplexF>);
static_assert(vrtigo::gpu::ComplexType<vrtigo::gpu::ComplexD>);

// Verify GpuSampleType concept works for scalars and complex
static_assert(vrtigo::gpu::GpuSampleType<int8_t>);
static_assert(vrtigo::gpu::GpuSampleType<int16_t>);
static_assert(vrtigo::gpu::GpuSampleType<int32_t>);
static_assert(vrtigo::gpu::GpuSampleType<float>);
static_assert(vrtigo::gpu::GpuSampleType<double>);
static_assert(vrtigo::gpu::GpuSampleType<vrtigo::gpu::Complex16>);
static_assert(vrtigo::gpu::GpuSampleType<vrtigo::gpu::ComplexF>);

namespace {

// Verify layout compatibility for zero-copy adapters
static_assert(vrtigo::gpu::is_layout_compatible_v<vrtigo::gpu::Complex16, int16_t>);
static_assert(vrtigo::gpu::is_layout_compatible_v<vrtigo::gpu::ComplexF, float>);
static_assert(vrtigo::gpu::is_layout_compatible_v<vrtigo::gpu::ComplexD, double>);

// ============================================================================
// Dummy functions that use GPU APIs (never called, just compiled)
// ============================================================================

// Host function using endian helpers
void host_endian_check() {
    using namespace vrtigo::gpu;

    uint16_t val16 = 0x1234;
    uint32_t val32 = 0x12345678;
    uint64_t val64 = 0x123456789ABCDEF0ULL;

    // These should compile and call the host implementations
    (void)byteswap16(val16);
    (void)byteswap32(val32);
    (void)byteswap64(val64);
    (void)host_to_network16(val16);
    (void)host_to_network32(val32);
    (void)host_to_network64(val64);
    (void)network_to_host16(val16);
    (void)network_to_host32(val32);
    (void)network_to_host64(val64);
}

// Host function using Complex type
void host_complex_check() {
    using namespace vrtigo::gpu;

    // Construction
    Complex16 c1;
    Complex16 c2(10, 20);
    Complex16 c3(5);

    // Arithmetic
    auto sum = c2 + c3;
    auto diff = c2 - c3;
    auto prod = c2 * c3;
    (void)sum;
    (void)diff;
    (void)prod;

    // Compound assignment
    c1 += c2;
    c1 -= c3;
    c1 *= c2;

    // Unary
    auto neg = -c1;
    auto pos = +c1;
    (void)neg;
    (void)pos;

    // Comparison
    bool eq = (c1 == c2);
    bool neq = (c1 != c2);
    (void)eq;
    (void)neq;

    // complex_traits access
    auto r = complex_traits<Complex16>::real(c2);
    auto i = complex_traits<Complex16>::imag(c2);
    auto made = complex_traits<Complex16>::make(r, i);
    (void)r;
    (void)i;
    (void)made;

    // Free function accessors
    (void)real(c2);
    (void)imag(c2);
}

// Host function using raw sample I/O
void host_raw_sample_io_check() {
    using namespace vrtigo::gpu;

    uint8_t buffer[256] = {};
    std::span<uint8_t> payload(buffer, 256);
    std::span<const uint8_t> const_payload(buffer, 256);

    // Scalar raw I/O
    write_sample_raw<int16_t>(payload, 0, 0x1234);
    int16_t val16 = read_sample_raw<int16_t>(const_payload, 0);
    (void)val16;

    // Complex raw I/O
    write_sample_raw<Complex16>(payload, 0, Complex16{1, 2});
    Complex16 cval = read_sample_raw<Complex16>(const_payload, 0);
    (void)cval;

    // Bulk operations
    Complex16 samples[4];
    std::span<Complex16> sample_span(samples, 4);
    std::span<const Complex16> const_sample_span(samples, 4);

    (void)read_samples_raw<Complex16>(sample_span, const_payload, 0);
    (void)write_samples_raw<Complex16>(payload, const_sample_span, 0);

    // Copy raw payload
    uint8_t dest[128];
    (void)copy_raw_payload(std::span<uint8_t>(dest, 128), const_payload);
}

// Host function using POD complex adapter
void host_adapter_check() {
    using namespace vrtigo::gpu;

    // Test adapt_to_pod with layout-compatible types
    std::complex<int16_t> std_complex_arr[4] = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
    std::span<std::complex<int16_t>> std_span(std_complex_arr, 4);

    auto pod_span = adapt_to_pod<Complex16>(std_span);
    (void)pod_span;

    // Const version
    std::span<const std::complex<int16_t>> const_std_span(std_complex_arr, 4);
    auto const_pod_span = adapt_to_pod<Complex16>(const_std_span);
    (void)const_pod_span;

    // Test adapt_from_pod
    Complex16 pod_arr[4] = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
    std::span<Complex16> pod_src_span(pod_arr, 4);

    auto std_result_span = adapt_from_pod<int16_t>(pod_src_span);
    (void)std_result_span;

    // Copy versions (explicit copy even when layout compatible)
    auto copied_pod = adapt_to_pod_copy<Complex16>(const_std_span);
    (void)copied_pod;

    std::span<const Complex16> const_pod_src_span(pod_arr, 4);
    auto copied_std = adapt_from_pod_copy<int16_t>(const_pod_src_span);
    (void)copied_std;
}

// Host function using packet parser
void host_packet_parser_check() {
    using namespace vrtigo::gpu;

    // Test header decoding
    uint32_t header_word = 0x18000010;  // Signal data packet, 16 words
    DecodedHeader hdr = decode_header(header_word);

    (void)hdr.type;
    (void)hdr.size_words;
    (void)hdr.has_class_id;
    (void)hdr.tsi;
    (void)hdr.tsf;
    (void)hdr.packet_count;

    // Type checks
    (void)is_valid_packet_type(hdr.type);
    (void)is_signal_data_packet(hdr.type);
    (void)is_ext_data_packet(hdr.type);
    (void)is_context_packet(hdr.type);
    (void)is_command_packet(hdr.type);
    (void)has_stream_identifier(hdr.type);

    // Offset calculations
    (void)payload_offset_words(hdr);
    (void)payload_size_words(hdr);

    // Parse from buffer
    uint8_t packet[64] = {};
    packet[0] = 0x18;  // packet type and flags
    packet[3] = 0x10;  // size in words

    DecodedHeader parsed = parse_packet(packet);
    (void)parsed;

    const uint8_t* payload_ptr = get_payload_ptr(packet, parsed);
    (void)payload_ptr;
}

// Host function using SampleSpan with POD complex (via extended traits)
void host_sample_span_check() {
    using namespace vrtigo;
    using namespace vrtigo::gpu;

    uint8_t buffer[256] = {};
    std::span<uint8_t> payload(buffer, 256);

    // SampleSpan should work with Complex16 due to sample_traits_ext.hpp
    SampleSpan<Complex16> span(payload);

    (void)span.count();
    (void)span.size_bytes();

    // Write and read
    span.set(0, Complex16{100, 200});
    Complex16 read_val = span[0];
    (void)read_val;

    // Bulk operations
    Complex16 samples[4] = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
    std::span<const Complex16> src_span(samples, 4);
    (void)span.write(src_span);

    Complex16 dest[4];
    std::span<Complex16> dest_span(dest, 4);
    (void)span.read(dest_span);
}

// Device kernel for endian byte swap operations
__global__ void device_endian_kernel(uint16_t* out16, uint32_t* out32, uint64_t* out64) {
    using namespace vrtigo::gpu;

    // Device-side byte swap
    *out16 = byteswap16(0x1234);
    *out32 = byteswap32(0x12345678);
    *out64 = byteswap64(0x123456789ABCDEF0ULL);

    // Network byte order conversions
    *out16 = host_to_network16(*out16);
    *out32 = host_to_network32(*out32);
    *out64 = host_to_network64(*out64);

    *out16 = network_to_host16(*out16);
    *out32 = network_to_host32(*out32);
    *out64 = network_to_host64(*out64);
}

// Device kernel for Complex type operations
__global__ void device_complex_kernel(vrtigo::gpu::Complex16* out) {
    using namespace vrtigo::gpu;

    // Construction
    Complex16 c1;
    Complex16 c2(10, 20);
    Complex16 c3(5);

    // Arithmetic
    out[0] = c2 + c3;
    out[1] = c2 - c3;
    out[2] = c2 * c3;
    out[3] = c2 / c3;

    // Compound assignment
    c1 = c2;
    c1 += c3;
    out[4] = c1;

    c1 = c2;
    c1 -= c3;
    out[5] = c1;

    c1 = c2;
    c1 *= c3;
    out[6] = c1;

    // Unary
    out[7] = -c2;
    out[8] = +c2;

    // Scalar multiplication
    out[9] = c2 * int16_t(2);
    out[10] = int16_t(3) * c2;
    out[11] = c2 / int16_t(2);

    // complex_traits access
    int16_t r = complex_traits<Complex16>::real(c2);
    int16_t i = complex_traits<Complex16>::imag(c2);
    out[12] = complex_traits<Complex16>::make(r, i);
}

// Device kernel for sample read/write operations
__global__ void device_sample_io_kernel(uint8_t* buffer, vrtigo::gpu::Complex16* samples) {
    using namespace vrtigo::gpu;

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Write sample with endian swap
    write_sample<Complex16>(buffer + idx * sizeof(Complex16), samples[idx]);

    // Read sample with endian swap
    samples[idx] = read_sample<Complex16>(buffer + idx * sizeof(Complex16));

    // Raw write/read (no swap)
    write_sample_raw<Complex16>(buffer + idx * sizeof(Complex16), samples[idx]);
    samples[idx] = read_sample_raw<Complex16>(buffer + idx * sizeof(Complex16));

    // Strided operations
    write_sample_strided<Complex16>(buffer, samples[idx], idx, sizeof(Complex16) * 2);
    samples[idx] = read_sample_strided<Complex16>(buffer, idx, sizeof(Complex16) * 2);
}

// Device kernel for packet parsing
__global__ void device_packet_parser_kernel(const uint8_t* packets, vrtigo::gpu::DecodedHeader* headers, size_t packet_count) {
    using namespace vrtigo::gpu;

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= packet_count) return;

    // Parse packet header
    headers[idx] = parse_packet(packets + idx * 64);  // Assume 64-byte packet alignment

    // Use decoded header
    const DecodedHeader& hdr = headers[idx];

    // Type checks work on device
    bool is_data = is_signal_data_packet(hdr.type);
    bool has_stream = has_stream_identifier(hdr.type);
    (void)is_data;
    (void)has_stream;

    // Offset calculations work on device
    uint16_t payload_off = payload_offset_words(hdr);
    uint16_t payload_sz = payload_size_words(hdr);
    (void)payload_off;
    (void)payload_sz;

    // Get payload pointer
    const uint8_t* payload = get_payload_ptr(packets + idx * 64, hdr);
    (void)payload;
}

// Dummy function to prevent unused function warnings
// and ensure all code is compiled
void instantiate_all() {
    host_endian_check();
    host_complex_check();
    host_raw_sample_io_check();
    host_adapter_check();
    host_packet_parser_check();
    host_sample_span_check();

    // These are never actually launched, just compiled
    uint16_t* out16 = nullptr;
    uint32_t* out32 = nullptr;
    uint64_t* out64 = nullptr;
    vrtigo::gpu::Complex16* complex_out = nullptr;
    uint8_t* buffer = nullptr;
    vrtigo::gpu::DecodedHeader* headers = nullptr;

    device_endian_kernel<<<1, 1>>>(out16, out32, out64);
    device_complex_kernel<<<1, 1>>>(complex_out);
    device_sample_io_kernel<<<1, 1>>>(buffer, complex_out);
    device_packet_parser_kernel<<<1, 1>>>(buffer, headers, 0);
}

}  // anonymous namespace
