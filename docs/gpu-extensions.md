# GPU Extensions for vrtigo

This document describes the GPU extensions for vrtigo, which provide CUDA-compatible utilities for working with VRT (VITA Radio Transport) packets on NVIDIA GPUs.

## Overview

The GPU extensions solve a fundamental problem: `std::complex<T>` does not work in CUDA device code because it has non-trivial constructors and member functions not marked `__device__`. The extensions provide:

- **POD complex types** that work on both host and device
- **Device-side sample I/O** with optional endian conversion
- **GPU memory management** utilities (DeviceBuffer, PinnedBuffer)
- **Device-side packet parsing** for batch processing
- **Zero-copy adapters** between POD and std::complex types

## Quick Start

### Device-to-Host Pipeline (D2H)

Generate samples on GPU, transfer to host, write to VRT packet:

```cpp
#define VRTIGO_ENABLE_CUDA
#include <vrtigo/vrtigo_gpu.hpp>
#include <vrtigo/sample_span.hpp>

using namespace vrtigo;
using namespace vrtigo::gpu;

// GPU kernel generates samples
__global__ void generate_samples(Complex16* out, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = Complex16{static_cast<int16_t>(idx), static_cast<int16_t>(-idx)};
    }
}

void gpu_to_vrt_packet() {
    constexpr size_t sample_count = 1024;

    // 1. Generate samples on GPU
    DeviceBuffer<Complex16> d_samples(sample_count);
    generate_samples<<<(sample_count+255)/256, 256>>>(d_samples.data(), sample_count);

    // 2. Transfer to host (D2H)
    PinnedBuffer<Complex16> h_samples(sample_count);
    d_samples.download(h_samples.data());
    cudaDeviceSynchronize();

    // 3. Write to VRT packet payload (with endian conversion)
    std::vector<uint8_t> payload(sample_count * sizeof(Complex16));
    SampleSpan<Complex16> span(std::span<uint8_t>(payload));

    for (size_t i = 0; i < sample_count; ++i) {
        span.set(i, h_samples[i]);  // Host-to-network byte order conversion
    }
}
```

### Host-to-Device Pipeline (H2D)

Parse VRT packet on host, upload samples to GPU:

```cpp
void vrt_packet_to_gpu(std::span<const uint8_t> packet_buffer) {
    // 1. Parse VRT packet (on host)
    std::span<const uint8_t> payload = /* extract payload from packet */;
    size_t sample_count = payload.size() / sizeof(Complex16);

    // 2. Read samples with endian conversion
    PinnedBuffer<Complex16> h_samples(sample_count);
    SampleSpanView<Complex16> view(payload);

    for (size_t i = 0; i < sample_count; ++i) {
        h_samples[i] = view[i];  // Network-to-host byte order conversion
    }

    // 3. Upload to GPU (H2D)
    DeviceBuffer<Complex16> d_samples(sample_count);
    d_samples.upload(h_samples.data());

    // 4. Process on GPU
    my_gpu_kernel<<<...>>>(d_samples.data(), sample_count);
}
```

## POD Complex Types

### The Problem with std::complex

CUDA device code cannot use `std::complex<T>` because:
- Constructors are not marked `__device__`
- Member functions like `.real()` and `.imag()` are not device-callable
- The type is not trivially copyable in the CUDA sense

### The Solution: Complex<T>

vrtigo provides `vrtigo::gpu::Complex<T>`, a POD (Plain Old Data) complex type:

```cpp
#include <vrtigo/gpu/complex.hpp>

using namespace vrtigo::gpu;

// On host or device:
Complex16 c1;              // Default: (0, 0)
Complex16 c2(10, 20);      // (10, 20)
Complex16 c3(42);          // (42, 0)

// Access components (host or device):
int16_t r = c2.real();     // Accessor method (std::complex compatible)
int16_t i = c2.imag();     // Accessor method
r = c2.re;                 // Direct member access
i = c2.im;                 // Direct member access

// Arithmetic (all __host__ __device__)
auto sum = c2 + c3;
auto prod = c2 * c3;
c1 += c2;

// Host-only: implicit conversion to/from std::complex
std::complex<int16_t> std_c = c2;
Complex16 pod_c = std_c;
```

### Type Aliases

```cpp
using Complex8  = Complex<int8_t>;   // 2 bytes
using Complex16 = Complex<int16_t>;  // 4 bytes
using Complex32 = Complex<int32_t>;  // 8 bytes
using ComplexF  = Complex<float>;    // 8 bytes
using ComplexD  = Complex<double>;   // 16 bytes
```

### complex_traits System

The `complex_traits<T>` template provides a uniform interface for accessing complex components:

```cpp
#include <vrtigo/gpu/complex_traits.hpp>

template<typename T>
struct complex_traits {
    using value_type = ...;
    static value_type real(const T&);
    static value_type imag(const T&);
    static T make(value_type r, value_type i);
};

// Usage on host or device:
Complex16 c(10, 20);
int16_t r = complex_traits<Complex16>::real(c);
int16_t i = complex_traits<Complex16>::imag(c);
Complex16 made = complex_traits<Complex16>::make(r, i);
```

### Specializing for Third-Party Types

To use your own complex types with vrtigo GPU extensions, specialize `complex_traits`:

```cpp
// For cuComplex (uses .x/.y members)
template<>
struct vrtigo::gpu::complex_traits<cuFloatComplex> {
    using value_type = float;

    VRTIGO_HOST_DEVICE
    static float real(const cuFloatComplex& c) { return c.x; }

    VRTIGO_HOST_DEVICE
    static float imag(const cuFloatComplex& c) { return c.y; }

    VRTIGO_HOST_DEVICE
    static cuFloatComplex make(float r, float i) { return make_cuFloatComplex(r, i); }
};

// For thrust::complex<T> (uses .real()/.imag() accessors)
template<typename T>
struct vrtigo::gpu::complex_traits<thrust::complex<T>> {
    using value_type = T;

    VRTIGO_HOST_DEVICE
    static T real(const thrust::complex<T>& c) { return c.real(); }

    VRTIGO_HOST_DEVICE
    static T imag(const thrust::complex<T>& c) { return c.imag(); }

    VRTIGO_HOST_DEVICE
    static thrust::complex<T> make(T r, T i) { return thrust::complex<T>(r, i); }
};
```

## Include Order

**Important:** Include order matters for ODR (One Definition Rule) safety.

### Correct Order

```cpp
// GPU extensions BEFORE core vrtigo
#include <vrtigo/gpu/sample_traits_ext.hpp>  // MUST come first
#include <vrtigo/sample_span.hpp>

// Or use the aggregator header
#include <vrtigo/vrtigo_gpu.hpp>  // Includes sample_traits_ext.hpp
#include <vrtigo.hpp>
```

### Why This Matters

`sample_traits_ext.hpp` extends `SampleTraits<T>` to recognize POD complex types. If you include `sample_span.hpp` first in one translation unit and `sample_traits_ext.hpp` first in another, you get an ODR violation.

### Project-Wide Consistency

Ensure all translation units in your project use the same include order:

```cpp
// CORRECT: Consistent across all files
// file_a.cpp
#include <vrtigo/vrtigo_gpu.hpp>
#include <vrtigo.hpp>

// file_b.cpp
#include <vrtigo/vrtigo_gpu.hpp>
#include <vrtigo.hpp>
```

## Zero-Copy vs Copy Behavior

### Layout Compatibility

The `is_layout_compatible_v` trait determines if zero-copy is safe:

```cpp
#include <vrtigo/gpu/pod_complex_adapter.hpp>

// Check at compile time
static_assert(is_layout_compatible_v<Complex16, int16_t>,
    "Complex16 must be layout-compatible for zero-copy");

// Requirements for zero-copy:
// - sizeof(PodT) == sizeof(std::complex<T>)
// - alignof(PodT) <= alignof(std::complex<T>)
// - std::is_trivially_copyable_v<PodT>
```

### Zero-Copy Adapters

When layout-compatible, use `adapt_to_pod` and `adapt_from_pod`:

```cpp
// std::complex -> POD (zero-copy)
std::complex<int16_t> std_arr[100];
std::span<std::complex<int16_t>> std_span(std_arr, 100);
auto pod_span = adapt_to_pod<Complex16>(std_span);  // Same memory!

// POD -> std::complex (zero-copy)
Complex16 pod_arr[100];
std::span<Complex16> pod_span(pod_arr, 100);
auto std_span = adapt_from_pod<int16_t>(pod_span);  // Same memory!
```

### Explicit Copy

When you need a copy (e.g., decoupled lifetimes):

```cpp
// Explicit copy versions
auto pod_vec = adapt_to_pod_copy<Complex16>(std_span);    // Returns std::vector<Complex16>
auto std_vec = adapt_from_pod_copy<int16_t>(pod_span);    // Returns std::vector<std::complex<int16_t>>
```

### When Copies Occur

| Operation | Zero-Copy? | Notes |
|-----------|------------|-------|
| `DeviceBuffer` -> `PinnedBuffer` | Yes | Same type, direct cudaMemcpy |
| `span<Complex<T>>` -> `span<std::complex<T>>` | Yes | Layout-compatible, reinterpret |
| `SampleSpan::write()` | No | Endian conversion required |
| `write_sample_raw()` on GPU | Yes | No conversion |
| Pitched GPU -> contiguous host | No | Stride mismatch |

## No-Swap I/O

When your GPU kernel produces data already in network byte order, use raw I/O functions to avoid double-swapping:

```cpp
#include <vrtigo/gpu/raw_sample_io.hpp>

// Host-side raw I/O (no endian conversion)
std::vector<uint8_t> payload(1024);
Complex16 sample{10, 20};

// Write without swap
write_sample_raw<Complex16>(std::span<uint8_t>(payload), 0, sample);

// Read without swap
Complex16 read = read_sample_raw<Complex16>(std::span<const uint8_t>(payload), 0);

// Bulk operations
std::vector<Complex16> samples(100);
write_samples_raw<Complex16>(std::span<uint8_t>(payload), std::span<const Complex16>(samples));
read_samples_raw<Complex16>(std::span<Complex16>(samples), std::span<const uint8_t>(payload));

// Copy payload bytes directly
copy_raw_payload(dest_span, src_span);
```

### Device-Side Raw I/O

```cpp
#include <vrtigo/gpu/sample_span_device.hpp>

__global__ void kernel(uint8_t* payload, Complex16* samples, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        // With endian swap (network <-> host)
        write_sample<Complex16>(payload + idx * sizeof(Complex16), samples[idx]);
        samples[idx] = read_sample<Complex16>(payload + idx * sizeof(Complex16));

        // Without endian swap (raw)
        write_sample_raw<Complex16>(payload + idx * sizeof(Complex16), samples[idx]);
        samples[idx] = read_sample_raw<Complex16>(payload + idx * sizeof(Complex16));
    }
}
```

## Memory Helpers

### DeviceBuffer<T>

RAII wrapper for GPU device memory:

```cpp
#include <vrtigo/gpu/memory.hpp>

// Allocate
DeviceBuffer<Complex16> d_buf(1024);
if (!d_buf.valid()) { /* allocation failed */ }

// Access
Complex16* ptr = d_buf.data();
size_t count = d_buf.size();
size_t bytes = d_buf.size_bytes();

// Upload from host (H2D)
std::vector<Complex16> h_data(1024);
d_buf.upload(h_data.data(), h_data.size());

// Download to host (D2H)
d_buf.download(h_data.data(), h_data.size());

// Async operations
cudaStream_t stream;
d_buf.upload(h_data.data(), h_data.size(), stream);
d_buf.download(h_data.data(), h_data.size(), stream);

// Zero memory
d_buf.zero();

// Resize (contents not preserved)
d_buf.resize(2048);
```

### PinnedBuffer<T>

RAII wrapper for pinned (page-locked) host memory:

```cpp
PinnedBuffer<Complex16> h_buf(1024);

// Array access
h_buf[0] = Complex16{1, 2};
Complex16 val = h_buf[0];

// Span access
std::span<Complex16> s = h_buf.span();
```

### Move Semantics

Both buffer types are move-only:

```cpp
DeviceBuffer<Complex16> buf1(1024);
DeviceBuffer<Complex16> buf2 = std::move(buf1);  // buf1 is now empty
```

## Building with CUDA

### CMake Configuration

```cmake
# Enable CUDA in your project
project(myapp LANGUAGES CXX CUDA)

# Find vrtigo
find_package(Vrtigo REQUIRED)

# Build with GPU support
option(MYAPP_ENABLE_CUDA "Enable CUDA support" ON)

if(MYAPP_ENABLE_CUDA)
    target_link_libraries(myapp PRIVATE vrtigo::vrtigo)
    target_compile_definitions(myapp PRIVATE VRTIGO_ENABLE_CUDA)
endif()
```

### CMake Options for vrtigo

```bash
# Enable CUDA extensions
cmake -DVRTIGO_ENABLE_CUDA=ON ..

# Build GPU tests (requires GPU runtime)
cmake -DVRTIGO_ENABLE_CUDA=ON -DVRTIGO_BUILD_GPU_TESTS=ON ..

# Build GPU syntax check (no runtime needed)
cmake -DVRTIGO_ENABLE_CUDA=ON -DVRTIGO_BUILD_GPU_SYNTAX_CHECK=ON ..
```

## Testing

### Using CI Scripts (Recommended)

The CI scripts handle CMake compatibility issues with newer CUDA versions (e.g., CUDA 13.x):

```bash
# Run GPU syntax check (no GPU runtime required)
./scripts/ci/gpu-syntax-check.sh

# Run full GPU tests (requires GPU runtime)
./scripts/ci/gpu-test.sh
```

### Running GPU Tests with CMake

**Note:** CMake's CUDA language support may not work with very new CUDA versions. If you encounter `_CMAKE_CUDA_WHOLE_FLAG` errors, use the CI scripts above instead.

```bash
# Build with GPU tests
cmake -B build -DVRTIGO_ENABLE_CUDA=ON -DVRTIGO_BUILD_GPU_TESTS=ON
cmake --build build

# Run all GPU tests
cd build && ctest -L gpu --output-on-failure

# Run specific test
./build/tests/gpu_complex_test
./build/tests/gpu_integration_test
```

### Syntax Check (No GPU Required)

The syntax check compiles all GPU headers without running any code. Useful for CI without GPU access:

```bash
# Using CI script (recommended)
./scripts/ci/gpu-syntax-check.sh

# Or directly with nvcc
nvcc -c --std=c++20 -x cu -I include -DVRTIGO_ENABLE_CUDA \
    tests/gpu/compile_check.cu -o /dev/null
```

## Device-Side Packet Parsing

Parse VRT packet headers on the GPU for batch processing:

```cpp
#include <vrtigo/gpu/packet_parser.hpp>

__global__ void parse_packets_kernel(const uint8_t* packets, DecodedHeader* headers,
                                      size_t packet_count, size_t packet_stride) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= packet_count) return;

    // Parse header
    headers[idx] = parse_packet(packets + idx * packet_stride);

    // Use decoded info
    const DecodedHeader& hdr = headers[idx];

    if (is_signal_data_packet(hdr.type)) {
        uint16_t payload_words = payload_size_words(hdr);
        const uint8_t* payload = get_payload_ptr(packets + idx * packet_stride, hdr);
        // Process payload...
    }
}
```

### DecodedHeader Fields

```cpp
struct DecodedHeader {
    PacketType type;        // signal_data, context, command, etc.
    uint16_t size_words;    // Total packet size in 32-bit words
    bool has_class_id;      // Class ID present
    TsiType tsi;            // Integer timestamp type
    TsfType tsf;            // Fractional timestamp type
    uint8_t packet_count;   // Packet sequence counter

    // Type-specific fields
    bool trailer_included;  // Signal/ExtData packets
    bool signal_spectrum;   // Signal/ExtData packets
    bool context_tsm;       // Context packets
    bool command_ack;       // Command packets
    // ...
};
```

## Header Reference

| Header | Purpose |
|--------|---------|
| `vrtigo_gpu.hpp` | Aggregator header (includes all GPU extensions) |
| `gpu/complex.hpp` | POD complex type `Complex<T>` |
| `gpu/complex_traits.hpp` | Trait system for complex types |
| `gpu/sample_traits_ext.hpp` | SampleSpan support for POD complex |
| `gpu/raw_sample_io.hpp` | No-swap I/O (host-side) |
| `gpu/sample_span_device.hpp` | Device-side sample I/O |
| `gpu/pod_complex_adapter.hpp` | Zero-copy adapters |
| `gpu/endian.hpp` | Device-compatible byte swap |
| `gpu/memory.hpp` | DeviceBuffer, PinnedBuffer |
| `gpu/packet_parser.hpp` | Device-side packet parsing |
| `gpu/detail/cuda_macros.hpp` | VRTIGO_HOST_DEVICE macros |

## Concepts

The GPU extensions define the following C++20 concepts:

```cpp
// Type with valid complex_traits specialization
template<typename T>
concept ComplexType = requires(const T t, typename complex_traits<T>::value_type v) {
    typename complex_traits<T>::value_type;
    { complex_traits<T>::real(t) } -> std::convertible_to<decltype(v)>;
    { complex_traits<T>::imag(t) } -> std::convertible_to<decltype(v)>;
    { complex_traits<T>::make(v, v) } -> std::same_as<T>;
};

// GPU-compatible sample type (scalar or complex)
template<typename T>
concept GpuSampleType =
    std::same_as<T, int8_t>  || std::same_as<T, int16_t> ||
    std::same_as<T, int32_t> || std::same_as<T, float>   ||
    std::same_as<T, double>  || ComplexType<T>;
```
