// GPU consumer to verify vrtigo::gpu install works correctly
// This is built out-of-tree using find_package(vrtigo) to ensure
// downstream consumers can use vrtigo::gpu with CUDA

#include <iostream>
#include <vector>

#include <vrtigo/gpu/complex.hpp>
#include <vrtigo/gpu/memory.hpp>
#include <vrtigo/gpu/raw_sample_io.hpp>
#include <vrtigo/version.hpp>

// Simple kernel to verify device code compiles
__global__ void add_complex_kernel(vrtigo::gpu::ComplexF* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += vrtigo::gpu::ComplexF{1.0f, 0.0f};
    }
}

int main() {
    std::cout << "VRTIGO GPU Install Verification\n";
    std::cout << "================================\n";
    std::cout << "vrtigo version: " << vrtigo::version_string << "\n\n";

    // Verify GPU headers compile and concepts work
    static_assert(vrtigo::gpu::ComplexType<vrtigo::gpu::ComplexF>,
                  "ComplexF must satisfy ComplexType concept");
    static_assert(vrtigo::gpu::GpuSampleType<vrtigo::gpu::ComplexF>,
                  "ComplexF must satisfy GpuSampleType concept");

    std::cout << "Concepts verified at compile time.\n";

    // Create device buffer
    constexpr size_t N = 16;
    vrtigo::gpu::DeviceBuffer<vrtigo::gpu::ComplexF> buffer(N);

    if (!buffer.valid()) {
        std::cerr << "ERROR: Failed to allocate device buffer\n";
        return 1;
    }

    std::cout << "Allocated device buffer: " << buffer.size() << " elements\n";

    // Upload test data
    std::vector<vrtigo::gpu::ComplexF> host_data(N);
    for (size_t i = 0; i < N; ++i) {
        host_data[i] = vrtigo::gpu::ComplexF{static_cast<float>(i), 0.0f};
    }
    buffer.upload(host_data.data(), N);

    // Run kernel
    add_complex_kernel<<<1, N>>>(buffer.data(), static_cast<int>(N));
    cudaDeviceSynchronize();

    // Download and verify
    std::vector<vrtigo::gpu::ComplexF> result(N);
    buffer.download(result.data(), N);

    bool passed = true;
    for (size_t i = 0; i < N; ++i) {
        float expected_real = static_cast<float>(i) + 1.0f;
        if (result[i].real() != expected_real || result[i].imag() != 0.0f) {
            std::cerr << "ERROR: Mismatch at index " << i << "\n";
            passed = false;
        }
    }

    if (!passed) {
        return 1;
    }

    std::cout << "Kernel execution verified.\n";
    std::cout << "\nGPU install verification PASSED!\n";
    std::cout << "vrtigo::gpu headers are correctly installed and functional.\n";

    return 0;
}
