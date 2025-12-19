/**
 * @file integration_test.cu
 * @brief End-to-end integration tests for vrtigo GPU extensions
 *
 * Tests the complete pipeline:
 * 1. Generate samples on GPU
 * 2. Transfer to host (D2H)
 * 3. Write to VRT packet via SampleSpan
 * 4. Parse packet and verify samples match
 *
 * Also tests H2D pipeline:
 * 1. Create VRT packet with samples on host
 * 2. Parse and extract samples
 * 3. Upload to GPU
 * 4. Process on GPU
 * 5. Verify results
 *
 * Requires: CUDA runtime, Google Test
 */

// VRTIGO_ENABLE_CUDA should be defined via compiler flag (-DVRTIGO_ENABLE_CUDA)
#ifndef VRTIGO_ENABLE_CUDA
#define VRTIGO_ENABLE_CUDA
#endif

// Include GPU extensions before core vrtigo for proper SampleTraits setup
#include <vrtigo/vrtigo_gpu.hpp>
#include <vrtigo/sample_span.hpp>
#include <vrtigo.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <vector>

using namespace vrtigo;
using namespace vrtigo::gpu;

// ============================================================================
// Test fixture
// ============================================================================

class GpuIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceSynchronize();
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    }

    void TearDown() override {
        cudaDeviceSynchronize();
    }
};

// ============================================================================
// GPU kernels for sample generation
// ============================================================================

// Generate complex samples with known pattern on GPU
__global__ void generate_complex16_samples_kernel(Complex16* out, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        // Generate pattern: real = idx, imag = -idx
        out[idx] = Complex16{static_cast<int16_t>(idx), static_cast<int16_t>(-static_cast<int16_t>(idx))};
    }
}

// Generate float samples with known pattern
__global__ void generate_float_samples_kernel(float* out, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = static_cast<float>(idx) * 0.5f;
    }
}

// Process samples on GPU (double each value)
__global__ void process_complex16_samples_kernel(Complex16* data, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx].re *= 2;
        data[idx].im *= 2;
    }
}

// Write samples to payload in network byte order (using device sample I/O)
__global__ void write_samples_to_payload_kernel(uint8_t* payload, const Complex16* samples, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        // write_sample performs endian swap (host to network)
        write_sample<Complex16>(payload + idx * sizeof(Complex16), samples[idx]);
    }
}

// Read samples from payload with endian conversion
__global__ void read_samples_from_payload_kernel(Complex16* samples, const uint8_t* payload, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        // read_sample performs endian swap (network to host)
        samples[idx] = read_sample<Complex16>(payload + idx * sizeof(Complex16));
    }
}

// ============================================================================
// D2H Pipeline Tests: GPU -> Device-to-Host -> VRT Packet
// ============================================================================

TEST_F(GpuIntegrationTest, D2H_GenerateTransferWriteVerify) {
    constexpr size_t sample_count = 256;
    constexpr size_t payload_bytes = sample_count * sizeof(Complex16);

    // 1. Generate samples on GPU
    DeviceBuffer<Complex16> d_samples(sample_count);
    ASSERT_TRUE(d_samples.valid());

    int block_size = 256;
    int num_blocks = (sample_count + block_size - 1) / block_size;
    generate_complex16_samples_kernel<<<num_blocks, block_size>>>(d_samples.data(), sample_count);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

    // 2. Transfer to pinned host buffer (D2H)
    PinnedBuffer<Complex16> h_samples(sample_count);
    ASSERT_TRUE(h_samples.valid());

    ASSERT_EQ(d_samples.download(h_samples.data(), sample_count), cudaSuccess);
    cudaDeviceSynchronize();

    // Verify samples on host
    for (size_t i = 0; i < sample_count; ++i) {
        EXPECT_EQ(h_samples[i].re, static_cast<int16_t>(i));
        EXPECT_EQ(h_samples[i].im, static_cast<int16_t>(-static_cast<int16_t>(i)));
    }

    // 3. Write samples to VRT packet payload via SampleSpan
    // Create a buffer for the packet (header + payload)
    std::vector<uint8_t> packet_buf(4 + payload_bytes);  // 1 word header + payload

    // Simple header: signal data packet, size in words
    size_t packet_words = 1 + (payload_bytes / 4);  // header + payload
    uint32_t header = (0x10 << 24) | static_cast<uint32_t>(packet_words);  // Signal data, no stream ID
    // Write header in network byte order
    packet_buf[0] = (header >> 24) & 0xFF;
    packet_buf[1] = (header >> 16) & 0xFF;
    packet_buf[2] = (header >> 8) & 0xFF;
    packet_buf[3] = header & 0xFF;

    // Get payload span
    std::span<uint8_t> payload(packet_buf.data() + 4, payload_bytes);

    // Use SampleSpan to write samples (with endian conversion)
    SampleSpan<Complex16> sample_span(payload);
    EXPECT_EQ(sample_span.count(), sample_count);

    for (size_t i = 0; i < sample_count; ++i) {
        sample_span.set(i, h_samples[i]);
    }

    // 4. Parse packet and verify samples match
    // Read samples back using SampleSpanView
    std::span<const uint8_t> const_payload(packet_buf.data() + 4, payload_bytes);
    SampleSpanView<Complex16> read_span(const_payload);

    for (size_t i = 0; i < sample_count; ++i) {
        Complex16 read_val = read_span[i];
        EXPECT_EQ(read_val.re, static_cast<int16_t>(i))
            << "Mismatch at index " << i << " real component";
        EXPECT_EQ(read_val.im, static_cast<int16_t>(-static_cast<int16_t>(i)))
            << "Mismatch at index " << i << " imag component";
    }
}

TEST_F(GpuIntegrationTest, D2H_BulkWriteRead) {
    constexpr size_t sample_count = 128;
    constexpr size_t payload_bytes = sample_count * sizeof(Complex16);

    // Generate samples on GPU and transfer to host
    DeviceBuffer<Complex16> d_samples(sample_count);
    generate_complex16_samples_kernel<<<1, sample_count>>>(d_samples.data(), sample_count);

    PinnedBuffer<Complex16> h_samples(sample_count);
    d_samples.download(h_samples.data());
    cudaDeviceSynchronize();

    // Create payload buffer
    std::vector<uint8_t> payload_buf(payload_bytes);
    std::span<uint8_t> payload(payload_buf);

    // Bulk write using SampleSpan
    SampleSpan<Complex16> sample_span(payload);
    size_t written = sample_span.write(h_samples.span());
    EXPECT_EQ(written, sample_count);

    // Bulk read back
    std::vector<Complex16> read_buf(sample_count);
    SampleSpanView<Complex16> view{std::span<const uint8_t>{payload_buf}};
    size_t read_count = view.read(std::span<Complex16>{read_buf});
    EXPECT_EQ(read_count, sample_count);

    // Verify
    for (size_t i = 0; i < sample_count; ++i) {
        EXPECT_EQ(read_buf[i].re, h_samples[i].re);
        EXPECT_EQ(read_buf[i].im, h_samples[i].im);
    }
}

// ============================================================================
// H2D Pipeline Tests: VRT Packet -> Host-to-Device -> GPU Processing
// ============================================================================

TEST_F(GpuIntegrationTest, H2D_ParseUploadProcess) {
    constexpr size_t sample_count = 256;
    constexpr size_t payload_bytes = sample_count * sizeof(Complex16);

    // 1. Create VRT packet with known samples on host
    std::vector<uint8_t> packet_buf(4 + payload_bytes);

    // Write header
    size_t packet_words = 1 + (payload_bytes / 4);
    uint32_t header = (0x10 << 24) | static_cast<uint32_t>(packet_words);
    packet_buf[0] = (header >> 24) & 0xFF;
    packet_buf[1] = (header >> 16) & 0xFF;
    packet_buf[2] = (header >> 8) & 0xFF;
    packet_buf[3] = header & 0xFF;

    // Write samples to payload
    std::span<uint8_t> payload(packet_buf.data() + 4, payload_bytes);
    SampleSpan<Complex16> writer(payload);
    for (size_t i = 0; i < sample_count; ++i) {
        writer.set(i, Complex16{static_cast<int16_t>(i * 2), static_cast<int16_t>(i * 3)});
    }

    // 2. Parse packet and extract samples to pinned buffer
    PinnedBuffer<Complex16> h_samples(sample_count);
    std::span<const uint8_t> const_payload(packet_buf.data() + 4, payload_bytes);
    SampleSpanView<Complex16> reader(const_payload);

    for (size_t i = 0; i < sample_count; ++i) {
        h_samples[i] = reader[i];
    }

    // 3. Upload to GPU
    DeviceBuffer<Complex16> d_samples(sample_count);
    ASSERT_EQ(d_samples.upload(h_samples.data(), sample_count), cudaSuccess);

    // 4. Process on GPU (double the values)
    int block_size = 256;
    int num_blocks = (sample_count + block_size - 1) / block_size;
    process_complex16_samples_kernel<<<num_blocks, block_size>>>(d_samples.data(), sample_count);

    // 5. Download and verify results
    d_samples.download(h_samples.data());
    cudaDeviceSynchronize();

    for (size_t i = 0; i < sample_count; ++i) {
        EXPECT_EQ(h_samples[i].re, static_cast<int16_t>(i * 4))
            << "Mismatch at index " << i << " real (expected 4*i)";
        EXPECT_EQ(h_samples[i].im, static_cast<int16_t>(i * 6))
            << "Mismatch at index " << i << " imag (expected 6*i)";
    }
}

// ============================================================================
// GPU-side endian conversion tests
// ============================================================================

TEST_F(GpuIntegrationTest, GpuSideEndianConversion) {
    constexpr size_t sample_count = 64;
    constexpr size_t payload_bytes = sample_count * sizeof(Complex16);

    // Create samples on GPU
    DeviceBuffer<Complex16> d_samples(sample_count);
    generate_complex16_samples_kernel<<<1, sample_count>>>(d_samples.data(), sample_count);

    // Create device-side payload buffer
    DeviceBuffer<uint8_t> d_payload(payload_bytes);

    // Write samples to payload using GPU kernel (with endian swap)
    write_samples_to_payload_kernel<<<1, sample_count>>>(
        d_payload.data(), d_samples.data(), sample_count);

    // Download payload to host
    std::vector<uint8_t> h_payload(payload_bytes);
    d_payload.download(h_payload.data());
    cudaDeviceSynchronize();

    // Verify using host-side SampleSpanView (expects network byte order)
    SampleSpanView<Complex16> view2{std::span<const uint8_t>{h_payload}};
    for (size_t i = 0; i < sample_count; ++i) {
        Complex16 val = view2[i];
        EXPECT_EQ(val.re, static_cast<int16_t>(i));
        EXPECT_EQ(val.im, static_cast<int16_t>(-static_cast<int16_t>(i)));
    }
}

TEST_F(GpuIntegrationTest, GpuSidePayloadParsing) {
    constexpr size_t sample_count = 64;
    constexpr size_t payload_bytes = sample_count * sizeof(Complex16);

    // Create payload on host with known samples
    std::vector<uint8_t> h_payload(payload_bytes);
    SampleSpan<Complex16> writer3{std::span<uint8_t>{h_payload}};
    for (size_t i = 0; i < sample_count; ++i) {
        writer3.set(i, Complex16{static_cast<int16_t>(i * 10), static_cast<int16_t>(i * 20)});
    }

    // Upload payload to GPU
    DeviceBuffer<uint8_t> d_payload(payload_bytes);
    cudaMemcpy(d_payload.data(), h_payload.data(), payload_bytes, cudaMemcpyHostToDevice);

    // Read samples from payload on GPU (with endian conversion)
    DeviceBuffer<Complex16> d_samples(sample_count);
    read_samples_from_payload_kernel<<<1, sample_count>>>(
        d_samples.data(), d_payload.data(), sample_count);

    // Download and verify
    std::vector<Complex16> h_samples(sample_count);
    d_samples.download(h_samples.data());
    cudaDeviceSynchronize();

    for (size_t i = 0; i < sample_count; ++i) {
        EXPECT_EQ(h_samples[i].re, static_cast<int16_t>(i * 10));
        EXPECT_EQ(h_samples[i].im, static_cast<int16_t>(i * 20));
    }
}

// ============================================================================
// Packet parser integration tests
// ============================================================================

TEST_F(GpuIntegrationTest, PacketParserHostDevice) {
    // Create a signal data packet
    uint8_t packet[64] = {};

    // Header: Signal data with stream ID, class ID, TSI=UTC, TSF=sample count
    // Type=1, ClassID=1, TSI=01, TSF=01, Count=4, Size=16 words
    // 0x19540010 = 0001_1001 0101_0100 0000_0000 0001_0000
    //              Type=1    ClassID=1 TSI=01 TSF=01 Count=0100=4  Size=0x10=16
    uint32_t header = 0x19540010;
    packet[0] = (header >> 24) & 0xFF;
    packet[1] = (header >> 16) & 0xFF;
    packet[2] = (header >> 8) & 0xFF;
    packet[3] = header & 0xFF;

    // Parse on host
    DecodedHeader host_hdr = parse_packet(packet);

    EXPECT_EQ(host_hdr.type, gpu::PacketType::signal_data);
    EXPECT_TRUE(host_hdr.has_class_id);
    EXPECT_EQ(host_hdr.tsi, gpu::TsiType::utc);
    EXPECT_EQ(host_hdr.tsf, gpu::TsfType::sample_count);
    EXPECT_EQ(host_hdr.packet_count, 4);
    EXPECT_EQ(host_hdr.size_words, 16);
    EXPECT_TRUE(has_stream_identifier(host_hdr.type));

    // Calculate offsets
    uint16_t payload_off = payload_offset_words(host_hdr);
    // Header(1) + StreamID(1) + ClassID(2) + TSI(1) + TSF(2) = 7
    EXPECT_EQ(payload_off, 7);
}

// ============================================================================
// Raw sample I/O tests
// ============================================================================

TEST_F(GpuIntegrationTest, RawSampleIO_NoEndianSwap) {
    constexpr size_t sample_count = 32;

    // Create samples
    std::vector<Complex16> original(sample_count);
    for (size_t i = 0; i < sample_count; ++i) {
        original[i] = Complex16{static_cast<int16_t>(i), static_cast<int16_t>(i + 100)};
    }

    // Write raw (no endian swap)
    std::vector<uint8_t> buffer(sample_count * sizeof(Complex16));
    write_samples_raw<Complex16>(std::span<uint8_t>(buffer),
                                  std::span<const Complex16>(original));

    // Read raw (no endian swap)
    std::vector<Complex16> read_back(sample_count);
    read_samples_raw<Complex16>(std::span<Complex16>(read_back),
                                 std::span<const uint8_t>(buffer));

    // Should match exactly (no byte order changes)
    for (size_t i = 0; i < sample_count; ++i) {
        EXPECT_EQ(read_back[i].re, original[i].re);
        EXPECT_EQ(read_back[i].im, original[i].im);
    }
}

TEST_F(GpuIntegrationTest, CopyRawPayload) {
    std::vector<uint8_t> src = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<uint8_t> dst(8);

    size_t copied = copy_raw_payload(std::span<uint8_t>(dst), std::span<const uint8_t>(src));

    EXPECT_EQ(copied, 8u);
    EXPECT_EQ(dst, src);
}

// ============================================================================
// Float sample tests
// ============================================================================

TEST_F(GpuIntegrationTest, FloatSamplePipeline) {
    constexpr size_t sample_count = 128;

    // Generate float samples on GPU
    DeviceBuffer<float> d_samples(sample_count);
    generate_float_samples_kernel<<<1, sample_count>>>(d_samples.data(), sample_count);

    // Download to host
    std::vector<float> h_samples(sample_count);
    d_samples.download(h_samples.data());
    cudaDeviceSynchronize();

    // Write to payload
    std::vector<uint8_t> payload(sample_count * sizeof(float));
    SampleSpan<float> float_writer{std::span<uint8_t>{payload}};

    for (size_t i = 0; i < sample_count; ++i) {
        float_writer.set(i, h_samples[i]);
    }

    // Read back and verify
    SampleSpanView<float> float_reader{std::span<const uint8_t>{payload}};
    for (size_t i = 0; i < sample_count; ++i) {
        float expected = static_cast<float>(i) * 0.5f;
        EXPECT_FLOAT_EQ(float_reader[i], expected);
    }
}

// ============================================================================
// POD complex adapter integration
// ============================================================================

TEST_F(GpuIntegrationTest, PodAdapterWithGpuData) {
    constexpr size_t sample_count = 64;

    // Generate samples on GPU
    DeviceBuffer<Complex16> d_samples(sample_count);
    generate_complex16_samples_kernel<<<1, sample_count>>>(d_samples.data(), sample_count);

    // Download to pinned buffer
    PinnedBuffer<Complex16> h_pod_samples(sample_count);
    d_samples.download(h_pod_samples.data());
    cudaDeviceSynchronize();

    // Adapt to std::complex span (zero-copy)
    auto std_span = adapt_from_pod<int16_t>(h_pod_samples.span());
    EXPECT_EQ(std_span.size(), sample_count);

    // Verify values through std::complex interface
    for (size_t i = 0; i < sample_count; ++i) {
        EXPECT_EQ(std_span[i].real(), static_cast<int16_t>(i));
        EXPECT_EQ(std_span[i].imag(), static_cast<int16_t>(-static_cast<int16_t>(i)));
    }

    // Modify through std::complex interface
    std_span[0] = std::complex<int16_t>(999, 888);

    // Verify modification is visible through POD interface
    EXPECT_EQ(h_pod_samples[0].re, 999);
    EXPECT_EQ(h_pod_samples[0].im, 888);
}

// ============================================================================
// Stream-based async operations
// ============================================================================

TEST_F(GpuIntegrationTest, AsyncStreamOperations) {
    constexpr size_t sample_count = 1024;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocate buffers
    DeviceBuffer<Complex16> d_samples(sample_count);
    PinnedBuffer<Complex16> h_samples(sample_count);

    // Initialize host data
    for (size_t i = 0; i < sample_count; ++i) {
        h_samples[i] = Complex16{static_cast<int16_t>(i % 256), static_cast<int16_t>((i + 1) % 256)};
    }

    // Async upload
    ASSERT_EQ(d_samples.upload(h_samples.data(), sample_count, stream), cudaSuccess);

    // Process on GPU (same stream)
    int block_size = 256;
    int num_blocks = (sample_count + block_size - 1) / block_size;
    process_complex16_samples_kernel<<<num_blocks, block_size, 0, stream>>>(
        d_samples.data(), sample_count);

    // Async download
    PinnedBuffer<Complex16> h_result(sample_count);
    ASSERT_EQ(d_samples.download(h_result.data(), sample_count, stream), cudaSuccess);

    // Wait for completion
    cudaStreamSynchronize(stream);

    // Verify (values should be doubled)
    for (size_t i = 0; i < sample_count; ++i) {
        EXPECT_EQ(h_result[i].re, static_cast<int16_t>((i % 256) * 2));
        EXPECT_EQ(h_result[i].im, static_cast<int16_t>(((i + 1) % 256) * 2));
    }

    cudaStreamDestroy(stream);
}

// ============================================================================
// Error handling tests
// ============================================================================

TEST_F(GpuIntegrationTest, BufferValidityChecks) {
    // Empty buffer
    DeviceBuffer<Complex16> empty_buf;
    EXPECT_FALSE(empty_buf.valid());
    EXPECT_EQ(empty_buf.size(), 0u);
    EXPECT_EQ(empty_buf.data(), nullptr);

    // Upload to empty buffer should fail
    Complex16 sample{1, 2};
    EXPECT_NE(empty_buf.upload(&sample, 1), cudaSuccess);
}

TEST_F(GpuIntegrationTest, BufferResizing) {
    DeviceBuffer<Complex16> buf(100);
    EXPECT_EQ(buf.size(), 100u);

    // Resize up
    EXPECT_EQ(buf.resize(200), cudaSuccess);
    EXPECT_EQ(buf.size(), 200u);
    EXPECT_TRUE(buf.valid());

    // Resize down
    EXPECT_EQ(buf.resize(50), cudaSuccess);
    EXPECT_EQ(buf.size(), 50u);

    // Resize to zero
    EXPECT_EQ(buf.resize(0), cudaSuccess);
    EXPECT_EQ(buf.size(), 0u);
    EXPECT_FALSE(buf.valid());
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
