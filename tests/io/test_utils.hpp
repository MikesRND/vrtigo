#pragma once

#include <complex>
#include <span>
#include <vector>

#include <cstdint>
#include <cstring>
#include <vrtigo/sample_span.hpp>

namespace test_utils {

/**
 * @brief Compute signal energy from IQ sample payload
 *
 * Assumes payload contains interleaved 16-bit I/Q samples in network byte order.
 * Energy is computed as sum of (I² + Q²) for all samples.
 *
 * @param payload Raw packet payload bytes
 * @return Total signal energy
 */
inline double compute_signal_energy(std::span<const uint8_t> payload) {
    if (payload.size() < 4 || payload.size() % 4 != 0) {
        return 0.0; // Invalid payload size
    }

    vrtigo::SampleSpanView<std::complex<int16_t>> samples(payload);
    double energy = 0.0;

    for (size_t i = 0; i < samples.count(); ++i) {
        auto sample = samples[i];
        energy += static_cast<double>(sample.real()) * sample.real() +
                  static_cast<double>(sample.imag()) * sample.imag();
    }

    return energy;
}

/**
 * @brief Compute simple checksum of data
 *
 * Computes a simple 32-bit checksum by summing all 32-bit words.
 * Useful for detecting data corruption in test scenarios.
 *
 * @param data Data bytes to checksum
 * @return 32-bit checksum value
 */
inline uint32_t compute_checksum(std::span<const uint8_t> data) {
    uint32_t checksum = 0;
    size_t num_words = data.size() / 4;

    for (size_t i = 0; i < num_words; ++i) {
        size_t offset = i * 4;
        uint32_t word = (static_cast<uint32_t>(data[offset]) << 24) |
                        (static_cast<uint32_t>(data[offset + 1]) << 16) |
                        (static_cast<uint32_t>(data[offset + 2]) << 8) |
                        static_cast<uint32_t>(data[offset + 3]);
        checksum += word;
    }

    // Handle remaining bytes
    size_t remaining = data.size() % 4;
    if (remaining > 0) {
        uint32_t partial = 0;
        for (size_t i = 0; i < remaining; ++i) {
            partial |= static_cast<uint32_t>(data[num_words * 4 + i]) << ((3 - i) * 8);
        }
        checksum += partial;
    }

    return checksum;
}

/**
 * @brief Extract IQ samples from payload as complex numbers
 *
 * Converts raw payload bytes into a vector of complex<int16_t> samples.
 * Assumes interleaved 16-bit I/Q samples in network byte order.
 *
 * @param payload Raw packet payload bytes
 * @return Vector of complex samples (empty if invalid payload)
 */
inline std::vector<std::complex<int16_t>> extract_iq_samples(std::span<const uint8_t> payload) {
    if (payload.size() < 4 || payload.size() % 4 != 0) {
        return {}; // Invalid payload size
    }

    vrtigo::SampleSpanView<std::complex<int16_t>> view(payload);
    std::vector<std::complex<int16_t>> samples(view.count());
    view.read(samples);
    return samples;
}

/**
 * @brief Check if all samples in payload are non-zero
 *
 * Useful for validating that signal data is present and not all zeros.
 *
 * @param payload Raw packet payload bytes
 * @return true if at least one non-zero sample found
 */
inline bool has_nonzero_samples(std::span<const uint8_t> payload) {
    for (auto byte : payload) {
        if (byte != 0) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Count number of IQ samples in payload
 *
 * @param payload Raw packet payload bytes
 * @return Number of complete IQ sample pairs
 */
inline size_t count_iq_samples(std::span<const uint8_t> payload) {
    return vrtigo::SampleSpanView<std::complex<int16_t>>(payload).count();
}

/**
 * @brief Compute peak sample magnitude from payload
 *
 * Finds the maximum |I| + |Q| magnitude across all samples.
 *
 * @param payload Raw packet payload bytes
 * @return Peak magnitude, or 0 if invalid payload
 */
inline uint32_t compute_peak_magnitude(std::span<const uint8_t> payload) {
    if (payload.size() < 4 || payload.size() % 4 != 0) {
        return 0;
    }

    vrtigo::SampleSpanView<std::complex<int16_t>> samples(payload);
    uint32_t peak = 0;

    for (size_t i = 0; i < samples.count(); ++i) {
        auto sample = samples[i];
        uint32_t magnitude = static_cast<uint32_t>(std::abs(sample.real())) +
                             static_cast<uint32_t>(std::abs(sample.imag()));
        if (magnitude > peak) {
            peak = magnitude;
        }
    }

    return peak;
}

/**
 * @brief Create a minimal valid VRT data packet for testing
 *
 * Creates a simple SignalData packet (type 1) with stream ID and one payload word.
 * All values are in network byte order (big-endian), ready for UDP transmission.
 *
 * Packet structure:
 * - Header word (4 bytes): Type 1, no class ID, no trailer, size 3 words
 * - Stream ID (4 bytes): Configurable
 * - Payload (4 bytes): 0xDEADBEEF
 *
 * @param stream_id Stream identifier to use
 * @return Vector of bytes in network byte order
 */
inline std::vector<uint8_t> create_minimal_vrt_packet(uint32_t stream_id = 0x12345678) {
    std::vector<uint8_t> packet(12); // 3 words = 12 bytes

    // Header word: Type 1 (SignalData with stream ID), size 3, no optional fields
    // Bits 31-28: Type = 0001 (SignalData)
    // Bit 27: Class ID = 0
    // Bit 26: Trailer = 0
    // Bits 15-0: Size = 3
    uint32_t header = 0x10000003;

    // Write header (network byte order = big endian)
    packet[0] = (header >> 24) & 0xFF;
    packet[1] = (header >> 16) & 0xFF;
    packet[2] = (header >> 8) & 0xFF;
    packet[3] = header & 0xFF;

    // Write stream ID (network byte order)
    packet[4] = (stream_id >> 24) & 0xFF;
    packet[5] = (stream_id >> 16) & 0xFF;
    packet[6] = (stream_id >> 8) & 0xFF;
    packet[7] = stream_id & 0xFF;

    // Write payload word (0xDEADBEEF in network byte order)
    packet[8] = 0xDE;
    packet[9] = 0xAD;
    packet[10] = 0xBE;
    packet[11] = 0xEF;

    return packet;
}

/**
 * @brief Create a VRT data packet with custom payload
 *
 * Creates a SignalData packet with configurable payload size.
 *
 * @param stream_id Stream identifier
 * @param payload_words Number of 32-bit payload words (1-65533)
 * @return Vector of bytes in network byte order
 */
inline std::vector<uint8_t> create_vrt_packet_with_payload(uint32_t stream_id,
                                                           uint16_t payload_words) {
    if (payload_words == 0 || payload_words > 65533) {
        return {}; // Invalid size
    }

    uint16_t packet_size = 2 + payload_words; // header + stream_id + payload
    std::vector<uint8_t> packet(packet_size * 4);

    // Header
    uint32_t header = 0x10000000 | packet_size; // Type 1, size
    packet[0] = (header >> 24) & 0xFF;
    packet[1] = (header >> 16) & 0xFF;
    packet[2] = (header >> 8) & 0xFF;
    packet[3] = header & 0xFF;

    // Stream ID
    packet[4] = (stream_id >> 24) & 0xFF;
    packet[5] = (stream_id >> 16) & 0xFF;
    packet[6] = (stream_id >> 8) & 0xFF;
    packet[7] = stream_id & 0xFF;

    // Payload (fill with pattern)
    for (uint16_t i = 0; i < payload_words; ++i) {
        uint32_t payload_word = 0xAA000000 | i; // Pattern: 0xAA0000xx
        size_t offset = 8 + (i * 4);
        packet[offset + 0] = (payload_word >> 24) & 0xFF;
        packet[offset + 1] = (payload_word >> 16) & 0xFF;
        packet[offset + 2] = (payload_word >> 8) & 0xFF;
        packet[offset + 3] = payload_word & 0xFF;
    }

    return packet;
}

} // namespace test_utils
