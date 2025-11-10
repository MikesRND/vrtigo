#pragma once

#include <complex>
#include <span>
#include <vector>

#include <cstdint>
#include <cstring>

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

    double energy = 0.0;
    size_t num_samples = payload.size() / 4; // Each IQ sample is 4 bytes (2 * int16)

    for (size_t i = 0; i < num_samples; ++i) {
        size_t offset = i * 4;

        // Read I sample (16-bit, network byte order)
        int16_t i_sample = static_cast<int16_t>((static_cast<uint16_t>(payload[offset]) << 8) |
                                                static_cast<uint16_t>(payload[offset + 1]));

        // Read Q sample (16-bit, network byte order)
        int16_t q_sample = static_cast<int16_t>((static_cast<uint16_t>(payload[offset + 2]) << 8) |
                                                static_cast<uint16_t>(payload[offset + 3]));

        // Accumulate energy: I² + Q²
        energy +=
            static_cast<double>(i_sample) * i_sample + static_cast<double>(q_sample) * q_sample;
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
    std::vector<std::complex<int16_t>> samples;

    if (payload.size() < 4 || payload.size() % 4 != 0) {
        return samples; // Invalid payload size
    }

    size_t num_samples = payload.size() / 4;
    samples.reserve(num_samples);

    for (size_t i = 0; i < num_samples; ++i) {
        size_t offset = i * 4;

        // Read I sample (16-bit, network byte order)
        int16_t i_sample = static_cast<int16_t>((static_cast<uint16_t>(payload[offset]) << 8) |
                                                static_cast<uint16_t>(payload[offset + 1]));

        // Read Q sample (16-bit, network byte order)
        int16_t q_sample = static_cast<int16_t>((static_cast<uint16_t>(payload[offset + 2]) << 8) |
                                                static_cast<uint16_t>(payload[offset + 3]));

        samples.emplace_back(i_sample, q_sample);
    }

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
    return payload.size() / 4;
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

    uint32_t peak = 0;
    size_t num_samples = payload.size() / 4;

    for (size_t i = 0; i < num_samples; ++i) {
        size_t offset = i * 4;

        int16_t i_sample = static_cast<int16_t>((static_cast<uint16_t>(payload[offset]) << 8) |
                                                static_cast<uint16_t>(payload[offset + 1]));

        int16_t q_sample = static_cast<int16_t>((static_cast<uint16_t>(payload[offset + 2]) << 8) |
                                                static_cast<uint16_t>(payload[offset + 3]));

        uint32_t magnitude =
            static_cast<uint32_t>(std::abs(i_sample)) + static_cast<uint32_t>(std::abs(q_sample));

        if (magnitude > peak) {
            peak = magnitude;
        }
    }

    return peak;
}

} // namespace test_utils
