#pragma once

#include <cstdint>

namespace vrtio::trailer {

// VITA 49.2 Trailer field bit positions
// The trailer is a 32-bit word with various status and indicator fields

// Bits 0-6: Associated Context Packets Count (7 bits)
inline constexpr uint32_t context_packets_shift = 0;
inline constexpr uint32_t context_packets_mask = 0x7F; // 7 bits

// Bit 7: Reserved
inline constexpr uint32_t reserved1_bit = 7;

// Bit 8: Reference Lock
inline constexpr uint32_t reference_lock_bit = 8;

// Bit 9: AGC/MGC (Automatic/Manual Gain Control)
inline constexpr uint32_t agc_mgc_bit = 9;

// Bit 10: Detected Signal
inline constexpr uint32_t detected_signal_bit = 10;

// Bit 11: Spectral Inversion
inline constexpr uint32_t spectral_inversion_bit = 11;

// Bit 12: Over-range
inline constexpr uint32_t over_range_bit = 12;

// Bit 13: Sample Loss
inline constexpr uint32_t sample_loss_bit = 13;

// Bits 14-15: Reserved
inline constexpr uint32_t reserved2_shift = 14;
inline constexpr uint32_t reserved2_mask = 0x03; // 2 bits

// Bit 16: Calibrated Time Indicator
inline constexpr uint32_t calibrated_time_bit = 16;

// Bit 17: Valid Data Indicator
inline constexpr uint32_t valid_data_bit = 17;

// Bit 18: Reference Point Indicator
inline constexpr uint32_t reference_point_bit = 18;

// Bit 19: Signal Detected
inline constexpr uint32_t signal_detected_bit = 19;

// Bits 20-31: Reserved
inline constexpr uint32_t reserved3_shift = 20;
inline constexpr uint32_t reserved3_mask = 0xFFF; // 12 bits

// Bit masks for direct manipulation
inline constexpr uint32_t reference_lock_mask = 1U << reference_lock_bit;
inline constexpr uint32_t agc_mgc_mask = 1U << agc_mgc_bit;
inline constexpr uint32_t detected_signal_mask = 1U << detected_signal_bit;
inline constexpr uint32_t spectral_inversion_mask = 1U << spectral_inversion_bit;
inline constexpr uint32_t over_range_mask = 1U << over_range_bit;
inline constexpr uint32_t sample_loss_mask = 1U << sample_loss_bit;
inline constexpr uint32_t calibrated_time_mask = 1U << calibrated_time_bit;
inline constexpr uint32_t valid_data_mask = 1U << valid_data_bit;
inline constexpr uint32_t reference_point_mask = 1U << reference_point_bit;
inline constexpr uint32_t signal_detected_mask = 1U << signal_detected_bit;

// Common status combinations
inline constexpr uint32_t status_all_good = valid_data_mask | calibrated_time_mask;
inline constexpr uint32_t status_errors = over_range_mask | sample_loss_mask;

// Helper functions for bit manipulation

/**
 * Get a single bit value from a trailer word
 * @tparam Bit The bit position (0-31)
 * @param value The trailer word value
 * @return true if bit is set, false otherwise
 */
template <uint32_t Bit>
constexpr bool get_bit(uint32_t value) noexcept {
    static_assert(Bit < 32, "Bit position must be less than 32");
    return (value >> Bit) & 0x01;
}

/**
 * Set a single bit in a trailer word
 * @tparam Bit The bit position (0-31)
 * @param value The trailer word value
 * @param bit_value The bit value to set (true = 1, false = 0)
 * @return Updated trailer word
 */
template <uint32_t Bit>
constexpr uint32_t set_bit(uint32_t value, bool bit_value) noexcept {
    static_assert(Bit < 32, "Bit position must be less than 32");
    return (value & ~(1U << Bit)) | (bit_value ? (1U << Bit) : 0);
}

/**
 * Get a multi-bit field from a trailer word
 * @tparam Shift The starting bit position
 * @tparam Mask The bit mask (after shifting)
 * @param value The trailer word value
 * @return The extracted field value
 */
template <uint32_t Shift, uint32_t Mask>
constexpr uint32_t get_field(uint32_t value) noexcept {
    return (value >> Shift) & Mask;
}

/**
 * Set a multi-bit field in a trailer word
 * @tparam Shift The starting bit position
 * @tparam Mask The bit mask (after shifting)
 * @param value The trailer word value
 * @param field_value The field value to set
 * @return Updated trailer word
 */
template <uint32_t Shift, uint32_t Mask>
constexpr uint32_t set_field(uint32_t value, uint32_t field_value) noexcept {
    return (value & ~(Mask << Shift)) | ((field_value & Mask) << Shift);
}

// Inline helper functions for common operations

/**
 * Check if the trailer indicates valid data
 */
inline constexpr bool is_valid_data(uint32_t trailer) noexcept {
    return get_bit<valid_data_bit>(trailer);
}

/**
 * Check if the trailer indicates calibrated time
 */
inline constexpr bool is_calibrated_time(uint32_t trailer) noexcept {
    return get_bit<calibrated_time_bit>(trailer);
}

/**
 * Check if the trailer indicates any error conditions
 */
inline constexpr bool has_errors(uint32_t trailer) noexcept {
    return (trailer & status_errors) != 0;
}

/**
 * Get the associated context packets count
 */
inline constexpr uint8_t get_context_packets(uint32_t trailer) noexcept {
    return static_cast<uint8_t>(get_field<context_packets_shift, context_packets_mask>(trailer));
}

/**
 * Create a trailer word with common good status
 */
inline constexpr uint32_t create_good_status() noexcept {
    return status_all_good;
}

} // namespace vrtio::trailer