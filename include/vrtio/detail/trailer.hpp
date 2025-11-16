#pragma once

#include <cstdint>

namespace vrtio::trailer {

// VITA 49.2 Section 5.1.6 Trailer field bit positions
// The trailer is a 32-bit word with enable/indicator bit pairing

// ============================================================================
// Bits 0-6: Associated Context Packet Count (7 bits, range 0-127)
// Bit 7: E bit (Associated Context Packet Count Enable)
// ============================================================================
inline constexpr uint32_t context_packet_count_shift = 0;
inline constexpr uint32_t context_packet_count_mask = 0x7F; // 7 bits (bits 0-6)
inline constexpr uint32_t e_bit = 7;                        // Enable bit for context packet count

// ============================================================================
// Bits 8-11: User-Defined and Sample Frame Indicators
// ============================================================================
inline constexpr uint32_t user_defined_0_bit = 8;
inline constexpr uint32_t user_defined_1_bit = 9;
inline constexpr uint32_t sample_frame_0_bit = 10;
inline constexpr uint32_t sample_frame_1_bit = 11;

// ============================================================================
// Bits 12-19: Indicator Bits (for 8 named state/event indicators)
// ============================================================================
inline constexpr uint32_t sample_loss_indicator_bit = 12;
inline constexpr uint32_t over_range_indicator_bit = 13;
inline constexpr uint32_t spectral_inversion_indicator_bit = 14;
inline constexpr uint32_t detected_signal_indicator_bit = 15;
inline constexpr uint32_t agc_mgc_indicator_bit = 16;
inline constexpr uint32_t reference_lock_indicator_bit = 17;
inline constexpr uint32_t valid_data_indicator_bit = 18;
inline constexpr uint32_t calibrated_time_indicator_bit = 19;

// ============================================================================
// Bits 24-31: Enable Bits (for 8 named state/event indicators)
// These are separate from the E bit (bit 7) for context packet count
// ============================================================================
inline constexpr uint32_t sample_loss_enable_bit = 24;
inline constexpr uint32_t over_range_enable_bit = 25;
inline constexpr uint32_t spectral_inversion_enable_bit = 26;
inline constexpr uint32_t detected_signal_enable_bit = 27;
inline constexpr uint32_t agc_mgc_enable_bit = 28;
inline constexpr uint32_t reference_lock_enable_bit = 29;
inline constexpr uint32_t valid_data_enable_bit = 30;
inline constexpr uint32_t calibrated_time_enable_bit = 31;

// ============================================================================
// Bit masks for direct manipulation
// ============================================================================

// E bit and context packet count
inline constexpr uint32_t e_bit_mask = 1U << e_bit;

// User-Defined and Sample Frame
inline constexpr uint32_t user_defined_0_mask = 1U << user_defined_0_bit;
inline constexpr uint32_t user_defined_1_mask = 1U << user_defined_1_bit;
inline constexpr uint32_t sample_frame_0_mask = 1U << sample_frame_0_bit;
inline constexpr uint32_t sample_frame_1_mask = 1U << sample_frame_1_bit;

// Indicator bits
inline constexpr uint32_t sample_loss_indicator_mask = 1U << sample_loss_indicator_bit;
inline constexpr uint32_t over_range_indicator_mask = 1U << over_range_indicator_bit;
inline constexpr uint32_t spectral_inversion_indicator_mask = 1U
                                                              << spectral_inversion_indicator_bit;
inline constexpr uint32_t detected_signal_indicator_mask = 1U << detected_signal_indicator_bit;
inline constexpr uint32_t agc_mgc_indicator_mask = 1U << agc_mgc_indicator_bit;
inline constexpr uint32_t reference_lock_indicator_mask = 1U << reference_lock_indicator_bit;
inline constexpr uint32_t valid_data_indicator_mask = 1U << valid_data_indicator_bit;
inline constexpr uint32_t calibrated_time_indicator_mask = 1U << calibrated_time_indicator_bit;

// Enable bits
inline constexpr uint32_t sample_loss_enable_mask = 1U << sample_loss_enable_bit;
inline constexpr uint32_t over_range_enable_mask = 1U << over_range_enable_bit;
inline constexpr uint32_t spectral_inversion_enable_mask = 1U << spectral_inversion_enable_bit;
inline constexpr uint32_t detected_signal_enable_mask = 1U << detected_signal_enable_bit;
inline constexpr uint32_t agc_mgc_enable_mask = 1U << agc_mgc_enable_bit;
inline constexpr uint32_t reference_lock_enable_mask = 1U << reference_lock_enable_bit;
inline constexpr uint32_t valid_data_enable_mask = 1U << valid_data_enable_bit;
inline constexpr uint32_t calibrated_time_enable_mask = 1U << calibrated_time_enable_bit;

// ============================================================================
// Helper functions for bit manipulation
// ============================================================================

/**
 * Extract a single bit value from a trailer word
 * @tparam Bit The bit position (0-31)
 * @param value The trailer word value
 * @return true if bit is set, false otherwise
 */
template <uint32_t Bit>
constexpr bool extract_bit(uint32_t value) noexcept {
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
 * Clear a single bit in a trailer word
 * @tparam Bit The bit position (0-31)
 * @param value The trailer word value
 * @return Updated trailer word with bit cleared
 */
template <uint32_t Bit>
constexpr uint32_t clear_bit(uint32_t value) noexcept {
    static_assert(Bit < 32, "Bit position must be less than 32");
    return value & ~(1U << Bit);
}

/**
 * Extract a multi-bit field from a trailer word
 * @tparam Shift The starting bit position
 * @tparam Mask The bit mask (after shifting)
 * @param value The trailer word value
 * @return The extracted field value
 */
template <uint32_t Shift, uint32_t Mask>
constexpr uint32_t extract_field(uint32_t value) noexcept {
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

/**
 * Clear a multi-bit field in a trailer word
 * @tparam Shift The starting bit position
 * @tparam Mask The bit mask (after shifting)
 * @param value The trailer word value
 * @return Updated trailer word with field cleared
 */
template <uint32_t Shift, uint32_t Mask>
constexpr uint32_t clear_field(uint32_t value) noexcept {
    return value & ~(Mask << Shift);
}

} // namespace vrtio::trailer
