#pragma once

#include "vrtigo/detail/bitfield.hpp"

#include <cstdint>

namespace vrtigo::trailer {

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

} // namespace vrtigo::trailer

// ============================================================================
// BitField API Definitions (for migration)
// ============================================================================
namespace vrtigo::trailer_fields {

using detail::BitField;
using detail::BitFieldLayout;
using detail::BitFlag;

// Context packet count (bits 0-6 = 7-bit field, bit 7 = E bit)
using ContextPacketCount = BitField<uint32_t, 0, 7>; // 7 bits wide (bits 0-6)
using EBit = BitFlag<7>;

// Indicators (bits 12-19)
using UserDefined0Indicator = BitFlag<8>;
using UserDefined1Indicator = BitFlag<9>;
using SampleFrame0Indicator = BitFlag<10>;
using SampleFrame1Indicator = BitFlag<11>;
using SampleLossIndicator = BitFlag<12>;
using OverRangeIndicator = BitFlag<13>;
using SpectralInversionIndicator = BitFlag<14>;
using DetectedSignalIndicator = BitFlag<15>;
using AgcMgcIndicator = BitFlag<16>;
using ReferenceLockIndicator = BitFlag<17>;
using ValidDataIndicator = BitFlag<18>;
using CalibratedTimeIndicator = BitFlag<19>;

// Enable bits (20-31) - one enable bit for each indicator
using UserDefined0Enable = BitFlag<20>;
using UserDefined1Enable = BitFlag<21>;
using SampleFrame0Enable = BitFlag<22>;
using SampleFrame1Enable = BitFlag<23>;
using SampleLossEnable = BitFlag<24>;
using OverRangeEnable = BitFlag<25>;
using SpectralInversionEnable = BitFlag<26>;
using DetectedSignalEnable = BitFlag<27>;
using AgcMgcEnable = BitFlag<28>;
using ReferenceLockEnable = BitFlag<29>;
using ValidDataEnable = BitFlag<30>;
using CalibratedTimeEnable = BitFlag<31>;

// Layout for all trailer fields
using TrailerLayout =
    BitFieldLayout<ContextPacketCount, EBit,

                   UserDefined0Indicator, UserDefined1Indicator, SampleFrame0Indicator,
                   SampleFrame1Indicator, SampleLossIndicator, OverRangeIndicator,
                   SpectralInversionIndicator, DetectedSignalIndicator, AgcMgcIndicator,
                   ReferenceLockIndicator, ValidDataIndicator, CalibratedTimeIndicator,

                   UserDefined0Enable, UserDefined1Enable, SampleFrame0Enable, SampleFrame1Enable,
                   SampleLossEnable, OverRangeEnable, SpectralInversionEnable, DetectedSignalEnable,
                   AgcMgcEnable, ReferenceLockEnable, ValidDataEnable, CalibratedTimeEnable>;

static_assert(TrailerLayout::required_bytes == 4, "Trailer is one VRT word (32 bits)");

} // namespace vrtigo::trailer_fields
