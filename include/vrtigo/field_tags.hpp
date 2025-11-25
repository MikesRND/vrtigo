#pragma once

#include <cstdint>

namespace vrtigo::field {

/// Minimal tag type - self-contained, no dependencies
/// Tags store only CIF word and bit position as compile-time constants
template <uint8_t CifWord, uint8_t BitPos>
struct field_tag_t {
    static constexpr uint8_t cif = CifWord;
    static constexpr uint8_t bit = BitPos;
};

// ============================================================================
// CIF0 Field Tags (VITA 49.2 Table 9.4-1)
// ============================================================================

// Bit 0: Reserved
// Bit 1: CIF1 Enable (control bit)
// Bit 2: CIF2 Enable (control bit)
// Bits 3-8: Reserved or unsupported

inline constexpr field_tag_t<0, 9> context_association_lists;
inline constexpr field_tag_t<0, 10> gps_ascii;
inline constexpr field_tag_t<0, 11> ephemeris_ref_id;
inline constexpr field_tag_t<0, 12> relative_ephemeris;
inline constexpr field_tag_t<0, 13> ecef_ephemeris;
inline constexpr field_tag_t<0, 14> formatted_gps_ins;
inline constexpr field_tag_t<0, 15> data_payload_format;
inline constexpr field_tag_t<0, 16> state_event_indicators;
inline constexpr field_tag_t<0, 17> device_id;
inline constexpr field_tag_t<0, 18> temperature;
inline constexpr field_tag_t<0, 19> timestamp_calibration_time;
inline constexpr field_tag_t<0, 20> timestamp_adjustment;
inline constexpr field_tag_t<0, 21> sample_rate;
inline constexpr field_tag_t<0, 22> over_range_count;
inline constexpr field_tag_t<0, 23> gain;
inline constexpr field_tag_t<0, 24> reference_level;
inline constexpr field_tag_t<0, 25> if_band_offset;
inline constexpr field_tag_t<0, 26> rf_frequency_offset;
inline constexpr field_tag_t<0, 27> rf_reference_frequency;
inline constexpr field_tag_t<0, 28> if_reference_frequency;
inline constexpr field_tag_t<0, 29> bandwidth;
inline constexpr field_tag_t<0, 30> reference_point_id;
// CIF0 bit 31: Change Indicator - accessed via packet.change_indicator()
// No field tag provided - bit is in CIF0 word itself, not separate data

// Convenient aliases
inline constexpr auto& context_assoc_lists = context_association_lists;
inline constexpr auto& formatted_gps = formatted_gps_ins;

// ============================================================================
// CIF1 Field Tags (VITA 49.2 Table 9.5-1)
// ============================================================================

// Bit 0: Reserved
inline constexpr field_tag_t<1, 1> buffer_size;
inline constexpr field_tag_t<1, 2> version_build_code;
inline constexpr field_tag_t<1, 3> v49_spec_compliance;
inline constexpr field_tag_t<1, 4> health_status;
inline constexpr field_tag_t<1, 5> discrete_io_64;
inline constexpr field_tag_t<1, 6> discrete_io_32;
// Bit 7: Index List (unsupported - variable)
// Bit 8: Reserved
// Bit 9: Sector Scan (unsupported - variable)
inline constexpr field_tag_t<1, 10> spectrum;
// Bit 11: Array of CIFs (unsupported - variable)
// Bit 12: Reserved
inline constexpr field_tag_t<1, 13> aux_bandwidth;
inline constexpr field_tag_t<1, 14> aux_gain;
inline constexpr field_tag_t<1, 15> aux_frequency;
inline constexpr field_tag_t<1, 16> snr_noise_figure;
inline constexpr field_tag_t<1, 17> intercept_points;
inline constexpr field_tag_t<1, 18> compression_point;
inline constexpr field_tag_t<1, 19> threshold;
inline constexpr field_tag_t<1, 20> eb_no;
// Bits 21-23: Reserved
inline constexpr field_tag_t<1, 24> range;
inline constexpr field_tag_t<1, 25> beam_width;
inline constexpr field_tag_t<1, 26> spatial_reference_type;
inline constexpr field_tag_t<1, 27> spatial_scan_type;
// Bit 28: 3D Pointing Structure (unsupported - variable)
inline constexpr field_tag_t<1, 29> pointing_vector_3d_single;
inline constexpr field_tag_t<1, 30> polarization;
inline constexpr field_tag_t<1, 31> phase_offset;

// Aliases
inline constexpr auto& snr = snr_noise_figure;

// ============================================================================
// CIF2 Field Tags (VITA 49.2 Table 9.6-1)
// ============================================================================

// Bits 0-2: Reserved
inline constexpr field_tag_t<2, 3> rf_footprint_range;
inline constexpr field_tag_t<2, 4> rf_footprint;
inline constexpr field_tag_t<2, 5> communication_priority;
inline constexpr field_tag_t<2, 6> function_priority;
inline constexpr field_tag_t<2, 7> event_id;
inline constexpr field_tag_t<2, 8> mode_id;
inline constexpr field_tag_t<2, 9> function_id;
inline constexpr field_tag_t<2, 10> modulation_type;
inline constexpr field_tag_t<2, 11> modulation_class;
inline constexpr field_tag_t<2, 12> ems_device_instance;
inline constexpr field_tag_t<2, 13> ems_device_type;
inline constexpr field_tag_t<2, 14> ems_device_class;
inline constexpr field_tag_t<2, 15> platform_display;
inline constexpr field_tag_t<2, 16> platform_instance;
inline constexpr field_tag_t<2, 17> platform_class;
inline constexpr field_tag_t<2, 18> operator_id;
inline constexpr field_tag_t<2, 19> country_code;
inline constexpr field_tag_t<2, 20> track_id;
inline constexpr field_tag_t<2, 21> information_source;
inline constexpr field_tag_t<2, 22> controller_uuid;
inline constexpr field_tag_t<2, 23> controller_id;
inline constexpr field_tag_t<2, 24> controllee_uuid;
inline constexpr field_tag_t<2, 25> controllee_id;
inline constexpr field_tag_t<2, 26> cited_message_id;
inline constexpr field_tag_t<2, 27> child_stream_id;
inline constexpr field_tag_t<2, 28> parent_stream_id;
inline constexpr field_tag_t<2, 29> sibling_stream_id;
inline constexpr field_tag_t<2, 30> cited_sid;
inline constexpr field_tag_t<2, 31> bind;

// ============================================================================
// CIF3 Field Tags (VITA 49.2 Table 9.7-1 - Temporal and Environmental)
// ============================================================================

// Bit 0: Reserved
inline constexpr field_tag_t<3, 1> network_id;
inline constexpr field_tag_t<3, 2> tropospheric_state;
inline constexpr field_tag_t<3, 3> sea_swell_state;
inline constexpr field_tag_t<3, 4> barometric_pressure;
inline constexpr field_tag_t<3, 5> humidity;
inline constexpr field_tag_t<3, 6> sea_ground_temperature;
inline constexpr field_tag_t<3, 7> air_temperature;
// Bits 8-15: Reserved
// Bits 16-17: Age/Shelf Life - TSI/TSF dependent (deferred)
// Bits 18-19: Reserved
inline constexpr field_tag_t<3, 20> jitter;
inline constexpr field_tag_t<3, 21> dwell;
inline constexpr field_tag_t<3, 22> duration;
inline constexpr field_tag_t<3, 23> period;
inline constexpr field_tag_t<3, 24> pulse_width;
inline constexpr field_tag_t<3, 25> offset_time;
inline constexpr field_tag_t<3, 26> fall_time;
inline constexpr field_tag_t<3, 27> rise_time;
// Bits 28-29: Reserved
inline constexpr field_tag_t<3, 30> timestamp_skew;
inline constexpr field_tag_t<3, 31> timestamp_details;

// Convenient aliases
inline constexpr auto& sea_and_swell_state = sea_swell_state;

} // namespace vrtigo::field
