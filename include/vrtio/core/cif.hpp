#pragma once

#include <climits>
#include <cstdint>
#include <cstring>

#include "detail/buffer_io.hpp"
#include "endian.hpp"

namespace vrtio {
namespace cif {

// FieldInfo struct - describes each CIF field
struct FieldInfo {
    uint8_t size_words; // Size in 32-bit words (0 for variable)
    bool is_variable;   // True for variable-length fields
    bool is_supported;  // True if VRTIO supports this field
    const char* name;   // Field name for documentation
};

// CIF enable bit positions in CIF0
constexpr uint32_t CIF1_ENABLE_BIT = 1;
constexpr uint32_t CIF2_ENABLE_BIT = 2;
constexpr uint32_t CIF3_ENABLE_BIT = 3;
constexpr uint32_t CIF_ENABLE_MASK = 0x0E; // Bits 1, 2, 3

// Variable field bit positions in CIF0
constexpr uint32_t CONTEXT_ASSOC_BIT = 9;
constexpr uint32_t GPS_ASCII_BIT = 10;

// Buffer access helpers (redirect to shared implementation)
inline uint32_t read_u32_safe(const uint8_t* buffer, size_t offset) noexcept {
    return detail::read_u32(buffer, offset);
}

inline uint64_t read_u64_safe(const uint8_t* buffer, size_t offset) noexcept {
    return detail::read_u64(buffer, offset);
}

inline void write_u32_safe(uint8_t* buffer, size_t offset, uint32_t value) noexcept {
    detail::write_u32(buffer, offset, value);
}

inline void write_u64_safe(uint8_t* buffer, size_t offset, uint64_t value) noexcept {
    detail::write_u64(buffer, offset, value);
}

// COMPLETE CIF0 field table - all 32 bits with verified sizes from VITA 49.2
constexpr FieldInfo CIF0_FIELDS[32] = {
    //  size, var,  supp, name
    {0, false, false, "reserved"},                  // bit 0
    {0, false, true, "cif1_enable"},                // bit 1 - control bit, supported
    {0, false, true, "cif2_enable"},                // bit 2 - control bit, supported
    {0, false, true, "cif3_enable"},                // bit 3 - control bit, supported
    {0, false, false, "reserved"},                  // bit 4
    {0, false, false, "reserved"},                  // bit 5
    {0, false, false, "reserved"},                  // bit 6
    {0, false, false, "field_attributes"},          // bit 7 - unsupported
    {0, false, false, "reserved"},                  // bit 8
    {0, true, true, "context_association_lists"},   // bit 9 - VARIABLE, SUPPORTED
    {0, true, true, "gps_ascii"},                   // bit 10 - VARIABLE, SUPPORTED
    {1, false, true, "ephemeris_ref_id"},           // bit 11
    {13, false, true, "relative_ephemeris"},        // bit 12
    {13, false, true, "ecef_ephemeris"},            // bit 13
    {11, false, true, "formatted_gps_ins"},         // bit 14
    {2, false, true, "data_payload_format"},        // bit 15
    {1, false, true, "state_event_indicators"},     // bit 16
    {2, false, true, "device_id"},                  // bit 17
    {1, false, true, "temperature"},                // bit 18
    {1, false, true, "timestamp_calibration_time"}, // bit 19
    {2, false, true, "timestamp_adjustment"},       // bit 20
    {2, false, true, "sample_rate"},                // bit 21
    {1, false, true, "over_range_count"},           // bit 22
    {1, false, true, "gain"},                       // bit 23
    {1, false, true, "reference_level"},            // bit 24
    {2, false, true, "if_band_offset"},             // bit 25
    {2, false, true, "rf_frequency_offset"},        // bit 26
    {2, false, true, "rf_reference_frequency"},     // bit 27
    {2, false, true, "if_reference_frequency"},     // bit 28
    {2, false, true, "bandwidth"},                  // bit 29
    {1, false, true, "reference_point_id"},         // bit 30
    {0, false, true, "change_indicator"}            // bit 31 - flag only
};

// COMPLETE CIF1 field table with verified sizes
constexpr FieldInfo CIF1_FIELDS[32] = {
    //  size, var,  supp, name
    {0, false, false, "reserved"},                                // bit 0
    {1, false, true, "buffer_size"},                              // bit 1
    {1, false, true, "version_build_code"},                       // bit 2
    {1, false, true, "v49_spec_compliance"},                      // bit 3
    {1, false, true, "health_status"},                            // bit 4
    {2, false, true, "discrete_io_64"},                           // bit 5
    {1, false, true, "discrete_io_32"},                           // bit 6
    {0, true, false, "index_list_unsupported"},                   // bit 7 - VARIABLE
    {0, false, false, "reserved"},                                // bit 8
    {0, true, false, "sector_scan_unsupported"},                  // bit 9 - VARIABLE
    {13, false, true, "spectrum"},                                // bit 10
    {0, true, false, "array_of_cifs_unsupported"},                // bit 11 - VARIABLE
    {0, false, false, "reserved"},                                // bit 12
    {2, false, true, "aux_bandwidth"},                            // bit 13
    {1, false, true, "aux_gain"},                                 // bit 14
    {2, false, true, "aux_frequency"},                            // bit 15
    {1, false, true, "snr_noise_figure"},                         // bit 16
    {1, false, true, "intercept_points"},                         // bit 17
    {1, false, true, "compression_point"},                        // bit 18
    {1, false, true, "threshold"},                                // bit 19
    {1, false, true, "eb_no"},                                    // bit 20
    {0, false, false, "reserved"},                                // bit 21
    {0, false, false, "reserved"},                                // bit 22
    {0, false, false, "reserved"},                                // bit 23
    {1, false, true, "range"},                                    // bit 24
    {1, false, true, "beam_width"},                               // bit 25
    {1, false, true, "spatial_reference_type"},                   // bit 26
    {1, false, true, "spatial_scan_type"},                        // bit 27
    {0, true, false, "pointing_vector_3d_structure_unsupported"}, // bit 28 - VARIABLE
    {1, false, true, "pointing_vector_3d_single"},                // bit 29
    {1, false, true, "polarization"},                             // bit 30
    {1, false, true, "phase_offset"}                              // bit 31
};

// COMPLETE CIF2 field table (all supported where not reserved)
constexpr FieldInfo CIF2_FIELDS[32] = {
    //  size, var,  supp, name
    {0, false, false, "reserved"},              // bit 0
    {0, false, false, "reserved"},              // bit 1
    {0, false, false, "reserved"},              // bit 2
    {1, false, true, "rf_footprint_range"},     // bit 3
    {1, false, true, "rf_footprint"},           // bit 4
    {1, false, true, "communication_priority"}, // bit 5
    {1, false, true, "function_priority"},      // bit 6
    {1, false, true, "event_id"},               // bit 7
    {1, false, true, "mode_id"},                // bit 8
    {1, false, true, "function_id"},            // bit 9
    {1, false, true, "modulation_type"},        // bit 10
    {1, false, true, "modulation_class"},       // bit 11
    {1, false, true, "ems_device_instance"},    // bit 12
    {1, false, true, "ems_device_type"},        // bit 13
    {1, false, true, "ems_device_class"},       // bit 14
    {1, false, true, "platform_display"},       // bit 15
    {1, false, true, "platform_instance"},      // bit 16
    {1, false, true, "platform_class"},         // bit 17
    {1, false, true, "operator_id"},            // bit 18
    {1, false, true, "country_code"},           // bit 19
    {1, false, true, "track_id"},               // bit 20
    {1, false, true, "information_source"},     // bit 21
    {4, false, true, "controller_uuid"},        // bit 22
    {1, false, true, "controller_id"},          // bit 23
    {4, false, true, "controllee_uuid"},        // bit 24
    {1, false, true, "controllee_id"},          // bit 25
    {1, false, true, "cited_message_id"},       // bit 26
    {1, false, true, "child_stream_id"},        // bit 27
    {1, false, true, "parent_stream_id"},       // bit 28
    {1, false, true, "sibling_stream_id"},      // bit 29
    {1, false, true, "cited_sid"},              // bit 30
    {1, false, true, "bind"}                    // bit 31
};

// COMPLETE CIF3 field table - Temporal and Environmental Fields
constexpr FieldInfo CIF3_FIELDS[32] = {
    //  size, var,  supp, name
    {0, false, false, "reserved"},              // bit 0
    {1, false, true, "network_id"},             // bit 1
    {1, false, true, "tropospheric_state"},     // bit 2
    {1, false, true, "sea_swell_state"},        // bit 3
    {1, false, true, "barometric_pressure"},    // bit 4
    {1, false, true, "humidity"},               // bit 5
    {1, false, true, "sea_ground_temperature"}, // bit 6
    {1, false, true, "air_temperature"},        // bit 7
    {0, false, false, "reserved"},              // bit 8
    {0, false, false, "reserved"},              // bit 9
    {0, false, false, "reserved"},              // bit 10
    {0, false, false, "reserved"},              // bit 11
    {0, false, false, "reserved"},              // bit 12
    {0, false, false, "reserved"},              // bit 13
    {0, false, false, "reserved"},              // bit 14
    {0, false, false, "reserved"},              // bit 15
    {0, false, false, "shelf_life"},            // bit 16 - deferred (TSI/TSF dependent)
    {0, false, false, "age"},                   // bit 17 - deferred (TSI/TSF dependent)
    {0, false, false, "reserved"},              // bit 18
    {0, false, false, "reserved"},              // bit 19
    {2, false, true, "jitter"},                 // bit 20
    {2, false, true, "dwell"},                  // bit 21
    {2, false, true, "duration"},               // bit 22
    {2, false, true, "period"},                 // bit 23
    {2, false, true, "pulse_width"},            // bit 24
    {2, false, true, "offset_time"},            // bit 25
    {2, false, true, "fall_time"},              // bit 26
    {2, false, true, "rise_time"},              // bit 27
    {0, false, false, "reserved"},              // bit 28
    {0, false, false, "reserved"},              // bit 29
    {2, false, true, "timestamp_skew"},         // bit 30
    {2, false, true, "timestamp_details"}       // bit 31
};

// Build SUPPORTED masks from the tables
// Includes CIF1/CIF2/CIF3 enable control bits (1, 2, 3) since they are valid in CIF0
constexpr uint32_t CIF0_SUPPORTED_MASK = (1U << 1) |  // CIF1 Enable (control bit)
                                         (1U << 2) |  // CIF2 Enable (control bit)
                                         (1U << 3) |  // CIF3 Enable (control bit)
                                         (1U << 9) |  // Context Association Lists
                                         (1U << 10) | // GPS ASCII
                                         (1U << 11) | // Ephemeris Ref ID
                                         (1U << 12) | // Relative Ephemeris
                                         (1U << 13) | // ECEF Ephemeris
                                         (1U << 14) | // Formatted GPS/INS
                                         (1U << 15) | // Data Payload Format
                                         (1U << 16) | // State/Event Indicators
                                         (1U << 17) | // Device ID
                                         (1U << 18) | // Temperature
                                         (1U << 19) | // Timestamp Cal Time
                                         (1U << 20) | // Timestamp Adjustment
                                         (1U << 21) | // Sample Rate
                                         (1U << 22) | // Over-Range Count
                                         (1U << 23) | // Gain
                                         (1U << 24) | // Reference Level
                                         (1U << 25) | // IF Band Offset
                                         (1U << 26) | // RF Frequency Offset
                                         (1U << 27) | // RF Reference Freq
                                         (1U << 28) | // IF Reference Freq
                                         (1U << 29) | // Bandwidth
                                         (1U << 30) | // Reference Point ID
                                         (1U << 31);  // Change Indicator

constexpr uint32_t CIF1_SUPPORTED_MASK = (1U << 1) |  // Buffer Size
                                         (1U << 2) |  // Version and Build Code
                                         (1U << 3) |  // V49 Spec Compliance
                                         (1U << 4) |  // Health Status
                                         (1U << 5) |  // Discrete I/O (64 bit)
                                         (1U << 6) |  // Discrete I/O (32 bit)
                                         (1U << 10) | // Spectrum
                                         (1U << 13) | // Aux Bandwidth
                                         (1U << 14) | // Aux Gain
                                         (1U << 15) | // Aux Frequency
                                         (1U << 16) | // SNR/Noise Figure
                                         (1U << 17) | // Intercept Points
                                         (1U << 18) | // Compression Point
                                         (1U << 19) | // Threshold
                                         (1U << 20) | // Eb/No BER
                                         (1U << 24) | // Range
                                         (1U << 25) | // Beam Width
                                         (1U << 26) | // Spatial Ref Type
                                         (1U << 27) | // Spatial Scan Type
                                         (1U << 29) | // 3D Pointing Single
                                         (1U << 30) | // Polarization
                                         (1U << 31);  // Phase Offset

constexpr uint32_t CIF2_SUPPORTED_MASK = (1U << 3) |  // RF Footprint Range
                                         (1U << 4) |  // RF Footprint
                                         (1U << 5) |  // Comm Priority ID
                                         (1U << 6) |  // Function Priority ID
                                         (1U << 7) |  // Event ID
                                         (1U << 8) |  // Mode ID
                                         (1U << 9) |  // Function ID
                                         (1U << 10) | // Modulation Type
                                         (1U << 11) | // Modulation Class
                                         (1U << 12) | // EMS Device Instance
                                         (1U << 13) | // EMS Device Type
                                         (1U << 14) | // EMS Device Class
                                         (1U << 15) | // Platform Display
                                         (1U << 16) | // Platform Instance
                                         (1U << 17) | // Platform Class
                                         (1U << 18) | // Operator
                                         (1U << 19) | // Country Code
                                         (1U << 20) | // Track ID
                                         (1U << 21) | // Information Source
                                         (1U << 22) | // Controller UUID
                                         (1U << 23) | // Controller ID
                                         (1U << 24) | // Controllee UUID
                                         (1U << 25) | // Controllee ID
                                         (1U << 26) | // Cited Message ID
                                         (1U << 27) | // Child(ren) SID
                                         (1U << 28) | // Parent(s) SID
                                         (1U << 29) | // Sibling(s) SID
                                         (1U << 30) | // Cited SID
                                         (1U << 31);  // Bind

constexpr uint32_t CIF3_SUPPORTED_MASK = (1U << 1) |  // Network ID
                                         (1U << 2) |  // Tropospheric State
                                         (1U << 3) |  // Sea and Swell State
                                         (1U << 4) |  // Barometric Pressure
                                         (1U << 5) |  // Humidity
                                         (1U << 6) |  // Sea/Ground Temperature
                                         (1U << 7) |  // Air Temperature
                                         (1U << 20) | // Jitter
                                         (1U << 21) | // Dwell
                                         (1U << 22) | // Duration
                                         (1U << 23) | // Period
                                         (1U << 24) | // Pulse Width
                                         (1U << 25) | // Offset Time
                                         (1U << 26) | // Fall Time
                                         (1U << 27) | // Rise Time
                                         (1U << 30) | // Timestamp Skew
                                         (1U << 31);  // Timestamp Details

// Variable fields that need special handling at compile-time
constexpr uint32_t CIF0_VARIABLE_MASK = (1U << 10) | // GPS ASCII
                                        (1U << 9);   // Context Association Lists

// For compile-time: supported minus variable fields
constexpr uint32_t CIF0_COMPILETIME_SUPPORTED_MASK = CIF0_SUPPORTED_MASK & ~CIF0_VARIABLE_MASK;

// Helper to build supported mask at compile time
template <size_t N>
constexpr uint32_t build_supported_mask(const FieldInfo (&table)[N]) {
    uint32_t mask = 0;
    for (size_t i = 0; i < N; ++i) {
        if (table[i].is_supported) {
            mask |= (1U << i);
        }
    }
    return mask;
}

// Verify our hand-coded masks match the tables
static_assert(build_supported_mask(CIF0_FIELDS) == CIF0_SUPPORTED_MASK,
              "CIF0_SUPPORTED_MASK doesn't match table");
static_assert(build_supported_mask(CIF1_FIELDS) == CIF1_SUPPORTED_MASK,
              "CIF1_SUPPORTED_MASK doesn't match table");
static_assert(build_supported_mask(CIF2_FIELDS) == CIF2_SUPPORTED_MASK,
              "CIF2_SUPPORTED_MASK doesn't match table");
static_assert(build_supported_mask(CIF3_FIELDS) == CIF3_SUPPORTED_MASK,
              "CIF3_SUPPORTED_MASK doesn't match table");

// Variable field length reading functions

// GPS ASCII format: 32-bit character count + ASCII data
inline size_t read_gps_ascii_length_words(const uint8_t* buffer, size_t offset) noexcept {
    uint32_t char_count = read_u32_safe(buffer, offset);
    // Return total words: 1 (count) + ceiling(chars/4)
    return 1 + (char_count + 3) / 4;
}

// Context Association Lists format: Two 16-bit counts + IDs
inline size_t read_context_assoc_length_words(const uint8_t* buffer, size_t offset) noexcept {
    // First word contains TWO 16-bit counts
    uint32_t counts_word = read_u32_safe(buffer, offset);
    uint16_t signal_stream_count = (counts_word >> 16) & 0xFFFF;
    uint16_t context_count = counts_word & 0xFFFF;

    // Total: 1 (counts) + signal_stream_count + context_count
    return 1 + signal_stream_count + context_count;
}

// Runtime field offset calculation with bounds checking - DEPRECATED
// This implementation is kept for backward compatibility only
// New code should use detail::calculate_field_offset_runtime() from variable_field_dispatch.hpp
// which provides trait-based variable field handling
inline size_t calculate_field_offset_runtime(uint32_t cif0, uint32_t cif1, uint32_t cif2,
                                             uint32_t cif3, uint8_t target_cif_word,
                                             uint8_t target_bit, const uint8_t* buffer,
                                             size_t base_offset_bytes, size_t buffer_size) {
    size_t offset_words = 0;

    // Process CIF0 fields before target
    if (target_cif_word == 0) {
        for (int bit = 31; bit > static_cast<int>(target_bit); --bit) {
            if (cif0 & (1U << bit)) {
                if (CIF0_FIELDS[bit].is_variable) {
                    size_t field_offset = base_offset_bytes + (offset_words * 4);

                    // Bounds check before reading length
                    if (field_offset + 4 > buffer_size) {
                        return SIZE_MAX; // Error indicator
                    }

                    if (bit == GPS_ASCII_BIT) {
                        offset_words += read_gps_ascii_length_words(buffer, field_offset);
                    } else if (bit == CONTEXT_ASSOC_BIT) {
                        offset_words += read_context_assoc_length_words(buffer, field_offset);
                    }
                } else {
                    offset_words += CIF0_FIELDS[bit].size_words;
                }
            }
        }
        return base_offset_bytes + (offset_words * 4);
    }

    // Count all CIF0 fields if target is in CIF1/CIF2/CIF3
    for (int bit = 31; bit >= 0; --bit) {
        if (cif0 & (1U << bit)) {
            if (CIF0_FIELDS[bit].is_variable) {
                size_t field_offset = base_offset_bytes + (offset_words * 4);

                if (field_offset + 4 > buffer_size) {
                    return SIZE_MAX;
                }

                if (bit == GPS_ASCII_BIT) {
                    offset_words += read_gps_ascii_length_words(buffer, field_offset);
                } else if (bit == CONTEXT_ASSOC_BIT) {
                    offset_words += read_context_assoc_length_words(buffer, field_offset);
                }
            } else {
                offset_words += CIF0_FIELDS[bit].size_words;
            }
        }
    }

    // Process CIF1 fields if needed
    if (target_cif_word >= 1 && (cif0 & (1U << CIF1_ENABLE_BIT))) {
        const FieldInfo* table = CIF1_FIELDS;
        uint32_t cif = cif1;

        if (target_cif_word == 1) {
            for (int bit = 31; bit > static_cast<int>(target_bit); --bit) {
                if (cif & (1U << bit)) {
                    offset_words += table[bit].size_words;
                }
            }
        } else {
            for (int bit = 31; bit >= 0; --bit) {
                if (cif & (1U << bit)) {
                    offset_words += table[bit].size_words;
                }
            }
        }
    }

    // Process CIF2 fields if needed
    if (target_cif_word >= 2 && (cif0 & (1U << CIF2_ENABLE_BIT))) {
        if (target_cif_word == 2) {
            for (int bit = 31; bit > static_cast<int>(target_bit); --bit) {
                if (cif2 & (1U << bit)) {
                    offset_words += CIF2_FIELDS[bit].size_words;
                }
            }
        } else {
            for (int bit = 31; bit >= 0; --bit) {
                if (cif2 & (1U << bit)) {
                    offset_words += CIF2_FIELDS[bit].size_words;
                }
            }
        }
    }

    // Process CIF3 fields if target is in CIF3
    if (target_cif_word == 3 && (cif0 & (1U << CIF3_ENABLE_BIT))) {
        for (int bit = 31; bit > static_cast<int>(target_bit); --bit) {
            if (cif3 & (1U << bit)) {
                offset_words += CIF3_FIELDS[bit].size_words;
            }
        }
    }

    return base_offset_bytes + (offset_words * 4);
}

// Compile-time field offset calculation (no variable fields)
template <uint32_t CIF0, uint32_t CIF1, uint32_t CIF2, uint32_t CIF3, uint8_t TargetCIF,
          uint8_t TargetBit>
constexpr size_t calculate_field_offset_ct() {
    static_assert((CIF0 & CIF0_VARIABLE_MASK) == 0,
                  "Compile-time offset does not support variable fields");

    size_t offset_words = 0;

    // Process CIF0 fields (excluding control bits 1, 2, and 3)
    if constexpr (TargetCIF == 0) {
        for (int bit = 31; bit > static_cast<int>(TargetBit); --bit) {
            // Skip CIF1/CIF2/CIF3 enable bits
            if (bit == CIF1_ENABLE_BIT || bit == CIF2_ENABLE_BIT || bit == CIF3_ENABLE_BIT)
                continue;

            if (CIF0 & (1U << bit)) {
                offset_words += CIF0_FIELDS[bit].size_words;
            }
        }
        return offset_words * 4;
    }

    // Count all CIF0 fields if target is in CIF1/CIF2/CIF3 (excluding control bits)
    for (int bit = 31; bit >= 0; --bit) {
        // Skip CIF1/CIF2/CIF3 enable bits
        if (bit == CIF1_ENABLE_BIT || bit == CIF2_ENABLE_BIT || bit == CIF3_ENABLE_BIT)
            continue;

        if (CIF0 & (1U << bit)) {
            offset_words += CIF0_FIELDS[bit].size_words;
        }
    }

    // Process CIF1 fields if CIF1 is used
    if constexpr (TargetCIF >= 1 && CIF1 != 0) {
        if constexpr (TargetCIF == 1) {
            for (int bit = 31; bit > static_cast<int>(TargetBit); --bit) {
                if (CIF1 & (1U << bit)) {
                    offset_words += CIF1_FIELDS[bit].size_words;
                }
            }
        } else {
            for (int bit = 31; bit >= 0; --bit) {
                if (CIF1 & (1U << bit)) {
                    offset_words += CIF1_FIELDS[bit].size_words;
                }
            }
        }
    }

    // Process CIF2 fields if CIF2 is used
    if constexpr (TargetCIF >= 2 && CIF2 != 0) {
        if constexpr (TargetCIF == 2) {
            for (int bit = 31; bit > static_cast<int>(TargetBit); --bit) {
                if (CIF2 & (1U << bit)) {
                    offset_words += CIF2_FIELDS[bit].size_words;
                }
            }
        } else {
            for (int bit = 31; bit >= 0; --bit) {
                if (CIF2 & (1U << bit)) {
                    offset_words += CIF2_FIELDS[bit].size_words;
                }
            }
        }
    }

    // Process CIF3 fields if CIF3 is used
    if constexpr (TargetCIF == 3 && CIF3 != 0) {
        for (int bit = 31; bit > static_cast<int>(TargetBit); --bit) {
            if (CIF3 & (1U << bit)) {
                offset_words += CIF3_FIELDS[bit].size_words;
            }
        }
    }

    return offset_words * 4;
}

// Compile-time context field size calculation
template <uint32_t CIF0, uint32_t CIF1, uint32_t CIF2, uint32_t CIF3>
constexpr size_t calculate_context_size_ct() {
    static_assert((CIF0 & CIF0_VARIABLE_MASK) == 0,
                  "Compile-time size calculation does not support variable fields");

    size_t total_words = 0;

    // Count CIF0 fields (excluding control bits 1, 2, and 3)
    for (int bit = 31; bit >= 0; --bit) {
        // Skip CIF1/CIF2/CIF3 enable bits - they're control bits, not data fields
        if (bit == 1 || bit == 2 || bit == 3)
            continue;

        if (CIF0 & (1U << bit)) {
            total_words += CIF0_FIELDS[bit].size_words;
        }
    }

    // Count CIF1 fields if CIF1 has any bits set
    if constexpr (CIF1 != 0) {
        for (int bit = 31; bit >= 0; --bit) {
            if (CIF1 & (1U << bit)) {
                total_words += CIF1_FIELDS[bit].size_words;
            }
        }
    }

    // Count CIF2 fields if CIF2 has any bits set
    if constexpr (CIF2 != 0) {
        for (int bit = 31; bit >= 0; --bit) {
            if (CIF2 & (1U << bit)) {
                total_words += CIF2_FIELDS[bit].size_words;
            }
        }
    }

    // Count CIF3 fields if CIF3 has any bits set
    if constexpr (CIF3 != 0) {
        for (int bit = 31; bit >= 0; --bit) {
            if (CIF3 & (1U << bit)) {
                total_words += CIF3_FIELDS[bit].size_words;
            }
        }
    }

    return total_words;
}

} // namespace cif

} // namespace vrtio