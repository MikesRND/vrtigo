#pragma once

#include "endian.hpp"
#include <cstdint>
#include <cstring>
#include <climits>

namespace vrtio {
namespace cif {

// FieldInfo struct - describes each CIF field
struct FieldInfo {
    uint8_t size_words;       // Size in 32-bit words (0 for variable)
    bool is_variable;         // True for variable-length fields
    bool is_supported;        // True if VRTIO supports this field
    const char* name;         // Field name for documentation
};

// CIF3 is indicated by bit 3 of CIF0
constexpr uint32_t CIF3_ENABLE_BIT = 3;

// Safe aligned buffer access helpers
inline uint32_t read_u32_safe(const uint8_t* buffer, size_t offset) noexcept {
    uint32_t value;
    std::memcpy(&value, buffer + offset, sizeof(value));
    return detail::network_to_host32(value);
}

inline uint64_t read_u64_safe(const uint8_t* buffer, size_t offset) noexcept {
    uint64_t value;
    std::memcpy(&value, buffer + offset, sizeof(value));
    return detail::network_to_host64(value);
}

inline void write_u32_safe(uint8_t* buffer, size_t offset, uint32_t value) noexcept {
    value = detail::host_to_network32(value);
    std::memcpy(buffer + offset, &value, sizeof(value));
}

inline void write_u64_safe(uint8_t* buffer, size_t offset, uint64_t value) noexcept {
    value = detail::host_to_network64(value);
    std::memcpy(buffer + offset, &value, sizeof(value));
}

// COMPLETE CIF0 field table - all 32 bits with verified sizes from VITA 49.2
constexpr FieldInfo CIF0_FIELDS[32] = {
    //  size, var,  supp, name
    {0, false, false, "Reserved"},                    // bit 0
    {0, false, true,  "CIF1 Enable"},                 // bit 1 - control bit, supported
    {0, false, true,  "CIF2 Enable"},                 // bit 2 - control bit, supported
    {0, false, false, "CIF3 Enable (UNSUPPORTED)"},   // bit 3
    {0, false, false, "Reserved"},                    // bit 4
    {0, false, false, "Reserved"},                    // bit 5
    {0, false, false, "Reserved"},                    // bit 6
    {0, false, false, "Field Attributes (UNSUPPORTED)"}, // bit 7
    {0, false, false, "Reserved"},                    // bit 8
    {0, true,  true,  "Context Association Lists"},   // bit 9 - VARIABLE, SUPPORTED
    {0, true,  true,  "GPS ASCII"},                   // bit 10 - VARIABLE, SUPPORTED
    {1, false, true,  "Ephemeris Ref ID"},           // bit 11
    {13, false, true, "Relative Ephemeris"},         // bit 12
    {13, false, true, "ECEF Ephemeris"},             // bit 13
    {11, false, true, "Formatted GPS/INS"},          // bit 14
    {2, false, true,  "Data Payload Format"},        // bit 15
    {1, false, true,  "State/Event Indicators"},     // bit 16
    {2, false, true,  "Device ID"},                  // bit 17
    {1, false, true,  "Temperature"},                // bit 18
    {1, false, true,  "Timestamp Cal Time"},         // bit 19
    {2, false, true,  "Timestamp Adjustment"},       // bit 20
    {2, false, true,  "Sample Rate"},                // bit 21
    {1, false, true,  "Over-Range Count"},           // bit 22
    {1, false, true,  "Gain"},                       // bit 23
    {1, false, true,  "Reference Level"},            // bit 24
    {2, false, true,  "IF Band Offset"},             // bit 25
    {2, false, true,  "RF Frequency Offset"},        // bit 26
    {2, false, true,  "RF Reference Freq"},          // bit 27
    {2, false, true,  "IF Reference Freq"},          // bit 28
    {2, false, true,  "Bandwidth"},                  // bit 29
    {1, false, true,  "Reference Point ID"},         // bit 30
    {0, false, true,  "Change Indicator"}            // bit 31 - flag only
};

// COMPLETE CIF1 field table with verified sizes
constexpr FieldInfo CIF1_FIELDS[32] = {
    //  size, var,  supp, name
    {0, false, false, "Reserved"},                    // bit 0
    {1, false, true,  "Buffer Size"},                // bit 1
    {1, false, true,  "Version and Build Code"},     // bit 2
    {1, false, true,  "V49 Spec Compliance"},        // bit 3
    {1, false, false, "Health Status"},              // bit 4
    {2, false, true,  "Discrete I/O (64 bit)"},      // bit 5
    {1, false, true,  "Discrete I/O (32 bit)"},      // bit 6
    {0, true,  false, "Index List (UNSUPPORTED)"},   // bit 7 - VARIABLE
    {0, false, false, "Reserved"},                    // bit 8
    {0, true,  false, "Sector Scan (UNSUPPORTED)"},  // bit 9 - VARIABLE
    {13, false, true, "Spectrum"},                   // bit 10
    {0, true,  false, "Array of CIFs (UNSUPPORTED)"}, // bit 11 - VARIABLE
    {0, false, false, "Reserved"},                    // bit 12
    {2, false, true,  "Aux Bandwidth"},              // bit 13
    {1, false, true,  "Aux Gain"},                   // bit 14
    {2, false, true,  "Aux Frequency"},              // bit 15
    {1, false, true,  "SNR/Noise Figure"},           // bit 16
    {1, false, true,  "Intercept Points"},           // bit 17
    {1, false, true,  "Compression Point"},          // bit 18
    {1, false, true,  "Threshold"},                  // bit 19
    {1, false, true,  "Eb/No BER"},                  // bit 20
    {0, false, false, "Reserved"},                    // bit 21
    {0, false, false, "Reserved"},                    // bit 22
    {0, false, false, "Reserved"},                    // bit 23
    {1, false, true,  "Range"},                      // bit 24
    {1, false, true,  "Beam Width"},                 // bit 25
    {1, false, true,  "Spatial Ref Type"},           // bit 26
    {1, false, true,  "Spatial Scan Type"},          // bit 27
    {0, true,  false, "3D Pointing Structure (UNSUPPORTED)"}, // bit 28
    {0, false, false, "3D Pointing Single (CONFLICT)"}, // bit 29 - SPEC CONFLICT
    {0, false, false, "Polarization (CONFLICT)"},    // bit 30 - SPEC CONFLICT
    {0, false, false, "Phase Offset (CONFLICT)"}     // bit 31 - SPEC CONFLICT
};

// COMPLETE CIF2 field table (all supported where not reserved)
constexpr FieldInfo CIF2_FIELDS[32] = {
    //  size, var,  supp, name
    {0, false, false, "Reserved"},                    // bit 0
    {0, false, false, "Reserved"},                    // bit 1
    {0, false, false, "Reserved"},                    // bit 2
    {1, false, true,  "RF Footprint Range"},         // bit 3
    {1, false, true,  "RF Footprint"},               // bit 4
    {1, false, true,  "Comm Priority ID"},           // bit 5
    {1, false, true,  "Function Priority ID"},       // bit 6
    {1, false, true,  "Event ID"},                   // bit 7
    {1, false, true,  "Mode ID"},                    // bit 8
    {1, false, true,  "Function ID"},                // bit 9
    {1, false, true,  "Modulation Type"},            // bit 10
    {1, false, true,  "Modulation Class"},           // bit 11
    {1, false, true,  "EMS Device Instance"},        // bit 12
    {1, false, true,  "EMS Device Type"},            // bit 13
    {1, false, true,  "EMS Device Class"},           // bit 14
    {1, false, true,  "Platform Display"},           // bit 15
    {1, false, true,  "Platform Instance"},          // bit 16
    {1, false, true,  "Platform Class"},             // bit 17
    {1, false, true,  "Operator"},                   // bit 18
    {1, false, true,  "Country Code"},               // bit 19
    {1, false, true,  "Track ID"},                   // bit 20
    {1, false, true,  "Information Source"},         // bit 21
    {4, false, true,  "Controller UUID"},            // bit 22
    {1, false, true,  "Controller ID"},              // bit 23
    {4, false, true,  "Controllee UUID"},            // bit 24
    {1, false, true,  "Controllee ID"},              // bit 25
    {1, false, true,  "Cited Message ID"},           // bit 26
    {1, false, true,  "Child(ren) SID"},             // bit 27
    {1, false, true,  "Parent(s) SID"},              // bit 28
    {1, false, true,  "Sibling(s) SID"},             // bit 29
    {1, false, true,  "Cited SID"},                  // bit 30
    {1, false, true,  "Bind"}                        // bit 31
};

// Build SUPPORTED masks from the tables
// Includes CIF1/CIF2 enable control bits (1 and 2) since they are valid in CIF0
constexpr uint32_t CIF0_SUPPORTED_MASK =
    (1U << 1) |   // CIF1 Enable (control bit)
    (1U << 2) |   // CIF2 Enable (control bit)
    (1U << 9) |   // Context Association Lists
    (1U << 10) |  // GPS ASCII
    (1U << 11) |  // Ephemeris Ref ID
    (1U << 12) |  // Relative Ephemeris
    (1U << 13) |  // ECEF Ephemeris
    (1U << 14) |  // Formatted GPS/INS
    (1U << 15) |  // Data Payload Format
    (1U << 16) |  // State/Event Indicators
    (1U << 17) |  // Device ID
    (1U << 18) |  // Temperature
    (1U << 19) |  // Timestamp Cal Time
    (1U << 20) |  // Timestamp Adjustment
    (1U << 21) |  // Sample Rate
    (1U << 22) |  // Over-Range Count
    (1U << 23) |  // Gain
    (1U << 24) |  // Reference Level
    (1U << 25) |  // IF Band Offset
    (1U << 26) |  // RF Frequency Offset
    (1U << 27) |  // RF Reference Freq
    (1U << 28) |  // IF Reference Freq
    (1U << 29) |  // Bandwidth
    (1U << 30) |  // Reference Point ID
    (1U << 31);   // Change Indicator

constexpr uint32_t CIF1_SUPPORTED_MASK =
    (1U << 1) |   // Buffer Size
    (1U << 2) |   // Version and Build Code
    (1U << 3) |   // V49 Spec Compliance
    (1U << 5) |   // Discrete I/O (64 bit)
    (1U << 6) |   // Discrete I/O (32 bit)
    (1U << 10) |  // Spectrum
    (1U << 13) |  // Aux Bandwidth
    (1U << 14) |  // Aux Gain
    (1U << 15) |  // Aux Frequency
    (1U << 16) |  // SNR/Noise Figure
    (1U << 17) |  // Intercept Points
    (1U << 18) |  // Compression Point
    (1U << 19) |  // Threshold
    (1U << 20) |  // Eb/No BER
    (1U << 24) |  // Range
    (1U << 25) |  // Beam Width
    (1U << 26) |  // Spatial Ref Type
    (1U << 27);   // Spatial Scan Type

constexpr uint32_t CIF2_SUPPORTED_MASK =
    (1U << 3) |   // RF Footprint Range
    (1U << 4) |   // RF Footprint
    (1U << 5) |   // Comm Priority ID
    (1U << 6) |   // Function Priority ID
    (1U << 7) |   // Event ID
    (1U << 8) |   // Mode ID
    (1U << 9) |   // Function ID
    (1U << 10) |  // Modulation Type
    (1U << 11) |  // Modulation Class
    (1U << 12) |  // EMS Device Instance
    (1U << 13) |  // EMS Device Type
    (1U << 14) |  // EMS Device Class
    (1U << 15) |  // Platform Display
    (1U << 16) |  // Platform Instance
    (1U << 17) |  // Platform Class
    (1U << 18) |  // Operator
    (1U << 19) |  // Country Code
    (1U << 20) |  // Track ID
    (1U << 21) |  // Information Source
    (1U << 22) |  // Controller UUID
    (1U << 23) |  // Controller ID
    (1U << 24) |  // Controllee UUID
    (1U << 25) |  // Controllee ID
    (1U << 26) |  // Cited Message ID
    (1U << 27) |  // Child(ren) SID
    (1U << 28) |  // Parent(s) SID
    (1U << 29) |  // Sibling(s) SID
    (1U << 30) |  // Cited SID
    (1U << 31);   // Bind

// Variable fields that need special handling at compile-time
constexpr uint32_t CIF0_VARIABLE_MASK =
    (1U << 10) |  // GPS ASCII
    (1U << 9);    // Context Association Lists

// For compile-time: supported minus variable fields
constexpr uint32_t CIF0_COMPILETIME_SUPPORTED_MASK =
    CIF0_SUPPORTED_MASK & ~CIF0_VARIABLE_MASK;

// Helper to build supported mask at compile time
template<size_t N>
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
inline size_t calculate_field_offset_runtime(
    uint32_t cif0, uint32_t cif1, uint32_t cif2,
    uint8_t target_cif_word, uint8_t target_bit,
    const uint8_t* buffer,
    size_t base_offset_bytes,
    size_t buffer_size) {

    size_t offset_words = 0;

    // Process CIF0 fields before target
    if (target_cif_word == 0) {
        for (int bit = 31; bit > static_cast<int>(target_bit); --bit) {
            if (cif0 & (1U << bit)) {
                if (CIF0_FIELDS[bit].is_variable) {
                    size_t field_offset = base_offset_bytes + (offset_words * 4);

                    // Bounds check before reading length
                    if (field_offset + 4 > buffer_size) {
                        return SIZE_MAX;  // Error indicator
                    }

                    if (bit == 10) {  // GPS ASCII
                        offset_words += read_gps_ascii_length_words(buffer, field_offset);
                    } else if (bit == 9) {  // Context Assoc
                        offset_words += read_context_assoc_length_words(buffer, field_offset);
                    }
                } else {
                    offset_words += CIF0_FIELDS[bit].size_words;
                }
            }
        }
        return base_offset_bytes + (offset_words * 4);
    }

    // Count all CIF0 fields if target is in CIF1/CIF2
    for (int bit = 31; bit >= 0; --bit) {
        if (cif0 & (1U << bit)) {
            if (CIF0_FIELDS[bit].is_variable) {
                size_t field_offset = base_offset_bytes + (offset_words * 4);

                if (field_offset + 4 > buffer_size) {
                    return SIZE_MAX;
                }

                if (bit == 10) {
                    offset_words += read_gps_ascii_length_words(buffer, field_offset);
                } else if (bit == 9) {
                    offset_words += read_context_assoc_length_words(buffer, field_offset);
                }
            } else {
                offset_words += CIF0_FIELDS[bit].size_words;
            }
        }
    }

    // Process CIF1 fields if needed
    if (target_cif_word >= 1 && (cif0 & 0x02)) {
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

    // Process CIF2 fields if target is in CIF2
    if (target_cif_word == 2 && (cif0 & 0x04)) {
        for (int bit = 31; bit > static_cast<int>(target_bit); --bit) {
            if (cif2 & (1U << bit)) {
                offset_words += CIF2_FIELDS[bit].size_words;
            }
        }
    }

    return base_offset_bytes + (offset_words * 4);
}

// Compile-time field offset calculation (no variable fields)
template<uint32_t CIF0, uint32_t CIF1, uint32_t CIF2,
         uint8_t TargetCIF, uint8_t TargetBit>
constexpr size_t calculate_field_offset_ct() {
    static_assert((CIF0 & CIF0_VARIABLE_MASK) == 0,
        "Compile-time offset does not support variable fields");

    size_t offset_words = 0;

    // Process CIF0 fields (excluding control bits 1 and 2)
    if constexpr (TargetCIF == 0) {
        for (int bit = 31; bit > static_cast<int>(TargetBit); --bit) {
            // Skip CIF1/CIF2 enable bits
            if (bit == 1 || bit == 2) continue;

            if (CIF0 & (1U << bit)) {
                offset_words += CIF0_FIELDS[bit].size_words;
            }
        }
        return offset_words * 4;
    }

    // Count all CIF0 fields if target is in CIF1/CIF2 (excluding control bits)
    for (int bit = 31; bit >= 0; --bit) {
        // Skip CIF1/CIF2 enable bits
        if (bit == 1 || bit == 2) continue;

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
    if constexpr (TargetCIF == 2 && CIF2 != 0) {
        for (int bit = 31; bit > static_cast<int>(TargetBit); --bit) {
            if (CIF2 & (1U << bit)) {
                offset_words += CIF2_FIELDS[bit].size_words;
            }
        }
    }

    return offset_words * 4;
}

// Compile-time context field size calculation
template<uint32_t CIF0, uint32_t CIF1, uint32_t CIF2>
constexpr size_t calculate_context_size_ct() {
    static_assert((CIF0 & CIF0_VARIABLE_MASK) == 0,
        "Compile-time size calculation does not support variable fields");

    size_t total_words = 0;

    // Count CIF0 fields (excluding control bits 1 and 2)
    for (int bit = 31; bit >= 0; --bit) {
        // Skip CIF1/CIF2 enable bits - they're control bits, not data fields
        if (bit == 1 || bit == 2) continue;

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

    return total_words;
}

}  // namespace cif

// ============================================================================
// Named constants for user-friendly API (in vrtio namespace)
// ============================================================================

// CIF0 field constants - organized by category
namespace cif0 {
    // Variable-length fields (runtime only - NOT for compile-time ContextPacket)
    constexpr uint32_t CONTEXT_ASSOCIATION_LISTS = (1U << 9);
    constexpr uint32_t GPS_ASCII = (1U << 10);

    // Fixed-size context fields (safe for compile-time)
    constexpr uint32_t EPHEMERIS_REF_ID = (1U << 11);
    constexpr uint32_t RELATIVE_EPHEMERIS = (1U << 12);
    constexpr uint32_t ECEF_EPHEMERIS = (1U << 13);
    constexpr uint32_t FORMATTED_GPS_INS = (1U << 14);
    constexpr uint32_t DATA_PAYLOAD_FORMAT = (1U << 15);
    constexpr uint32_t STATE_EVENT_INDICATORS = (1U << 16);
    constexpr uint32_t DEVICE_ID = (1U << 17);
    constexpr uint32_t TEMPERATURE = (1U << 18);
    constexpr uint32_t TIMESTAMP_CAL_TIME = (1U << 19);
    constexpr uint32_t TIMESTAMP_ADJUSTMENT = (1U << 20);
    constexpr uint32_t SAMPLE_RATE = (1U << 21);
    constexpr uint32_t OVER_RANGE_COUNT = (1U << 22);
    constexpr uint32_t GAIN = (1U << 23);
    constexpr uint32_t REFERENCE_LEVEL = (1U << 24);
    constexpr uint32_t IF_BAND_OFFSET = (1U << 25);
    constexpr uint32_t RF_FREQUENCY_OFFSET = (1U << 26);
    constexpr uint32_t RF_REFERENCE_FREQ = (1U << 27);
    constexpr uint32_t IF_REFERENCE_FREQ = (1U << 28);
    constexpr uint32_t BANDWIDTH = (1U << 29);
    constexpr uint32_t REFERENCE_POINT_ID = (1U << 30);
    constexpr uint32_t CHANGE_INDICATOR = (1U << 31);
}  // namespace cif0

// CIF1 field constants (only supported fields, matching CIF1_FIELDS table)
namespace cif1 {
    constexpr uint32_t BUFFER_SIZE = (1U << 1);
    constexpr uint32_t VERSION_BUILD_CODE = (1U << 2);
    constexpr uint32_t V49_SPEC_COMPLIANCE = (1U << 3);
    constexpr uint32_t DISCRETE_IO_64 = (1U << 5);
    constexpr uint32_t DISCRETE_IO_32 = (1U << 6);
    constexpr uint32_t SPECTRUM = (1U << 10);
    constexpr uint32_t AUX_BANDWIDTH = (1U << 13);
    constexpr uint32_t AUX_GAIN = (1U << 14);
    constexpr uint32_t AUX_FREQUENCY = (1U << 15);
    constexpr uint32_t SNR_NOISE_FIGURE = (1U << 16);
    constexpr uint32_t INTERCEPT_POINTS = (1U << 17);
    constexpr uint32_t COMPRESSION_POINT = (1U << 18);
    constexpr uint32_t THRESHOLD = (1U << 19);
    constexpr uint32_t EBNO_BER = (1U << 20);
    constexpr uint32_t RANGE = (1U << 24);
    constexpr uint32_t BEAM_WIDTH = (1U << 25);
    constexpr uint32_t SPATIAL_REF_TYPE = (1U << 26);
    constexpr uint32_t SPATIAL_SCAN_TYPE = (1U << 27);
}  // namespace cif1

// CIF2 field constants (only supported fields, matching CIF2_FIELDS table)
namespace cif2 {
    constexpr uint32_t RF_FOOTPRINT_RANGE = (1U << 3);
    constexpr uint32_t RF_FOOTPRINT = (1U << 4);
    constexpr uint32_t COMM_PRIORITY_ID = (1U << 5);
    constexpr uint32_t FUNCTION_PRIORITY_ID = (1U << 6);
    constexpr uint32_t EVENT_ID = (1U << 7);
    constexpr uint32_t MODE_ID = (1U << 8);
    constexpr uint32_t FUNCTION_ID = (1U << 9);
    constexpr uint32_t MODULATION_TYPE = (1U << 10);
    constexpr uint32_t MODULATION_CLASS = (1U << 11);
    constexpr uint32_t EMS_DEVICE_INSTANCE = (1U << 12);
    constexpr uint32_t EMS_DEVICE_TYPE = (1U << 13);
    constexpr uint32_t EMS_DEVICE_CLASS = (1U << 14);
    constexpr uint32_t PLATFORM_DISPLAY = (1U << 15);
    constexpr uint32_t PLATFORM_INSTANCE = (1U << 16);
    constexpr uint32_t PLATFORM_CLASS = (1U << 17);
    constexpr uint32_t OPERATOR = (1U << 18);
    constexpr uint32_t COUNTRY_CODE = (1U << 19);
    constexpr uint32_t TRACK_ID = (1U << 20);
    constexpr uint32_t INFORMATION_SOURCE = (1U << 21);
    constexpr uint32_t CONTROLLER_UUID = (1U << 22);
    constexpr uint32_t CONTROLLER_ID = (1U << 23);
    constexpr uint32_t CONTROLLEE_UUID = (1U << 24);
    constexpr uint32_t CONTROLLEE_ID = (1U << 25);
    constexpr uint32_t CITED_MESSAGE_ID = (1U << 26);
    constexpr uint32_t CHILD_SID = (1U << 27);
    constexpr uint32_t PARENT_SID = (1U << 28);
    constexpr uint32_t SIBLING_SID = (1U << 29);
    constexpr uint32_t CITED_SID = (1U << 30);
    constexpr uint32_t BIND = (1U << 31);
}  // namespace cif2

}  // namespace vrtio