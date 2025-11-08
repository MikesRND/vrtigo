#pragma once

#include "field_traits.hpp"
#include "../cif.hpp"
#include <cstddef>
#include <cstdint>

namespace vrtio::detail {

/// Dispatch to appropriate FieldTraits::compute_size_words for variable fields
/// Returns SIZE_MAX if the field is not a known variable field
inline size_t compute_variable_field_size(
    uint8_t cif_word,
    uint8_t bit,
    const uint8_t* buffer,
    size_t offset) noexcept
{
    // Dispatch to the appropriate FieldTraits::compute_size_words
    // Currently only CIF0 has variable fields (bits 9 and 10)
    if (cif_word == 0) {
        if (bit == 9) {
            // Context Association Lists
            return FieldTraits<0, 9>::compute_size_words(buffer, offset);
        } else if (bit == 10) {
            // GPS ASCII
            return FieldTraits<0, 10>::compute_size_words(buffer, offset);
        }
    }

    // Unknown variable field - should not happen with valid CIF
    return SIZE_MAX;
}

/// Runtime field offset calculation with bounds checking
/// Uses FieldTraits dispatch for variable field size computation
inline size_t calculate_field_offset_runtime(
    uint32_t cif0, uint32_t cif1, uint32_t cif2,
    uint8_t target_cif_word, uint8_t target_bit,
    const uint8_t* buffer,
    size_t base_offset_bytes,
    size_t buffer_size) noexcept
{
    size_t offset_words = 0;

    // Process CIF0 fields before target
    if (target_cif_word == 0) {
        for (int bit = 31; bit > static_cast<int>(target_bit); --bit) {
            if (cif0 & (1U << bit)) {
                if (cif::CIF0_FIELDS[bit].is_variable) {
                    size_t field_offset = base_offset_bytes + (offset_words * 4);

                    // Bounds check before reading length
                    if (field_offset + 4 > buffer_size) {
                        return SIZE_MAX;  // Error indicator
                    }

                    // Use FieldTraits dispatch for variable field size
                    size_t var_size = compute_variable_field_size(0, bit, buffer, field_offset);
                    if (var_size == SIZE_MAX) {
                        return SIZE_MAX;  // Unknown variable field
                    }
                    offset_words += var_size;
                } else {
                    offset_words += cif::CIF0_FIELDS[bit].size_words;
                }
            }
        }
        return base_offset_bytes + (offset_words * 4);
    }

    // Count all CIF0 fields if target is in CIF1/CIF2
    for (int bit = 31; bit >= 0; --bit) {
        if (cif0 & (1U << bit)) {
            if (cif::CIF0_FIELDS[bit].is_variable) {
                size_t field_offset = base_offset_bytes + (offset_words * 4);

                if (field_offset + 4 > buffer_size) {
                    return SIZE_MAX;
                }

                // Use FieldTraits dispatch for variable field size
                size_t var_size = compute_variable_field_size(0, bit, buffer, field_offset);
                if (var_size == SIZE_MAX) {
                    return SIZE_MAX;
                }
                offset_words += var_size;
            } else {
                offset_words += cif::CIF0_FIELDS[bit].size_words;
            }
        }
    }

    // Process CIF1 fields if needed
    if (target_cif_word >= 1 && (cif0 & 0x02)) {
        const cif::FieldInfo* table = cif::CIF1_FIELDS;
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
                offset_words += cif::CIF2_FIELDS[bit].size_words;
            }
        }
    }

    return base_offset_bytes + (offset_words * 4);
}

} // namespace vrtio::detail
