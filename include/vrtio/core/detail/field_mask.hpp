#pragma once

#include <tuple>

#include <cstdint>

#include "../fields.hpp"

namespace vrtio::detail {

// Helper to add a single field tag to the appropriate CIF mask
template <auto FieldTag>
constexpr void add_field_to_mask(uint32_t& cif0, uint32_t& cif1, uint32_t& cif2, uint32_t& cif3) {
    constexpr uint8_t cif_word = decltype(FieldTag)::cif;
    constexpr uint8_t bit_pos = decltype(FieldTag)::bit;

    if constexpr (cif_word == 0) {
        cif0 |= (1U << bit_pos);
    } else if constexpr (cif_word == 1) {
        cif1 |= (1U << bit_pos);
    } else if constexpr (cif_word == 2) {
        cif2 |= (1U << bit_pos);
    } else if constexpr (cif_word == 3) {
        cif3 |= (1U << bit_pos);
    }
}

// Compute CIF bitmasks from field tags at compile-time
template <auto... Fields>
struct FieldMask {
private:
    static constexpr auto compute_masks() {
        uint32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        (add_field_to_mask<Fields>(c0, c1, c2, c3), ...);
        return std::tuple{c0, c1, c2, c3};
    }

    static constexpr auto masks = compute_masks();

public:
    static constexpr uint32_t cif0 = std::get<0>(masks);
    static constexpr uint32_t cif1 = std::get<1>(masks);
    static constexpr uint32_t cif2 = std::get<2>(masks);
    static constexpr uint32_t cif3 = std::get<3>(masks);
};

// Specialization for no fields
template <>
struct FieldMask<> {
    static constexpr uint32_t cif0 = 0;
    static constexpr uint32_t cif1 = 0;
    static constexpr uint32_t cif2 = 0;
    static constexpr uint32_t cif3 = 0;
};

// Helper function to compute bitmask for a single field tag
// Useful for test code that needs to manually construct CIF words
template <auto FieldTag>
constexpr uint32_t field_bitmask() {
    return 1U << decltype(FieldTag)::bit;
}

} // namespace vrtio::detail
