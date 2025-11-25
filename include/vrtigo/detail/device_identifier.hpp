#pragma once

#include "vrtigo/detail/bitfield.hpp"
#include "vrtigo/detail/bitfield_tag.hpp"
#include "vrtigo/detail/layout_view.hpp"

#include <cstdint>

namespace vrtigo {

/**
 * Device Identifier field (VITA 49.2 Section 9.10.1)
 *
 * Identifies the manufacturer and model of the device generating
 * the VRT packet stream.
 *
 * Word 0: Reserved[8] | Manufacturer OUI[24]
 * Word 1: Reserved[16] | Device Code[16]
 *
 * Rules:
 * - Rule 9.10.1-2: OUI is 24-bit IEEE-registered company identifier (00-00-00 to FF-FE-FF)
 * - Rule 9.10.1-3: Device Code is 16-bit manufacturer-assigned model identifier
 * - Permission 9.10.1-3: OUI may be FF-FF-FF if manufacturer unknown or has no OUI
 */
struct DeviceIdentifier {
    // ========================================================================
    // Field Type Definitions (Word 0)
    // ========================================================================

    /**
     * Manufacturer OUI (bits 23-0 of word 0)
     * 24-bit IEEE-registered Organizationally Unique Identifier
     */
    struct ManufacturerOuiField : detail::BitField<uint32_t, 0, 24, 0> {};

    // ========================================================================
    // Field Type Definitions (Word 1)
    // ========================================================================

    /**
     * Device Code (bits 15-0 of word 1)
     * 16-bit manufacturer-assigned device model code
     */
    struct DeviceCodeField : detail::BitField<uint32_t, 0, 16, 1> {};

    // ========================================================================
    // Layout Definition
    // ========================================================================

    using Layout = detail::BitFieldLayout<ManufacturerOuiField, DeviceCodeField>;

    // ========================================================================
    // View Types
    // ========================================================================

    using View = detail::LayoutView<Layout>;
    using ConstView = detail::ConstLayoutView<Layout>;

    // ========================================================================
    // Field Tags for convenient access
    // ========================================================================

    static constexpr detail::bitfield_tag<ManufacturerOuiField> manufacturer_oui{};
    static constexpr detail::bitfield_tag<DeviceCodeField> device_code{};
};

} // namespace vrtigo
