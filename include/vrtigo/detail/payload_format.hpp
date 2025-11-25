#pragma once

#include "vrtigo/detail/bitfield.hpp"
#include "vrtigo/detail/bitfield_tag.hpp"
#include "vrtigo/detail/layout_view.hpp"

#include <cstdint>

namespace vrtigo {

/**
 * Data Sample Type codes (VITA 49.2 Table 9.13.3-3)
 */
enum class DataSampleType : uint8_t {
    real = 0b00,
    complex_cartesian = 0b01,
    complex_polar = 0b10,
    reserved = 0b11
};

/**
 * Data Item Format codes (VITA 49.2 Table 9.13.3-4)
 */
enum class DataItemFormatCode : uint8_t {
    signed_fixed_point = 0b00000,
    signed_vrt_1bit = 0b00001,
    signed_vrt_2bit = 0b00010,
    signed_vrt_3bit = 0b00011,
    signed_vrt_4bit = 0b00100,
    signed_vrt_5bit = 0b00101,
    signed_vrt_6bit = 0b00110,
    signed_fixed_point_non_normalized = 0b00111,
    // 0b01000-0b01100 reserved
    ieee_754_half_precision = 0b01101,
    ieee_754_single_precision = 0b01110,
    ieee_754_double_precision = 0b01111,
    unsigned_fixed_point = 0b10000,
    unsigned_vrt_1bit = 0b10001,
    unsigned_vrt_2bit = 0b10010,
    unsigned_vrt_3bit = 0b10011,
    unsigned_vrt_4bit = 0b10100,
    unsigned_vrt_5bit = 0b10101,
    unsigned_vrt_6bit = 0b10110,
    unsigned_fixed_point_non_normalized = 0b10111
    // 0b11000-0b11111 reserved
};

/**
 * Data Packet Payload Format field (VITA 49.2 Section 9.13.3)
 *
 * This 64-bit CIF field describes the format of data samples in the
 * paired Data Packet Stream. It specifies packing method, data type,
 * item format, size information, and vector/repeat counts.
 *
 * All field definitions, tags, and view types are contained within
 * this structure for easy access and to avoid namespace pollution.
 */
struct PayloadFormat {
    // ========================================================================
    // Field Type Definitions (Word 0)
    // ========================================================================

    /**
     * Packing Method (bit 31)
     * 0 = processing-efficient packing
     * 1 = link-efficient packing
     */
    struct PackingMethodField : detail::BitField<uint32_t, 31, 1> {};

    /**
     * Real/Complex Type (bits 30-29)
     * Indicates whether data samples are real or complex
     */
    using RealComplexTypeField = detail::EnumBitField<DataSampleType, uint32_t, 29, 2>;

    /**
     * Data Item Format (bits 28-24)
     * 5-bit code indicating the format type (Table 9.13.3-4)
     */
    using DataItemFormatField = detail::EnumBitField<DataItemFormatCode, uint32_t, 24, 5>;

    /**
     * Sample-Component Repeat Indicator (bit 23)
     * 0 = Sample Component Repeating not in use
     * 1 = Sample Component Repeating in use
     */
    struct SampleComponentRepeatField : detail::BitField<uint32_t, 23, 1> {};

    /**
     * Event-Tag Size (bits 22-20)
     * Unsigned number (one less than actual size)
     */
    struct EventTagSizeField : detail::BitField<uint32_t, 20, 3> {};

    /**
     * Channel-Tag Size (bits 19-16)
     * Unsigned number (one less than actual size)
     */
    struct ChannelTagSizeField : detail::BitField<uint32_t, 16, 4> {};

    /**
     * Data Item Fraction Size (bits 15-12)
     * For VRT fixed-point formats, number of bits in fraction
     */
    struct DataItemFractionSizeField : detail::BitField<uint32_t, 12, 4> {};

    /**
     * Item Packing Field Size (bits 11-6)
     * Unsigned number (one less than actual size)
     */
    struct ItemPackingFieldSizeField : detail::BitField<uint32_t, 6, 6> {};

    /**
     * Data Item Size (bits 5-0)
     * Unsigned number (one less than actual size)
     */
    struct DataItemSizeField : detail::BitField<uint32_t, 0, 6> {};

    // ========================================================================
    // Field Type Definitions (Word 1)
    // ========================================================================

    /**
     * Repeat Count (bits 31-16 of word 1)
     * Number (one less than actual count)
     */
    struct RepeatCountField : detail::BitField<uint32_t, 16, 16, 1> {};

    /**
     * Vector Size (bits 15-0 of word 1)
     * Number (one less than actual size)
     */
    struct VectorSizeField : detail::BitField<uint32_t, 0, 16, 1> {};

    // ========================================================================
    // Layout Definition
    // ========================================================================

    using Layout =
        detail::BitFieldLayout<PackingMethodField, RealComplexTypeField, DataItemFormatField,
                               SampleComponentRepeatField, EventTagSizeField, ChannelTagSizeField,
                               DataItemFractionSizeField, ItemPackingFieldSizeField,
                               DataItemSizeField, RepeatCountField, VectorSizeField>;

    // ========================================================================
    // View Types
    // ========================================================================

    using View = detail::LayoutView<Layout>;
    using ConstView = detail::ConstLayoutView<Layout>;

    // ========================================================================
    // Public Tag Objects (Clean Interface)
    // ========================================================================

    static constexpr detail::bitfield_tag<PackingMethodField> packing_method{};
    static constexpr detail::bitfield_tag<RealComplexTypeField> real_complex_type{};
    static constexpr detail::bitfield_tag<DataItemFormatField> data_item_format{};
    static constexpr detail::bitfield_tag<SampleComponentRepeatField> sample_component_repeat{};
    static constexpr detail::bitfield_tag<EventTagSizeField> event_tag_size{};
    static constexpr detail::bitfield_tag<ChannelTagSizeField> channel_tag_size{};
    static constexpr detail::bitfield_tag<DataItemFractionSizeField> data_item_fraction_size{};
    static constexpr detail::bitfield_tag<ItemPackingFieldSizeField> item_packing_field_size{};
    static constexpr detail::bitfield_tag<DataItemSizeField> data_item_size{};
    static constexpr detail::bitfield_tag<RepeatCountField> repeat_count{};
    static constexpr detail::bitfield_tag<VectorSizeField> vector_size{};
};

} // namespace vrtigo
