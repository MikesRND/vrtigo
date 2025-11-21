#pragma once

#include <concepts>

#include <cstdint>

#include "cif.hpp"
#include "field_values.hpp"

namespace vrtigo::detail {

/// Base FieldTraits template - must be specialized for each field
template <uint8_t Cif, uint8_t Bit>
struct FieldTraits; // Intentionally incomplete - forces specialization

/// Concept: All field traits must provide value_type and name
template <typename T>
concept FieldTraitLike = requires {
    typename T::value_type;
    { T::name } -> std::convertible_to<const char*>;
};

/// Concept: Fixed-size field traits must provide read/write
template <typename T>
concept FixedFieldTrait =
    FieldTraitLike<T> && requires(const uint8_t* base, size_t offset, uint8_t* mut_base,
                                  const typename T::value_type& v) {
        { T::read(base, offset) } -> std::same_as<typename T::value_type>;
        { T::write(mut_base, offset, v) } -> std::same_as<void>;
    };

/// Concept: Variable-length field traits MUST provide compute_size_words
/// Note: Variable fields have read() but NOT write() - they are read-only
template <typename T>
concept VariableFieldTrait = FieldTraitLike<T> && requires(const uint8_t* base, size_t offset) {
    { T::read(base, offset) } -> std::same_as<typename T::value_type>;
    { T::compute_size_words(base, offset) } -> std::same_as<size_t>;
};

/// Helper for dependent static_assert in template contexts
template <typename>
inline constexpr bool always_false = false;

/// Concept: Field has interpreted value support (unit conversion, etc.)
/// A field has interpreted access if its FieldTraits specialization defines:
/// - interpreted_type (e.g., double for Hz, dBm, °C)
/// - to_interpreted(value_type) -> interpreted_type
/// - from_interpreted(interpreted_type) -> value_type
template <typename Tag>
concept HasInterpretedAccess = requires {
    typename FieldTraits<Tag::cif, Tag::bit>::interpreted_type;
    {
        FieldTraits<Tag::cif, Tag::bit>::to_interpreted(
            std::declval<typename FieldTraits<Tag::cif, Tag::bit>::value_type>())
    } -> std::same_as<typename FieldTraits<Tag::cif, Tag::bit>::interpreted_type>;
    {
        FieldTraits<Tag::cif, Tag::bit>::from_interpreted(
            std::declval<typename FieldTraits<Tag::cif, Tag::bit>::interpreted_type>())
    } -> std::same_as<typename FieldTraits<Tag::cif, Tag::bit>::value_type>;
};

/// Dummy type for fields without interpreted support
struct NoInterpretedType {};

/// Helper: Get interpreted_type if it exists, otherwise NoInterpretedType
/// This allows return types and parameter types to be well-formed even for fields without
/// interpreted support
template <typename Tag>
struct InterpretedTypeOrDummy {
    using type = NoInterpretedType;
};

template <typename Tag>
    requires requires { typename FieldTraits<Tag::cif, Tag::bit>::interpreted_type; }
struct InterpretedTypeOrDummy<Tag> {
    using type = typename FieldTraits<Tag::cif, Tag::bit>::interpreted_type;
};

template <typename Tag>
using interpreted_type_or_dummy_t = typename InterpretedTypeOrDummy<Tag>::type;

// ============================================================================
// CIF0 Field Trait Specializations
// ============================================================================

// CIF0 Bit 9: Context Association Lists (VARIABLE)
template <>
struct FieldTraits<0, 9> {
    using value_type = ContextAssociationLists;
    static constexpr const char* name = "Context Association Lists";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        uint32_t counts = cif::read_u32_safe(base, offset);
        uint16_t streams = counts >> 16;
        uint16_t contexts = counts & 0xFFFF;
        return {VariableListView{base, offset + 4, streams},
                VariableListView{base, offset + 4 + streams * 4, contexts}};
    }

    static size_t compute_size_words(const uint8_t* base, size_t offset) noexcept {
        uint32_t counts = cif::read_u32_safe(base, offset);
        return 1 + (counts >> 16) + (counts & 0xFFFF);
    }
};

static_assert(VariableFieldTrait<FieldTraits<0, 9>>);

// CIF0 Bit 10: GPS ASCII (VARIABLE)
template <>
struct FieldTraits<0, 10> {
    using value_type = GPSASCIIView;
    static constexpr const char* name = "GPS ASCII";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        uint32_t count = cif::read_u32_safe(base, offset);
        return GPSASCIIView{base, offset, count};
    }

    static size_t compute_size_words(const uint8_t* base, size_t offset) noexcept {
        uint32_t count = cif::read_u32_safe(base, offset);
        return 1 + (count + 3) / 4; // Round up to word boundary
    }
};

static_assert(VariableFieldTrait<FieldTraits<0, 10>>);

// CIF0 Bit 11: Ephemeris Reference ID
template <>
struct FieldTraits<0, 11> {
    using value_type = uint32_t;
    static constexpr const char* name = "Ephemeris Reference ID";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF0 Bit 12: Relative Ephemeris (13 words)
template <>
struct FieldTraits<0, 12> {
    using value_type = FieldView<13>;
    static constexpr const char* name = "Relative Ephemeris";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return FieldView<13>{base, offset};
    }

    static void write(uint8_t* base, size_t offset, const value_type& v) noexcept {
        std::memcpy(base + offset, v.data(), 13 * 4);
    }
};

// CIF0 Bit 13: ECEF Ephemeris (13 words)
template <>
struct FieldTraits<0, 13> {
    using value_type = FieldView<13>;
    static constexpr const char* name = "ECEF Ephemeris";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return FieldView<13>{base, offset};
    }

    static void write(uint8_t* base, size_t offset, const value_type& v) noexcept {
        std::memcpy(base + offset, v.data(), 13 * 4);
    }
};

// CIF0 Bit 14: Formatted GPS/INS (11 words)
template <>
struct FieldTraits<0, 14> {
    using value_type = FieldView<11>;
    static constexpr const char* name = "Formatted GPS/INS";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return FieldView<11>{base, offset};
    }

    static void write(uint8_t* base, size_t offset, const value_type& v) noexcept {
        std::memcpy(base + offset, v.data(), 11 * 4);
    }
};

// CIF0 Bit 15: Data Payload Format (2 words)
template <>
struct FieldTraits<0, 15> {
    using value_type = FieldView<2>;
    static constexpr const char* name = "Data Payload Format";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return FieldView<2>{base, offset};
    }

    static void write(uint8_t* base, size_t offset, const value_type& v) noexcept {
        std::memcpy(base + offset, v.data(), 2 * 4);
    }
};

// CIF0 Bit 16: State/Event Indicators
template <>
struct FieldTraits<0, 16> {
    using value_type = uint32_t;
    static constexpr const char* name = "State/Event Indicators";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF0 Bit 17: Device ID (2 words)
template <>
struct FieldTraits<0, 17> {
    using value_type = uint64_t;
    static constexpr const char* name = "Device ID";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u64_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u64_safe(base, offset, v);
    }
};

// CIF0 Bit 18: Temperature
template <>
struct FieldTraits<0, 18> {
    using value_type = uint32_t;
    static constexpr const char* name = "Temperature";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF0 Bit 19: Timestamp Calibration Time
template <>
struct FieldTraits<0, 19> {
    using value_type = uint32_t;
    static constexpr const char* name = "Timestamp Calibration Time";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF0 Bit 20: Timestamp Adjustment (2 words)
template <>
struct FieldTraits<0, 20> {
    using value_type = uint64_t;
    static constexpr const char* name = "Timestamp Adjustment";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u64_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u64_safe(base, offset, v);
    }
};

// CIF0 Bit 21: Sample Rate (2 words)
template <>
struct FieldTraits<0, 21> {
    using value_type = uint64_t;
    static constexpr const char* name = "Sample Rate";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u64_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u64_safe(base, offset, v);
    }

    // Interpreted support: Q52.12 fixed-point → Hz (double)
    using interpreted_type = double;

    static interpreted_type to_interpreted(value_type raw) noexcept {
        return static_cast<double>(raw) / 4096.0; // Q52.12 → Hz
    }

    static value_type from_interpreted(interpreted_type hz) noexcept {
        return static_cast<value_type>(hz * 4096.0 + 0.5); // Round to nearest
    }
};

// CIF0 Bit 22: Over-Range Count
template <>
struct FieldTraits<0, 22> {
    using value_type = uint32_t;
    static constexpr const char* name = "Over-Range Count";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF0 Bit 23: Gain
template <>
struct FieldTraits<0, 23> {
    using value_type = uint32_t;
    static constexpr const char* name = "Gain";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF0 Bit 24: Reference Level
template <>
struct FieldTraits<0, 24> {
    using value_type = uint32_t;
    static constexpr const char* name = "Reference Level";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF0 Bit 25: IF Band Offset (2 words)
template <>
struct FieldTraits<0, 25> {
    using value_type = uint64_t;
    static constexpr const char* name = "IF Band Offset";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u64_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u64_safe(base, offset, v);
    }
};

// CIF0 Bit 26: RF Frequency Offset (2 words)
template <>
struct FieldTraits<0, 26> {
    using value_type = uint64_t;
    static constexpr const char* name = "RF Frequency Offset";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u64_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u64_safe(base, offset, v);
    }
};

// CIF0 Bit 27: RF Reference Frequency (2 words)
template <>
struct FieldTraits<0, 27> {
    using value_type = uint64_t;
    static constexpr const char* name = "RF Reference Frequency";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u64_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u64_safe(base, offset, v);
    }
};

// CIF0 Bit 28: IF Reference Frequency (2 words)
template <>
struct FieldTraits<0, 28> {
    using value_type = uint64_t;
    static constexpr const char* name = "IF Reference Frequency";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u64_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u64_safe(base, offset, v);
    }
};

// CIF0 Bit 29: Bandwidth (2 words)
template <>
struct FieldTraits<0, 29> {
    using value_type = uint64_t;
    static constexpr const char* name = "Bandwidth";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u64_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u64_safe(base, offset, v);
    }

    // Interpreted support: Q52.12 fixed-point → Hz (double)
    using interpreted_type = double;

    static interpreted_type to_interpreted(value_type raw) noexcept {
        // Q52.12 format: Divide by 2^12 = 4096
        return static_cast<double>(raw) / 4096.0;
    }

    static value_type from_interpreted(interpreted_type hz) noexcept {
        // Convert Hz to Q52.12 format with rounding
        return static_cast<value_type>(hz * 4096.0 + 0.5);
    }
};

// CIF0 Bit 30: Reference Point ID
template <>
struct FieldTraits<0, 30> {
    using value_type = uint32_t;
    static constexpr const char* name = "Reference Point ID";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF0 Bit 31: Change Indicator (flag only - no data)
template <>
struct FieldTraits<0, 31> {
    using value_type = bool;
    static constexpr const char* name = "Change Indicator";

    static value_type read([[maybe_unused]] const uint8_t* base,
                           [[maybe_unused]] size_t offset) noexcept {
        // Change indicator is just a flag - presence indicates change
        return true;
    }
};

// ============================================================================
// CIF1 Field Trait Specializations
// ============================================================================

// CIF1 Bit 1: Buffer Size
template <>
struct FieldTraits<1, 1> {
    using value_type = uint32_t;
    static constexpr const char* name = "Buffer Size";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF1 Bit 2: Version and Build Code
template <>
struct FieldTraits<1, 2> {
    using value_type = uint32_t;
    static constexpr const char* name = "Version and Build Code";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF1 Bit 3: V49 Spec Compliance
template <>
struct FieldTraits<1, 3> {
    using value_type = uint32_t;
    static constexpr const char* name = "V49 Spec Compliance";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF1 Bit 4: Health Status
template <>
struct FieldTraits<1, 4> {
    using value_type = uint32_t;
    static constexpr const char* name = "Health Status";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF1 Bit 5: Discrete I/O (64-bit) (2 words)
template <>
struct FieldTraits<1, 5> {
    using value_type = uint64_t;
    static constexpr const char* name = "Discrete I/O (64-bit)";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u64_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u64_safe(base, offset, v);
    }
};

// CIF1 Bit 6: Discrete I/O (32-bit)
template <>
struct FieldTraits<1, 6> {
    using value_type = uint32_t;
    static constexpr const char* name = "Discrete I/O (32-bit)";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF1 Bit 10: Spectrum (13 words)
template <>
struct FieldTraits<1, 10> {
    using value_type = FieldView<13>;
    static constexpr const char* name = "Spectrum";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return FieldView<13>{base, offset};
    }

    static void write(uint8_t* base, size_t offset, const value_type& v) noexcept {
        std::memcpy(base + offset, v.data(), 13 * 4);
    }
};

// CIF1 Bit 13: Auxiliary Bandwidth (2 words)
template <>
struct FieldTraits<1, 13> {
    using value_type = uint64_t;
    static constexpr const char* name = "Auxiliary Bandwidth";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u64_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u64_safe(base, offset, v);
    }
};

// CIF1 Bit 14: Auxiliary Gain
template <>
struct FieldTraits<1, 14> {
    using value_type = uint32_t;
    static constexpr const char* name = "Auxiliary Gain";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF1 Bit 15: Auxiliary Frequency (2 words)
template <>
struct FieldTraits<1, 15> {
    using value_type = uint64_t;
    static constexpr const char* name = "Auxiliary Frequency";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u64_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u64_safe(base, offset, v);
    }
};

// CIF1 Bit 16: SNR/Noise Figure
template <>
struct FieldTraits<1, 16> {
    using value_type = uint32_t;
    static constexpr const char* name = "SNR/Noise Figure";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF1 Bit 17: Intercept Points
template <>
struct FieldTraits<1, 17> {
    using value_type = uint32_t;
    static constexpr const char* name = "Intercept Points";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF1 Bit 18: Compression Point
template <>
struct FieldTraits<1, 18> {
    using value_type = uint32_t;
    static constexpr const char* name = "Compression Point";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF1 Bit 19: Threshold
template <>
struct FieldTraits<1, 19> {
    using value_type = uint32_t;
    static constexpr const char* name = "Threshold";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF1 Bit 20: Eb/No BER
template <>
struct FieldTraits<1, 20> {
    using value_type = uint32_t;
    static constexpr const char* name = "Eb/No BER";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF1 Bit 24: Range
template <>
struct FieldTraits<1, 24> {
    using value_type = uint32_t;
    static constexpr const char* name = "Range";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF1 Bit 25: Beam Width
template <>
struct FieldTraits<1, 25> {
    using value_type = uint32_t;
    static constexpr const char* name = "Beam Width";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF1 Bit 26: Spatial Reference Type
template <>
struct FieldTraits<1, 26> {
    using value_type = uint32_t;
    static constexpr const char* name = "Spatial Reference Type";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF1 Bit 27: Spatial Scan Type
template <>
struct FieldTraits<1, 27> {
    using value_type = uint32_t;
    static constexpr const char* name = "Spatial Scan Type";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF1 Bit 29: 3-D Pointing Vector (single)
template <>
struct FieldTraits<1, 29> {
    using value_type = uint32_t;
    static constexpr const char* name = "3-D Pointing Vector (single)";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF1 Bit 30: Polarization
template <>
struct FieldTraits<1, 30> {
    using value_type = uint32_t;
    static constexpr const char* name = "Polarization";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF1 Bit 31: Phase Offset
template <>
struct FieldTraits<1, 31> {
    using value_type = uint32_t;
    static constexpr const char* name = "Phase Offset";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// ============================================================================
// CIF2 Field Trait Specializations
// ============================================================================

// CIF2 Bit 3: RF Footprint Range
template <>
struct FieldTraits<2, 3> {
    using value_type = uint32_t;
    static constexpr const char* name = "RF Footprint Range";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF2 Bit 4: RF Footprint
template <>
struct FieldTraits<2, 4> {
    using value_type = uint32_t;
    static constexpr const char* name = "RF Footprint";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF2 Bit 5: Communication Priority
template <>
struct FieldTraits<2, 5> {
    using value_type = uint32_t;
    static constexpr const char* name = "Communication Priority";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF2 Bit 6: Function Priority
template <>
struct FieldTraits<2, 6> {
    using value_type = uint32_t;
    static constexpr const char* name = "Function Priority";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF2 Bit 7: Event ID
template <>
struct FieldTraits<2, 7> {
    using value_type = uint32_t;
    static constexpr const char* name = "Event ID";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF2 Bit 8: Mode ID
template <>
struct FieldTraits<2, 8> {
    using value_type = uint32_t;
    static constexpr const char* name = "Mode ID";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF2 Bit 9: Function ID
template <>
struct FieldTraits<2, 9> {
    using value_type = uint32_t;
    static constexpr const char* name = "Function ID";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF2 Bit 10: Modulation Type
template <>
struct FieldTraits<2, 10> {
    using value_type = uint32_t;
    static constexpr const char* name = "Modulation Type";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF2 Bit 11: Modulation Class
template <>
struct FieldTraits<2, 11> {
    using value_type = uint32_t;
    static constexpr const char* name = "Modulation Class";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF2 Bit 12: EMS Device Instance
template <>
struct FieldTraits<2, 12> {
    using value_type = uint32_t;
    static constexpr const char* name = "EMS Device Instance";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF2 Bit 13: EMS Device Type
template <>
struct FieldTraits<2, 13> {
    using value_type = uint32_t;
    static constexpr const char* name = "EMS Device Type";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF2 Bit 14: EMS Device Class
template <>
struct FieldTraits<2, 14> {
    using value_type = uint32_t;
    static constexpr const char* name = "EMS Device Class";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF2 Bit 15: Platform Display
template <>
struct FieldTraits<2, 15> {
    using value_type = uint32_t;
    static constexpr const char* name = "Platform Display";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF2 Bit 16: Platform Instance
template <>
struct FieldTraits<2, 16> {
    using value_type = uint32_t;
    static constexpr const char* name = "Platform Instance";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF2 Bit 17: Platform Class
template <>
struct FieldTraits<2, 17> {
    using value_type = uint32_t;
    static constexpr const char* name = "Platform Class";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF2 Bit 18: Operator ID
template <>
struct FieldTraits<2, 18> {
    using value_type = uint32_t;
    static constexpr const char* name = "Operator ID";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF2 Bit 19: Country Code
template <>
struct FieldTraits<2, 19> {
    using value_type = uint32_t;
    static constexpr const char* name = "Country Code";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF2 Bit 20: Track ID
template <>
struct FieldTraits<2, 20> {
    using value_type = uint32_t;
    static constexpr const char* name = "Track ID";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF2 Bit 21: Information Source
template <>
struct FieldTraits<2, 21> {
    using value_type = uint32_t;
    static constexpr const char* name = "Information Source";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF2 Bit 22: Controller UUID (4 words)
template <>
struct FieldTraits<2, 22> {
    using value_type = FieldView<4>;
    static constexpr const char* name = "Controller UUID";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return FieldView<4>{base, offset};
    }

    static void write(uint8_t* base, size_t offset, const value_type& v) noexcept {
        std::memcpy(base + offset, v.data(), 4 * 4);
    }
};

// CIF2 Bit 23: Controller ID
template <>
struct FieldTraits<2, 23> {
    using value_type = uint32_t;
    static constexpr const char* name = "Controller ID";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF2 Bit 24: Controllee UUID (4 words)
template <>
struct FieldTraits<2, 24> {
    using value_type = FieldView<4>;
    static constexpr const char* name = "Controllee UUID";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return FieldView<4>{base, offset};
    }

    static void write(uint8_t* base, size_t offset, const value_type& v) noexcept {
        std::memcpy(base + offset, v.data(), 4 * 4);
    }
};

// CIF2 Bit 25: Controllee ID
template <>
struct FieldTraits<2, 25> {
    using value_type = uint32_t;
    static constexpr const char* name = "Controllee ID";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF2 Bit 26: Cited Message ID
template <>
struct FieldTraits<2, 26> {
    using value_type = uint32_t;
    static constexpr const char* name = "Cited Message ID";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF2 Bit 27: Child Stream ID
template <>
struct FieldTraits<2, 27> {
    using value_type = uint32_t;
    static constexpr const char* name = "Child Stream ID";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF2 Bit 28: Parent Stream ID
template <>
struct FieldTraits<2, 28> {
    using value_type = uint32_t;
    static constexpr const char* name = "Parent Stream ID";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF2 Bit 29: Sibling Stream ID
template <>
struct FieldTraits<2, 29> {
    using value_type = uint32_t;
    static constexpr const char* name = "Sibling Stream ID";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF2 Bit 30: Cited SID
template <>
struct FieldTraits<2, 30> {
    using value_type = uint32_t;
    static constexpr const char* name = "Cited SID";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF2 Bit 31: Bind
template <>
struct FieldTraits<2, 31> {
    using value_type = uint32_t;
    static constexpr const char* name = "Bind";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// ============================================================================
// CIF3 Field Trait Specializations - Temporal and Environmental
// ============================================================================

// CIF3 Bit 1: Network ID
template <>
struct FieldTraits<3, 1> {
    using value_type = uint32_t;
    static constexpr const char* name = "Network ID";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF3 Bit 2: Tropospheric State
template <>
struct FieldTraits<3, 2> {
    using value_type = uint32_t;
    static constexpr const char* name = "Tropospheric State";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF3 Bit 3: Sea and Swell State
template <>
struct FieldTraits<3, 3> {
    using value_type = uint32_t;
    static constexpr const char* name = "Sea and Swell State";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF3 Bit 4: Barometric Pressure
template <>
struct FieldTraits<3, 4> {
    using value_type = uint32_t;
    static constexpr const char* name = "Barometric Pressure";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF3 Bit 5: Humidity
template <>
struct FieldTraits<3, 5> {
    using value_type = uint32_t;
    static constexpr const char* name = "Humidity";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF3 Bit 6: Sea/Ground Temperature
template <>
struct FieldTraits<3, 6> {
    using value_type = uint32_t;
    static constexpr const char* name = "Sea/Ground Temperature";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF3 Bit 7: Air Temperature
template <>
struct FieldTraits<3, 7> {
    using value_type = uint32_t;
    static constexpr const char* name = "Air Temperature";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u32_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u32_safe(base, offset, v);
    }
};

// CIF3 Bit 20: Jitter (2 words)
template <>
struct FieldTraits<3, 20> {
    using value_type = uint64_t;
    static constexpr const char* name = "Jitter";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u64_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u64_safe(base, offset, v);
    }
};

// CIF3 Bit 21: Dwell (2 words)
template <>
struct FieldTraits<3, 21> {
    using value_type = uint64_t;
    static constexpr const char* name = "Dwell";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u64_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u64_safe(base, offset, v);
    }
};

// CIF3 Bit 22: Duration (2 words)
template <>
struct FieldTraits<3, 22> {
    using value_type = uint64_t;
    static constexpr const char* name = "Duration";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u64_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u64_safe(base, offset, v);
    }
};

// CIF3 Bit 23: Period (2 words)
template <>
struct FieldTraits<3, 23> {
    using value_type = uint64_t;
    static constexpr const char* name = "Period";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u64_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u64_safe(base, offset, v);
    }
};

// CIF3 Bit 24: Pulse Width (2 words)
template <>
struct FieldTraits<3, 24> {
    using value_type = uint64_t;
    static constexpr const char* name = "Pulse Width";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u64_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u64_safe(base, offset, v);
    }
};

// CIF3 Bit 25: Offset Time (2 words)
template <>
struct FieldTraits<3, 25> {
    using value_type = uint64_t;
    static constexpr const char* name = "Offset Time";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u64_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u64_safe(base, offset, v);
    }
};

// CIF3 Bit 26: Fall Time (2 words)
template <>
struct FieldTraits<3, 26> {
    using value_type = uint64_t;
    static constexpr const char* name = "Fall Time";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u64_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u64_safe(base, offset, v);
    }
};

// CIF3 Bit 27: Rise Time (2 words)
template <>
struct FieldTraits<3, 27> {
    using value_type = uint64_t;
    static constexpr const char* name = "Rise Time";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u64_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u64_safe(base, offset, v);
    }
};

// CIF3 Bit 30: Timestamp Skew (2 words)
template <>
struct FieldTraits<3, 30> {
    using value_type = uint64_t;
    static constexpr const char* name = "Timestamp Skew";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u64_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u64_safe(base, offset, v);
    }
};

// CIF3 Bit 31: Timestamp Details (2 words)
template <>
struct FieldTraits<3, 31> {
    using value_type = uint64_t;
    static constexpr const char* name = "Timestamp Details";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u64_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u64_safe(base, offset, v);
    }
};

} // namespace vrtigo::detail
