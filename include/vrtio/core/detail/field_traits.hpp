#pragma once

#include "vrtio/core/field_values.hpp"
#include "vrtio/core/cif.hpp"
#include <cstdint>
#include <concepts>

namespace vrtio::detail {

/// Base FieldTraits template - must be specialized for each field
template<uint8_t Cif, uint8_t Bit>
struct FieldTraits;  // Intentionally incomplete - forces specialization

/// Concept: All field traits must provide value_type and name
template<typename T>
concept FieldTraitLike = requires {
    typename T::value_type;
    { T::name } -> std::convertible_to<const char*>;
};

/// Concept: Fixed-size field traits must provide read/write
template<typename T>
concept FixedFieldTrait = FieldTraitLike<T> && requires(
    const uint8_t* base, size_t offset,
    uint8_t* mut_base, const typename T::value_type& v) {
    { T::read(base, offset) } -> std::same_as<typename T::value_type>;
    { T::write(mut_base, offset, v) } -> std::same_as<void>;
};

/// Concept: Variable-length field traits MUST provide compute_size_words
/// Note: Variable fields have read() but NOT write() - they are read-only
template<typename T>
concept VariableFieldTrait = FieldTraitLike<T> && requires(
    const uint8_t* base, size_t offset) {
    { T::read(base, offset) } -> std::same_as<typename T::value_type>;
    { T::compute_size_words(base, offset) } -> std::same_as<size_t>;
};

// ============================================================================
// CIF0 Field Trait Specializations
// ============================================================================

// CIF0 Bit 9: Context Association Lists (VARIABLE)
template<>
struct FieldTraits<0, 9> {
    using value_type = ContextAssociationLists;
    static constexpr const char* name = "Context Association Lists";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        uint32_t counts = cif::read_u32_safe(base, offset);
        uint16_t streams = counts >> 16;
        uint16_t contexts = counts & 0xFFFF;
        return {
            VariableListView{base, offset + 4, streams},
            VariableListView{base, offset + 4 + streams*4, contexts}
        };
    }

    static size_t compute_size_words(const uint8_t* base, size_t offset) noexcept {
        uint32_t counts = cif::read_u32_safe(base, offset);
        return 1 + (counts >> 16) + (counts & 0xFFFF);
    }
};

static_assert(VariableFieldTrait<FieldTraits<0, 9>>);

// CIF0 Bit 10: GPS ASCII (VARIABLE)
template<>
struct FieldTraits<0, 10> {
    using value_type = GPSASCIIView;
    static constexpr const char* name = "GPS ASCII";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        uint32_t count = cif::read_u32_safe(base, offset);
        return GPSASCIIView{base, offset, count};
    }

    static size_t compute_size_words(const uint8_t* base, size_t offset) noexcept {
        uint32_t count = cif::read_u32_safe(base, offset);
        return 1 + (count + 3) / 4;  // Round up to word boundary
    }
};

static_assert(VariableFieldTrait<FieldTraits<0, 10>>);

// CIF0 Bit 11: Ephemeris Reference ID
template<>
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
template<>
struct FieldTraits<0, 12> {
    using value_type = FieldView<13>;
    static constexpr const char* name = "Relative Ephemeris";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return FieldView<13>{base, offset};
    }
};

// CIF0 Bit 13: ECEF Ephemeris (13 words)
template<>
struct FieldTraits<0, 13> {
    using value_type = FieldView<13>;
    static constexpr const char* name = "ECEF Ephemeris";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return FieldView<13>{base, offset};
    }
};

// CIF0 Bit 14: Formatted GPS/INS (11 words)
template<>
struct FieldTraits<0, 14> {
    using value_type = FieldView<11>;
    static constexpr const char* name = "Formatted GPS/INS";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return FieldView<11>{base, offset};
    }
};

// CIF0 Bit 15: Data Payload Format (2 words)
template<>
struct FieldTraits<0, 15> {
    using value_type = FieldView<2>;
    static constexpr const char* name = "Data Payload Format";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return FieldView<2>{base, offset};
    }
};

// CIF0 Bit 16: State/Event Indicators
template<>
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
template<>
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
template<>
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
template<>
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
template<>
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
template<>
struct FieldTraits<0, 21> {
    using value_type = uint64_t;
    static constexpr const char* name = "Sample Rate";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u64_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u64_safe(base, offset, v);
    }
};

// CIF0 Bit 22: Over-Range Count
template<>
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
template<>
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
template<>
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
template<>
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
template<>
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
template<>
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
template<>
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
template<>
struct FieldTraits<0, 29> {
    using value_type = uint64_t;
    static constexpr const char* name = "Bandwidth";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return cif::read_u64_safe(base, offset);
    }

    static void write(uint8_t* base, size_t offset, value_type v) noexcept {
        cif::write_u64_safe(base, offset, v);
    }
};

// CIF0 Bit 30: Reference Point ID
template<>
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
template<>
struct FieldTraits<0, 31> {
    using value_type = bool;
    static constexpr const char* name = "Change Indicator";

    static value_type read([[maybe_unused]] const uint8_t* base, [[maybe_unused]] size_t offset) noexcept {
        // Change indicator is just a flag - presence indicates change
        return true;
    }
};

// ============================================================================
// CIF1 Field Trait Specializations
// ============================================================================

// CIF1 Bit 1: Buffer Size
template<>
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
template<>
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
template<>
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

// CIF1 Bit 5: Discrete I/O (64-bit) (2 words)
template<>
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
template<>
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
template<>
struct FieldTraits<1, 10> {
    using value_type = FieldView<13>;
    static constexpr const char* name = "Spectrum";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return FieldView<13>{base, offset};
    }
};

// CIF1 Bit 13: Auxiliary Bandwidth (2 words)
template<>
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
template<>
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
template<>
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
template<>
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
template<>
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
template<>
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
template<>
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
template<>
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
template<>
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
template<>
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
template<>
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
template<>
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

// ============================================================================
// CIF2 Field Trait Specializations
// ============================================================================

// CIF2 Bit 3: RF Footprint Range
template<>
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
template<>
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
template<>
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
template<>
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
template<>
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
template<>
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
template<>
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
template<>
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
template<>
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
template<>
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
template<>
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
template<>
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
template<>
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
template<>
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
template<>
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
template<>
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
template<>
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
template<>
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
template<>
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
template<>
struct FieldTraits<2, 22> {
    using value_type = FieldView<4>;
    static constexpr const char* name = "Controller UUID";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return FieldView<4>{base, offset};
    }
};

// CIF2 Bit 23: Controller ID
template<>
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
template<>
struct FieldTraits<2, 24> {
    using value_type = FieldView<4>;
    static constexpr const char* name = "Controllee UUID";

    static value_type read(const uint8_t* base, size_t offset) noexcept {
        return FieldView<4>{base, offset};
    }
};

// CIF2 Bit 25: Controllee ID
template<>
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
template<>
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
template<>
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
template<>
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
template<>
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
template<>
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
template<>
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

} // namespace vrtio::detail
