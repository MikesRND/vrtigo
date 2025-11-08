#pragma once

#include <cstdint>
#include <type_traits>

namespace vrtio {

// NoClassId marker type
struct NoClassId {};

// ClassId trait system for consistent interface
template<typename T>
struct ClassIdTraits;

// NoClassId specialization
template<>
struct ClassIdTraits<NoClassId> {
    static constexpr size_t size_words = 0;
    static constexpr bool has_class_id = false;
};

// ClassId type with full 32-bit PCC per VITA-49
template<uint32_t OUI, uint32_t PCC>
struct ClassId {
    static_assert(OUI <= 0xFFFFFF, "OUI must fit in 24 bits");
    // PCC is full 32 bits - no constraint needed

    static constexpr uint32_t oui = OUI;
    static constexpr uint32_t pcc = PCC;

    // VITA-49 Class ID encoding:
    // Word 0: [31:8] OUI (24 bits), [7:0] Information Class Code (ICC)
    // Word 1: Full 32-bit Packet Class Code (PCC)
    static constexpr uint32_t word0(uint8_t icc = 0) {
        return (OUI << 8) | (icc & 0xFF);
    }

    static constexpr uint32_t word1() {
        return PCC;  // NO SHIFT - full 32 bits as per VITA-49
    }
};

// ClassIdTraits specialization for ClassId
template<uint32_t OUI, uint32_t PCC>
struct ClassIdTraits<ClassId<OUI, PCC>> {
    static constexpr size_t size_words = 2;  // 64-bit class ID
    static constexpr bool has_class_id = true;
};

// Concept for valid ClassId types
template<typename T>
concept ValidClassIdType = requires {
    { ClassIdTraits<T>::size_words } -> std::convertible_to<size_t>;
    { ClassIdTraits<T>::has_class_id } -> std::convertible_to<bool>;
};

}  // namespace vrtio