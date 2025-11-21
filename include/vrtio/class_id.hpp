#pragma once

#include <concepts>
#include <type_traits>

#include <cstddef>
#include <cstdint>

namespace vrtigo {

// Marker types for ClassId presence/absence
struct NoClassId {}; // Indicates packet has no ClassId field
struct ClassId {};   // Indicates packet has ClassId field (2 words)

// Runtime ClassId value type - trivially copyable
class ClassIdValue {
private:
    uint32_t oui_; // 24 bits
    uint16_t icc_; // 16 bits
    uint16_t pcc_; // 16 bits
    uint8_t pbc_;  // 5 bits (Pad Bit Count)

public:
    // Constructor
    constexpr ClassIdValue(uint32_t oui, uint16_t icc, uint16_t pcc = 0, uint8_t pbc = 0) noexcept
        : oui_(oui),
          icc_(icc),
          pcc_(pcc),
          pbc_(pbc) {}

    // Factory method to decode from packet words
    // Word 0: [31:27] PBC | [26:24] Reserved | [23:0] OUI
    // Word 1: [31:16] ICC | [15:0] PCC
    [[nodiscard]] static constexpr ClassIdValue fromWords(uint32_t word0, uint32_t word1) noexcept {
        uint8_t pbc = (word0 >> 27) & 0x1F;
        uint32_t oui = word0 & 0xFFFFFF;
        uint16_t icc = (word1 >> 16) & 0xFFFF;
        uint16_t pcc = word1 & 0xFFFF;
        return ClassIdValue(oui, icc, pcc, pbc);
    }

    // Accessors
    [[nodiscard]] constexpr uint32_t oui() const noexcept { return oui_; }
    [[nodiscard]] constexpr uint16_t icc() const noexcept { return icc_; }
    [[nodiscard]] constexpr uint16_t pcc() const noexcept { return pcc_; }
    [[nodiscard]] constexpr uint8_t pbc() const noexcept { return pbc_; }

    // Encoding helpers for packet writing
    // Word 0: [31:27] PBC | [26:24] Reserved (0) | [23:0] OUI
    [[nodiscard]] constexpr uint32_t word0() const noexcept {
        return ((pbc_ & 0x1F) << 27) | (oui_ & 0xFFFFFF);
    }
    // Word 1: [31:16] ICC | [15:0] PCC
    [[nodiscard]] constexpr uint32_t word1() const noexcept {
        return (static_cast<uint32_t>(icc_) << 16) | pcc_;
    }
};

// Verify trivially copyable for performance and constexpr use
static_assert(std::is_trivially_copyable_v<ClassIdValue>,
              "ClassIdValue must be trivially copyable");

// ClassId trait system for consistent interface
template <typename T>
struct ClassIdTraits;

// NoClassId specialization
template <>
struct ClassIdTraits<NoClassId> {
    static constexpr size_t size_words = 0;
    static constexpr bool has_class_id = false;
};

// ClassId specialization
template <>
struct ClassIdTraits<ClassId> {
    static constexpr size_t size_words = 2; // 64-bit class ID (2 words)
    static constexpr bool has_class_id = true;
};

// Concept for valid ClassId types
template <typename T>
concept ValidClassIdType = requires {
    { ClassIdTraits<T>::size_words } -> std::convertible_to<size_t>;
    { ClassIdTraits<T>::has_class_id } -> std::convertible_to<bool>;
};

// Concept for types that actually have a class ID
template <typename T>
concept HasClassId = ValidClassIdType<T> && ClassIdTraits<T>::has_class_id;

} // namespace vrtigo
