#pragma once

#include "vrtigo/detail/bitfield.hpp"
#include "vrtigo/detail/bitfield_tag.hpp"

#include <algorithm>
#include <span>
#include <tuple>
#include <type_traits>

#include <cstddef>
#include <cstdint>

namespace vrtigo::detail {

/**
 * Mutable view providing structured access to bitfields within a layout.
 *
 * This class template provides a generic interface for accessing complex
 * multi-field structures in CIF fields. It wraps a BitFieldAccessor and
 * provides get/set methods that work with bitfield_tag identifiers.
 *
 * @tparam Layout A BitFieldLayout containing the field definitions
 * @tparam IsBigEndian Whether to use network byte order (default true for VRT)
 */
template <typename Layout, bool IsBigEndian = true>
class LayoutView {
private:
    std::span<std::byte, Layout::required_bytes> buffer_;
    BitFieldAccessor<Layout, IsBigEndian> accessor_;

public:
    /**
     * Construct from exact-sized mutable span.
     */
    explicit constexpr LayoutView(std::span<std::byte, Layout::required_bytes> buffer) noexcept
        : buffer_(buffer),
          accessor_(buffer) {}

    /**
     * Construct from raw pointer (for FieldTraits integration).
     * Explicit to prevent accidental conversions.
     */
    explicit constexpr LayoutView(uint8_t* data) noexcept
        : buffer_(reinterpret_cast<std::byte*>(data), Layout::required_bytes),
          accessor_(buffer_) {}

    /**
     * Get a field value using its bitfield_tag.
     * Returns enum_type if field declares one, otherwise returns value_type.
     *
     * @tparam FieldType The bitfield type to access
     * @param tag The bitfield_tag<FieldType> identifier (unused except for type deduction)
     * @return The field value (enum or integer depending on field declaration)
     *
     * Examples:
     *   auto type = view.get(PayloadFormat::real_complex_type);  // Returns DataSampleType
     *   auto size = view.get(PayloadFormat::data_item_size);     // Returns uint8_t
     */
    // Overload 1: Field has enum_type - return the enum
    template <typename FieldType>
        requires Layout::template
    has_field<FieldType>&& requires { typename FieldType::enum_type; } [[nodiscard]] constexpr
        typename FieldType::enum_type
        get(bitfield_tag<FieldType>) const noexcept {
        return static_cast<typename FieldType::enum_type>(accessor_.template get<FieldType>());
    }

    // Overload 2: Field has no enum_type - return raw value
    template <typename FieldType>
        requires Layout::template
    has_field<FieldType> && (!requires { typename FieldType::enum_type; }) [[nodiscard]] constexpr
        typename FieldType::value_type get(bitfield_tag<FieldType>) const noexcept {
        return accessor_.template get<FieldType>();
    }

    /**
     * Get a field value as an enum type.
     *
     * @tparam Enum The enum type to return
     * @tparam FieldType The bitfield type to access
     * @param tag The bitfield_tag<FieldType> identifier
     * @return The field value cast to the enum type
     *
     * Example:
     *   auto type = view.get_enum<DataSampleType>(PayloadFormat::real_complex_type);
     */
    template <typename Enum, typename FieldType>
        requires Layout::template
    has_field<FieldType>&& std::is_enum_v<Enum>&&
        std::convertible_to<typename FieldType::value_type,
                            std::underlying_type_t<Enum>> [[nodiscard]] constexpr Enum
        get_enum(bitfield_tag<FieldType>) const noexcept {
        return static_cast<Enum>(accessor_.template get<FieldType>());
    }

    /**
     * Set a field value using its bitfield_tag.
     *
     * @tparam FieldType The bitfield type to access
     * @param tag The bitfield_tag<FieldType> identifier (unused except for type deduction)
     * @param value The value to set
     *
     * Example:
     *   view.set(PayloadFormat::data_item_size, uint8_t{16});
     */
    template <typename FieldType>
        requires Layout::template
    has_field<FieldType> constexpr void set(bitfield_tag<FieldType>,
                                            typename FieldType::value_type value) noexcept {
        accessor_.template set<FieldType>(value);
    }

    /**
     * Set a field value from an enum (enum class friendly).
     *
     * @tparam FieldType The bitfield type to access
     * @tparam Enum The enum type (deduced)
     * @param tag The bitfield_tag<FieldType> identifier
     * @param e The enum value to set
     *
     * Example:
     *   view.set(PayloadFormat::real_complex_type, DataSampleType::real);
     */
    template <typename FieldType, typename Enum>
        requires Layout::template
    has_field<FieldType>&& std::is_enum_v<Enum>&&
        std::convertible_to<std::underlying_type_t<Enum>,
                            typename FieldType::value_type> constexpr void
        set(bitfield_tag<FieldType>, Enum e) noexcept {
        accessor_.template set<FieldType>(static_cast<typename FieldType::value_type>(e));
    }

    /**
     * Get multiple field values at once.
     *
     * @tparam FieldTypes The bitfield types to access
     * @return A tuple of field values
     *
     * Example:
     *   auto [packing, size] = view.get_multiple(
     *       PayloadFormat::packing_method,
     *       PayloadFormat::data_item_size);
     */
    template <typename... FieldTypes>
        requires(Layout::template has_field<FieldTypes> && ...)
    [[nodiscard]] constexpr auto get_multiple(bitfield_tag<FieldTypes>... tags) const noexcept {
        return std::make_tuple(get(tags)...);
    }

    /**
     * Access to underlying buffer (mutable).
     */
    [[nodiscard]] constexpr std::span<std::byte, Layout::required_bytes> bytes() noexcept {
        return buffer_;
    }

    /**
     * Access to underlying buffer (const).
     */
    [[nodiscard]] constexpr std::span<const std::byte, Layout::required_bytes>
    bytes() const noexcept {
        return buffer_;
    }

    /**
     * Get the size of the view in bytes.
     */
    [[nodiscard]] static constexpr std::size_t size() noexcept { return Layout::required_bytes; }

    /**
     * Clear all fields to zero.
     */
    constexpr void clear() noexcept { std::fill(buffer_.begin(), buffer_.end(), std::byte{0}); }
};

/**
 * Read-only view providing structured access to bitfields within a layout.
 *
 * @tparam Layout A BitFieldLayout containing the field definitions
 * @tparam IsBigEndian Whether to use network byte order (default true)
 */
template <typename Layout, bool IsBigEndian = true>
class ConstLayoutView {
private:
    std::span<const std::byte, Layout::required_bytes> buffer_;
    ConstBitFieldAccessor<Layout, IsBigEndian> accessor_;

public:
    /**
     * Construct from exact-sized const span.
     */
    explicit constexpr ConstLayoutView(
        std::span<const std::byte, Layout::required_bytes> buffer) noexcept
        : buffer_(buffer),
          accessor_(buffer) {}

    /**
     * Construct from raw const pointer.
     */
    explicit constexpr ConstLayoutView(const uint8_t* data) noexcept
        : buffer_(reinterpret_cast<const std::byte*>(data), Layout::required_bytes),
          accessor_(buffer_) {}

    /**
     * Get a field value using its bitfield_tag.
     * Returns enum_type if field declares one, otherwise returns value_type.
     */
    // Overload 1: Field has enum_type - return the enum
    template <typename FieldType>
        requires Layout::template
    has_field<FieldType>&& requires { typename FieldType::enum_type; } [[nodiscard]] constexpr
        typename FieldType::enum_type
        get(bitfield_tag<FieldType>) const noexcept {
        return static_cast<typename FieldType::enum_type>(accessor_.template get<FieldType>());
    }

    // Overload 2: Field has no enum_type - return raw value
    template <typename FieldType>
        requires Layout::template
    has_field<FieldType> && (!requires { typename FieldType::enum_type; }) [[nodiscard]] constexpr
        typename FieldType::value_type get(bitfield_tag<FieldType>) const noexcept {
        return accessor_.template get<FieldType>();
    }

    /**
     * Get a field value as an enum type.
     *
     * @tparam Enum The enum type to return
     * @tparam FieldType The bitfield type to access
     * @param tag The bitfield_tag<FieldType> identifier
     * @return The field value cast to the enum type
     *
     * Example:
     *   auto type = view.get_enum<DataSampleType>(PayloadFormat::real_complex_type);
     */
    template <typename Enum, typename FieldType>
        requires Layout::template
    has_field<FieldType>&& std::is_enum_v<Enum>&&
        std::convertible_to<typename FieldType::value_type,
                            std::underlying_type_t<Enum>> [[nodiscard]] constexpr Enum
        get_enum(bitfield_tag<FieldType>) const noexcept {
        return static_cast<Enum>(accessor_.template get<FieldType>());
    }

    /**
     * Get multiple field values at once.
     */
    template <typename... FieldTypes>
        requires(Layout::template has_field<FieldTypes> && ...)
    [[nodiscard]] constexpr auto get_multiple(bitfield_tag<FieldTypes>... tags) const noexcept {
        return std::make_tuple(get(tags)...);
    }

    /**
     * Read-only access to buffer.
     */
    [[nodiscard]] constexpr std::span<const std::byte, Layout::required_bytes>
    bytes() const noexcept {
        return buffer_;
    }

    /**
     * Get the size in bytes.
     */
    [[nodiscard]] static constexpr std::size_t size() noexcept { return Layout::required_bytes; }
};

} // namespace vrtigo::detail
