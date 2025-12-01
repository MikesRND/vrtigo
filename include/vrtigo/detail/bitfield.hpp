#pragma once

#include "vrtigo/detail/endian.hpp"

#include <array>
#include <bit>
#include <concepts>
#include <optional>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace vrtigo::detail {

// Forward declarations
template <typename StorageType, std::size_t Offset, std::size_t Width, std::size_t WordIndex = 0>
struct BitField;

template <typename... Fields>
struct BitFieldLayout;

// Concepts for type constraints
template <typename T>
concept UnsignedIntegral =
    std::unsigned_integral<T> && (std::same_as<T, uint8_t> || std::same_as<T, uint16_t> ||
                                  std::same_as<T, uint32_t> || std::same_as<T, uint64_t>);

template <typename T>
concept IsBitField = requires {
    typename T::storage_type;
    { T::word_index } -> std::convertible_to<std::size_t>;
    { T::offset } -> std::convertible_to<std::size_t>;
    { T::width } -> std::convertible_to<std::size_t>;
    { T::mask } -> std::convertible_to<typename T::storage_type>;
};

// Helper to calculate mask safely without UB shift
template <typename StorageType, std::size_t Width>
constexpr StorageType calculate_mask() noexcept {
    if constexpr (Width == sizeof(StorageType) * 8) {
        // Full-width field - all bits set
        return ~StorageType{0};
    } else {
        // Partial field - shift is safe because Width < bit_width
        return (StorageType{1} << Width) - 1;
    }
}

// BitField: Compile-time metadata for a single field
// WordIndex defaults to 0 for backward compatibility (single-word case)
template <typename StorageType, std::size_t Offset, std::size_t Width, std::size_t WordIndex>
struct BitField {
    static_assert(UnsignedIntegral<StorageType>,
                  "BitField storage type must be an unsigned integral type");
    static_assert(Width > 0 && Width <= sizeof(StorageType) * 8,
                  "BitField width must be between 1 and storage type bit width");
    static_assert(Offset + Width <= sizeof(StorageType) * 8,
                  "BitField extends beyond storage type boundary");

    using storage_type = StorageType;
    using value_type = std::conditional_t<
        Width == 1, bool,
        std::conditional_t<
            Width <= 8, uint8_t,
            std::conditional_t<Width <= 16, uint16_t,
                               std::conditional_t<Width <= 32, uint32_t, uint64_t>>>>;

    static constexpr std::size_t word_index = WordIndex;
    static constexpr std::size_t offset = Offset;
    static constexpr std::size_t width = Width;
    static constexpr StorageType mask = calculate_mask<StorageType, Width>();

    // Extract field value from storage word
    static constexpr value_type extract(StorageType word) noexcept {
        if constexpr (Width == 1) {
            return (word >> offset) & 1;
        } else {
            return static_cast<value_type>((word >> offset) & mask);
        }
    }

    // Insert field value into storage word
    static constexpr StorageType insert(StorageType word, value_type value) noexcept {
        const StorageType cleared = word & ~(mask << offset);
        const StorageType shifted = (static_cast<StorageType>(value) & mask) << offset;
        return cleared | shifted;
    }

    // Check if this field overlaps with another
    template <typename Other>
    static constexpr bool overlaps_with() noexcept {
        constexpr std::size_t VRT_WORD_SIZE = 4;

        // Calculate byte ranges based on VRT 32-bit word alignment
        const auto this_byte_start = word_index * VRT_WORD_SIZE;
        const auto this_byte_end = word_index * VRT_WORD_SIZE + sizeof(storage_type);
        const auto other_byte_start = Other::word_index * VRT_WORD_SIZE;
        const auto other_byte_end =
            Other::word_index * VRT_WORD_SIZE + sizeof(typename Other::storage_type);

        // If fields access the same storage word (same byte range AND same storage type),
        // check bit-level overlap within that word
        if constexpr (std::same_as<storage_type, typename Other::storage_type>) {
            if (this_byte_start == other_byte_start && this_byte_end == other_byte_end) {
                // Same storage word - check bit positions
                const auto this_bit_start = offset;
                const auto this_bit_end = offset + width;
                const auto other_bit_start = Other::offset;
                const auto other_bit_end = Other::offset + Other::width;
                return !(this_bit_end <= other_bit_start || other_bit_end <= this_bit_start);
            }
        }

        // Different storage words or types - check if byte ranges overlap
        return !(this_byte_end <= other_byte_start || other_byte_end <= this_byte_start);
    }
};

// Specialization for single-bit flags
// WordIndex defaults to 0 for backward compatibility
template <std::size_t BitPos, std::size_t WordIndex = 0>
using BitFlag = BitField<uint32_t, BitPos, 1, WordIndex>;

/**
 * BitField variant with associated enum type.
 * Enables automatic enum return from LayoutView::get().
 *
 * Do not use for 1-bit bool fields (enums cannot have bool underlying type).
 * Only use for multi-bit fields where value_type is an unsigned integer type.
 *
 * @tparam Enum The enum class to associate with this field
 * @tparam StorageType The storage type for the field
 * @tparam Offset Bit offset within the storage word
 * @tparam Width Number of bits for this field
 * @tparam WordIndex VRT word index (defaults to 0)
 */
template <typename Enum, typename StorageType, std::size_t Offset, std::size_t Width,
          std::size_t WordIndex = 0>
struct EnumBitField : BitField<StorageType, Offset, Width, WordIndex> {
    using enum_type = Enum;
    using base = BitField<StorageType, Offset, Width, WordIndex>;

    static_assert(std::is_enum_v<Enum>, "EnumBitField first parameter must be an enum type");
    static_assert(std::is_same_v<std::underlying_type_t<Enum>, typename base::value_type>,
                  "enum_type underlying type must match BitField value_type");
    static_assert(Width > 1, "EnumBitField should not be used for 1-bit fields (bool value_type)");
};

// Helper to check if fields overlap (compile-time validation)
template <typename... Fields>
constexpr bool validate_no_overlaps() noexcept {
    if constexpr (sizeof...(Fields) < 2) {
        return true;
    }

    // Check each pair of fields
    bool result = true;
    auto check_pair = [&result]<typename F1, typename F2>() {
        if constexpr (std::same_as<F1, F2>) {
            return; // Skip checking field against itself
        }
        if constexpr (F1::template overlaps_with<F2>()) {
            result = false;
        }
    };

    // Check all combinations
    ((
        [&]<typename F2>() {
            (check_pair.template operator()<Fields, F2>(), ...);
        }.template operator()<Fields>(),
        ...));

    return result;
}

// Helper to calculate maximum word index from fields
template <typename... Fields>
constexpr std::size_t calculate_max_word_index() noexcept {
    std::size_t max_index = 0;
    ((max_index = (Fields::word_index > max_index ? Fields::word_index : max_index)), ...);
    return max_index;
}

// Helper to calculate maximum byte extent from all fields (for proper sizing)
template <typename... Fields>
constexpr std::size_t calculate_max_byte_extent() noexcept {
    std::size_t max_extent = 0;
    auto update_max = [&max_extent]<typename F>() {
        // Each field occupies: word_index * 4 + sizeof(storage_type) bytes
        const std::size_t field_extent = F::word_index * 4 + sizeof(typename F::storage_type);
        if (field_extent > max_extent) {
            max_extent = field_extent;
        }
    };
    (update_max.template operator()<Fields>(), ...);
    return max_extent;
}

// BitFieldLayout: Collection of fields with compile-time validation
template <typename... Fields>
struct BitFieldLayout {
    static_assert((IsBitField<Fields> && ...), "All template parameters must be BitField types");
    static_assert(validate_no_overlaps<Fields...>(), "BitField layout contains overlapping fields");

    // VRT uses fixed 32-bit (4-byte) words regardless of field storage types
    static constexpr std::size_t word_size = 4;

    // Total bytes required - must accommodate largest field extent
    // A field at word_index N with storage_type of size S needs N*4 + S bytes
    static constexpr std::size_t required_bytes = sizeof...(Fields) > 0
                                                      ? calculate_max_byte_extent<Fields...>()
                                                      : 0;

    // Calculate number of 32-bit VRT words needed
    // Must use required_bytes to account for multi-word spanning fields
    // Example: uint64_t at word_index=1 spans bytes 4-11, needs 3 words (0,1,2)
    static constexpr std::size_t num_words = sizeof...(Fields) > 0
                                                 ? (required_bytes + word_size - 1) / word_size
                                                 : 0;

    // Helper to find a field type in the pack
    template <typename Field>
    static constexpr bool has_field = (std::same_as<Field, Fields> || ...);

    // Get storage size needed for all fields
    template <typename StorageType>
    static constexpr std::size_t storage_bytes() noexcept {
        std::size_t max_bytes = 0;
        auto update_max = [&max_bytes]<typename F>() {
            if constexpr (std::same_as<typename F::storage_type, StorageType>) {
                const std::size_t field_bytes = sizeof(typename F::storage_type);
                if (field_bytes > max_bytes) {
                    max_bytes = field_bytes;
                }
            }
        };
        (update_max.template operator()<Fields>(), ...);
        return max_bytes;
    }
};

// ============================================================================
// Shared helpers for all accessors (prevent code duplication/drift)
// ============================================================================

// Shared read - works with const or mutable buffer
template <typename StorageType, bool IsBigEndian>
[[nodiscard]] constexpr StorageType read_word_unchecked(const std::byte* data,
                                                        std::size_t offset) noexcept {
    StorageType value{};
    std::memcpy(&value, data + offset, sizeof(StorageType));

    // Keep exact existing endian logic
    if constexpr (IsBigEndian) {
        if constexpr (sizeof(StorageType) == 1) {
            return value; // No byte swap needed for single byte
        } else if constexpr (sizeof(StorageType) == 2) {
            return network_to_host16(value);
        } else if constexpr (sizeof(StorageType) == 4) {
            return network_to_host32(value);
        } else if constexpr (sizeof(StorageType) == 8) {
            return network_to_host64(value);
        }
    }
    return value;
}

// Shared write - for mutable buffer
template <typename StorageType, bool IsBigEndian>
constexpr void write_word_unchecked(std::byte* data, std::size_t offset,
                                    StorageType value) noexcept {
    if constexpr (IsBigEndian) {
        if constexpr (sizeof(StorageType) == 1) {
            // No byte swap needed for single byte
        } else if constexpr (sizeof(StorageType) == 2) {
            value = host_to_network16(value);
        } else if constexpr (sizeof(StorageType) == 4) {
            value = host_to_network32(value);
        } else if constexpr (sizeof(StorageType) == 8) {
            value = host_to_network64(value);
        }
    }
    std::memcpy(data + offset, &value, sizeof(StorageType));
}

// ============================================================================
// BitFieldAccessor: Zero-cost accessor with static extent
// ============================================================================
template <typename Layout, bool IsBigEndian = true>
class BitFieldAccessor {
    std::span<std::byte, Layout::required_bytes> buffer_;

public:
    // Exact size required - static extent ensures compile-time safety
    constexpr explicit BitFieldAccessor(
        std::span<std::byte, Layout::required_bytes> buffer) noexcept
        : buffer_(buffer) {}

    constexpr explicit BitFieldAccessor(
        std::array<std::byte, Layout::required_bytes>& buffer) noexcept
        : buffer_(buffer) {}

    // Get field value - zero overhead
    template <typename Field>
    [[nodiscard]] constexpr auto get() const noexcept
        requires IsBitField<Field> && Layout::template
    has_field<Field> {
        static_assert(Layout::template has_field<Field>);
        constexpr std::size_t byte_offset = Field::word_index * 4;
        const auto word = read_word_unchecked<typename Field::storage_type, IsBigEndian>(
            buffer_.data(), byte_offset);
        return Field::extract(word);
    }

    // Set field value - zero overhead
    template <typename Field>
    constexpr void set(typename Field::value_type value) noexcept
        requires IsBitField<Field> && Layout::template
    has_field<Field> {
        static_assert(Layout::template has_field<Field>);
        constexpr std::size_t byte_offset = Field::word_index * 4;
        auto word = read_word_unchecked<typename Field::storage_type, IsBigEndian>(buffer_.data(),
                                                                                   byte_offset);
        word = Field::insert(word, value);
        write_word_unchecked<typename Field::storage_type, IsBigEndian>(buffer_.data(), byte_offset,
                                                                        word);
    }

    // Get multiple fields at once
    template <typename... Fields>
    [[nodiscard]] constexpr auto get_multiple() const noexcept
        requires(Layout::template has_field<Fields> && ...)
    {
        return std::make_tuple(get<Fields>()...);
    }

    // Get underlying buffer
    [[nodiscard]] constexpr std::span<std::byte, Layout::required_bytes> buffer() noexcept {
        return buffer_;
    }

    [[nodiscard]] constexpr std::span<const std::byte, Layout::required_bytes>
    buffer() const noexcept {
        return buffer_;
    }
};

// ConstBitFieldAccessor: Zero-cost read-only accessor with static extent
template <typename Layout, bool IsBigEndian = true>
class ConstBitFieldAccessor {
    std::span<const std::byte, Layout::required_bytes> buffer_;

public:
    // Exact size required - static extent ensures compile-time safety
    constexpr explicit ConstBitFieldAccessor(
        std::span<const std::byte, Layout::required_bytes> buffer) noexcept
        : buffer_(buffer) {}

    constexpr explicit ConstBitFieldAccessor(
        const std::array<std::byte, Layout::required_bytes>& buffer) noexcept
        : buffer_(buffer) {}

    // Get field value - zero overhead
    template <typename Field>
    [[nodiscard]] constexpr auto get() const noexcept
        requires IsBitField<Field> && Layout::template
    has_field<Field> {
        static_assert(Layout::template has_field<Field>);
        constexpr std::size_t byte_offset = Field::word_index * 4;
        const auto word = read_word_unchecked<typename Field::storage_type, IsBigEndian>(
            buffer_.data(), byte_offset);
        return Field::extract(word);
    }

    // Get multiple fields at once
    template <typename... Fields>
    [[nodiscard]] constexpr auto get_multiple() const noexcept
        requires(Layout::template has_field<Fields> && ...)
    {
        return std::make_tuple(get<Fields>()...);
    }

    [[nodiscard]] constexpr std::span<const std::byte, Layout::required_bytes>
    buffer() const noexcept {
        return buffer_;
    }
};

/**
 * RuntimeBitFieldAccessor - Dynamic extent with validation-as-data
 *
 * Follows dynamic::DataPacketView semantics:
 *   - Validation state computed at construction and stored
 *   - Operations return std::optional or are no-ops on invalid state
 *   - No undefined behavior from API misuse
 *
 * Usage:
 *   RuntimeBitFieldAccessor accessor(buffer);
 *   if (auto value = accessor.get<Field>()) {
 *       // use *value
 *   }
 *   accessor.set<Field>(42);  // no-op if invalid
 */
template <typename Layout, bool IsBigEndian = true>
class RuntimeBitFieldAccessor {
    std::span<std::byte> buffer_;
    bool valid_;

public:
    constexpr explicit RuntimeBitFieldAccessor(std::span<std::byte> buffer) noexcept
        : buffer_(buffer),
          valid_(buffer.size() >= Layout::required_bytes) {}

    // Primary API: validation state is observable data
    [[nodiscard]] constexpr bool is_valid() const noexcept { return valid_; }

    // Alias for compatibility
    [[nodiscard]] constexpr bool valid() const noexcept { return is_valid(); }

    // Returns optional - nullopt if !valid_
    template <typename Field>
    [[nodiscard]] constexpr std::optional<typename Field::value_type> get() const noexcept
        requires IsBitField<Field> && Layout::template
    has_field<Field> {
        static_assert(Layout::template has_field<Field>);

        if (!valid_) {
            return std::nullopt;
        }

        constexpr std::size_t byte_offset = Field::word_index * 4;
        const auto word = read_word_unchecked<typename Field::storage_type, IsBigEndian>(
            buffer_.data(), byte_offset);
        return Field::extract(word);
    }

    // No-op if !valid_ (safe silent failure)
    template <typename Field>
    constexpr void set(typename Field::value_type value) noexcept
        requires IsBitField<Field> && Layout::template
    has_field<Field> {
        static_assert(Layout::template has_field<Field>);

        if (!valid_) {
            return; // No-op on invalid state
        }

        constexpr std::size_t byte_offset = Field::word_index * 4;
        auto word = read_word_unchecked<typename Field::storage_type, IsBigEndian>(buffer_.data(),
                                                                                   byte_offset);
        word = Field::insert(word, value);
        write_word_unchecked<typename Field::storage_type, IsBigEndian>(buffer_.data(), byte_offset,
                                                                        word);
    }

    // Returns tuple of optionals (one per field)
    template <typename... Fields>
    [[nodiscard]] constexpr std::tuple<std::optional<typename Fields::value_type>...>
    get_multiple() const noexcept
        requires(Layout::template has_field<Fields> && ...)
    {
        return std::make_tuple(get<Fields>()...);
    }

    [[nodiscard]] constexpr std::span<std::byte> buffer() noexcept { return buffer_; }
    [[nodiscard]] constexpr std::span<const std::byte> buffer() const noexcept { return buffer_; }
};

/**
 * RuntimeConstBitFieldAccessor - Dynamic extent read-only with validation-as-data
 *
 * Follows dynamic::DataPacketView semantics (read-only variant):
 *   - Validation state computed at construction and stored
 *   - Operations return std::optional on invalid state
 *   - No undefined behavior from API misuse
 *
 * Usage:
 *   RuntimeConstBitFieldAccessor accessor(buffer);
 *   if (auto value = accessor.get<Field>()) {
 *       // use *value
 *   }
 */
template <typename Layout, bool IsBigEndian = true>
class RuntimeConstBitFieldAccessor {
    std::span<const std::byte> buffer_;
    bool valid_;

public:
    constexpr explicit RuntimeConstBitFieldAccessor(std::span<const std::byte> buffer) noexcept
        : buffer_(buffer),
          valid_(buffer.size() >= Layout::required_bytes) {}

    // Primary API: validation state is observable data
    [[nodiscard]] constexpr bool is_valid() const noexcept { return valid_; }

    // Alias for compatibility
    [[nodiscard]] constexpr bool valid() const noexcept { return is_valid(); }

    // Returns optional - nullopt if !valid_
    template <typename Field>
    [[nodiscard]] constexpr std::optional<typename Field::value_type> get() const noexcept
        requires IsBitField<Field> && Layout::template
    has_field<Field> {
        static_assert(Layout::template has_field<Field>);

        if (!valid_) {
            return std::nullopt;
        }

        constexpr std::size_t byte_offset = Field::word_index * 4;
        const auto word = read_word_unchecked<typename Field::storage_type, IsBigEndian>(
            buffer_.data(), byte_offset);
        return Field::extract(word);
    }

    // Returns tuple of optionals (one per field)
    template <typename... Fields>
    [[nodiscard]] constexpr std::tuple<std::optional<typename Fields::value_type>...>
    get_multiple() const noexcept
        requires(Layout::template has_field<Fields> && ...)
    {
        return std::make_tuple(get<Fields>()...);
    }

    [[nodiscard]] constexpr std::span<const std::byte> buffer() const noexcept { return buffer_; }
};

// Helper factory functions for zero-cost accessors (static extent)
template <typename Layout, bool IsBigEndian = true>
constexpr auto
make_bitfield_accessor(std::span<std::byte, Layout::required_bytes> buffer) noexcept {
    return BitFieldAccessor<Layout, IsBigEndian>(buffer);
}

template <typename Layout, bool IsBigEndian = true>
constexpr auto
make_bitfield_accessor(std::span<const std::byte, Layout::required_bytes> buffer) noexcept {
    return ConstBitFieldAccessor<Layout, IsBigEndian>(buffer);
}

// Helper factory functions for runtime accessors (dynamic extent)
template <typename Layout, bool IsBigEndian = true>
constexpr auto make_runtime_bitfield_accessor(std::span<std::byte> buffer) noexcept {
    return RuntimeBitFieldAccessor<Layout, IsBigEndian>(buffer);
}

template <typename Layout, bool IsBigEndian = true>
constexpr auto make_runtime_bitfield_accessor(std::span<const std::byte> buffer) noexcept {
    return RuntimeConstBitFieldAccessor<Layout, IsBigEndian>(buffer);
}

// Utility: Create a compile-time bitmask from multiple fields
template <typename... Fields>
constexpr auto create_bitmask() noexcept
    requires(IsBitField<Fields> && ...)
{
    using StorageType = typename std::common_type<typename Fields::storage_type...>::type;
    StorageType mask = 0;
    ((mask |= (Fields::mask << Fields::offset)), ...);
    return mask;
}

} // namespace vrtigo::detail