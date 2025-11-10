#pragma once

#include "vrtio/core/endian.hpp"

#include <string_view>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace vrtio {

/// Generic view for multi-word CIF fields (zero-copy)
template <size_t Words>
class FieldView {
    const uint8_t* base_;
    size_t offset_;

public:
    constexpr FieldView(const uint8_t* base, size_t offset) noexcept
        : base_(base),
          offset_(offset) {}

    /// Read a 32-bit word at the given index
    uint32_t word(size_t index) const noexcept {
        assert(index < Words);
        uint32_t value;
        std::memcpy(&value, base_ + offset_ + index * 4, sizeof(value));
        return detail::network_to_host32(value);
    }

    /// Read a 32-bit word as signed integer
    int32_t word_signed(size_t index) const noexcept { return static_cast<int32_t>(word(index)); }

    /// Read a 64-bit word starting at the given index (consumes 2 words)
    uint64_t word64(size_t index) const noexcept {
        assert(index + 1 < Words);
        uint64_t value;
        std::memcpy(&value, base_ + offset_ + index * 4, sizeof(value));
        return detail::network_to_host64(value);
    }

    /// Read a 64-bit word as signed integer
    int64_t word64_signed(size_t index) const noexcept {
        return static_cast<int64_t>(word64(index));
    }

    /// Read a fixed-point value and convert to double
    /// @param index Word index to read
    /// @param fractional_bits Number of fractional bits (e.g., 24 for Q8.24 format)
    double fixed_point(size_t index, int fractional_bits) const noexcept {
        int32_t val = word_signed(index);
        return val / static_cast<double>(1 << fractional_bits);
    }

    /// Get raw pointer to field data
    const uint8_t* data() const noexcept { return base_ + offset_; }

    /// Get size in bytes
    constexpr size_t size_bytes() const noexcept { return Words * 4; }

    /// Get size in 32-bit words
    constexpr size_t size_words() const noexcept { return Words; }
};

/// View for variable-length lists (Context Association, Index List, etc.)
class VariableListView {
    const uint8_t* base_;
    size_t offset_;
    size_t count_;

public:
    constexpr VariableListView(const uint8_t* base, size_t offset, size_t count) noexcept
        : base_(base),
          offset_(offset),
          count_(count) {}

    /// Access element by index
    uint32_t operator[](size_t index) const noexcept {
        assert(index < count_);
        uint32_t value;
        std::memcpy(&value, base_ + offset_ + index * 4, sizeof(value));
        return detail::network_to_host32(value);
    }

    /// Get number of elements
    size_t size() const noexcept { return count_; }

    /// Check if list is empty
    bool empty() const noexcept { return count_ == 0; }
};

/// View for Context Association Lists field (CIF0 bit 9)
/// Contains two variable-length lists: stream IDs and context IDs
struct ContextAssociationLists {
    VariableListView stream_ids;
    VariableListView context_ids;
};

/// View for GPS ASCII field (CIF0 bit 10)
class GPSASCIIView {
    const uint8_t* base_;
    size_t offset_;
    uint32_t char_count_;

public:
    constexpr GPSASCIIView(const uint8_t* base, size_t offset, uint32_t count) noexcept
        : base_(base),
          offset_(offset),
          char_count_(count) {}

    /// Get GPS string as string_view (zero-copy)
    std::string_view as_string() const noexcept {
        return {reinterpret_cast<const char*>(base_ + offset_ + 4), char_count_};
    }

    /// Get character count
    size_t size() const noexcept { return char_count_; }
};

} // namespace vrtio
