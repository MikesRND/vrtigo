#pragma once

#include <optional>
#include <span>
#include <type_traits>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "field_traits.hpp"

namespace vrtio {

/**
 * Proxy object for CIF field access
 *
 * Provides efficient access to packet fields by caching offset and presence.
 * Supports three levels of access:
 * - .bytes() / .set_bytes() - Literal on-wire bytes
 * - .encoded() / .set_encoded() - Structured wire format (uint32_t, FieldView, etc.)
 * - .value() / .set_value() - Interpreted values (Hz, dBm, etc.) - opt-in only
 *
 * @tparam FieldTag Field tag type (field::field_tag_t<CifWord, Bit>)
 * @tparam Packet Packet or view type (const or non-const)
 *
 * Usage:
 *   auto bw = packet[field::bandwidth];
 *   if (bw) {
 *       auto bytes = bw.bytes();              // Raw on-wire bytes
 *       uint64_t enc = bw.encoded();          // Structured Q52.12 encoding
 *       double hz = bw.value();               // Interpreted Hz (if supported)
 *   }
 */
template <typename FieldTag, typename Packet>
class FieldProxy {
public:
    using packet_type = Packet;
    using field_tag = FieldTag;

private:
    Packet* packet_;      // Non-owning pointer to packet
    size_t offset_bytes_; // Cached field offset in bytes
    size_t size_bytes_;   // Field size in bytes
    bool present_;        // Cached presence check

public:
    /**
     * Construct proxy (called by get() function)
     * @param packet Packet or view reference
     * @param offset Field offset in bytes
     * @param size Field size in bytes
     * @param present Whether field is present in packet
     */
    constexpr FieldProxy(Packet& packet, size_t offset, size_t size, bool present) noexcept
        : packet_(&packet),
          offset_bytes_(offset),
          size_bytes_(size),
          present_(present) {}

    // Copyable and movable (holds non-owning pointer)
    FieldProxy(const FieldProxy&) = default;
    FieldProxy& operator=(const FieldProxy&) = default;
    FieldProxy(FieldProxy&&) noexcept = default;
    FieldProxy& operator=(FieldProxy&&) noexcept = default;

    /**
     * Check if field is present in packet
     * @return true if field present, false otherwise
     */
    [[nodiscard]] constexpr bool has_value() const noexcept { return present_; }

    /**
     * Convert to bool for presence checking
     * Allows: if (auto field = get(...)) { ... }
     */
    [[nodiscard]] constexpr explicit operator bool() const noexcept { return present_; }

    /**
     * Get raw on-wire byte representation of field
     *
     * Returns the exact bytes as they appear in the packet, including:
     * - For fixed fields: The field's binary encoding
     * - For variable fields: Count word + data
     *
     * @return Span of const bytes (empty if field not present)
     */
    [[nodiscard]] std::span<const uint8_t> bytes() const noexcept {
        if (!present_ || !packet_) {
            return {};
        }

        // Get packet buffer
        const uint8_t* buffer = packet_->context_buffer();
        return std::span<const uint8_t>(buffer + offset_bytes_, size_bytes_);
    }

    /**
     * Write raw bytes to field (mutable packets only)
     *
     * Writes the on-wire byte representation. No interpretation is performed.
     * For variable-length fields, include the count word in the data.
     *
     * @param bytes Raw bytes to write (must match field size)
     *
     * Note: Only available for packets with mutable_context_buffer() method.
     *       ContextPacketView is read-only and will NOT have this method.
     *       ContextPacket (compile-time) DOES have this method.
     */
    void set_bytes(std::span<const uint8_t> bytes) noexcept
        requires requires(Packet& p) {
            { p.mutable_context_buffer() } -> std::same_as<uint8_t*>;
        }
    {
        if (!present_ || !packet_ || bytes.size() != size_bytes_) {
            return; // Size mismatch or field not present
        }

        // Get mutable packet buffer
        uint8_t* buffer = packet_->mutable_context_buffer();
        std::memcpy(buffer + offset_bytes_, bytes.data(), size_bytes_);
    }

    /**
     * Get field offset in bytes
     * @return Cached offset from packet start
     */
    [[nodiscard]] constexpr size_t offset() const noexcept { return offset_bytes_; }

    /**
     * Get field size in bytes
     * @return Field size
     */
    [[nodiscard]] constexpr size_t size() const noexcept { return size_bytes_; }

    /**
     * Get structured on-wire field value (FieldTraits::value_type)
     *
     * Returns the field value as decoded by FieldTraits::read().
     * This is NOT interpreted (e.g., Q52.12 encoding, not Hz).
     * For interpreted values (Hz, dBm, etc.), use .value() on fields that support it.
     *
     * @return Structured wire format (uint32_t, uint64_t, FieldView<N>, etc.)
     *
     * Precondition: has_value() must be true (checked by assertion in debug builds)
     */
    [[nodiscard]] auto encoded() const noexcept ->
        typename detail::FieldTraits<FieldTag::cif, FieldTag::bit>::value_type {
        assert(present_ && "FieldProxy::encoded() called on field that is not present");
        assert(packet_ && "FieldProxy::encoded() called on invalid proxy");

        using Trait = detail::FieldTraits<FieldTag::cif, FieldTag::bit>;
        return Trait::read(packet_->context_buffer(), offset_bytes_);
    }

    /**
     * Write structured value to field (mutable packets only)
     *
     * Writes the value using FieldTraits::write().
     * This writes the raw on-wire format (e.g., Q52.12 encoding).
     * For interpreted writes (Hz, dBm, etc.), use .set_value() on fields that support it.
     *
     * @param v Value to write (FieldTraits::value_type)
     *
     * Note: Only available for fixed-size fields with FieldTraits::write() defined.
     *       Variable-length fields are read-only.
     */
    void set_encoded(
        const typename detail::FieldTraits<FieldTag::cif, FieldTag::bit>::value_type& v) noexcept
        requires requires(Packet& p) {
            { p.mutable_context_buffer() } -> std::same_as<uint8_t*>;
        } && detail::FixedFieldTrait<detail::FieldTraits<FieldTag::cif, FieldTag::bit>>
    {
        if (!present_ || !packet_) {
            return;
        }

        using Trait = detail::FieldTraits<FieldTag::cif, FieldTag::bit>;
        Trait::write(packet_->mutable_context_buffer(), offset_bytes_, v);
    }

    /**
     * Get interpreted field value (Hz, dBm, °C, etc.)
     *
     * Returns the field value converted to human-meaningful units.
     * Only available for fields with interpreted support (FieldTraits defines
     * interpreted_type and conversion functions).
     *
     * @return Interpreted value (e.g., double Hz, dBm, °C)
     *
     * Precondition: has_value() must be true (checked by assertion in debug builds)
     *
     * Compile-time error: If called on a field without interpreted support,
     *                     you'll get a helpful error message. Use .encoded() instead.
     */
    [[nodiscard]] auto value() const noexcept -> detail::interpreted_type_or_dummy_t<FieldTag>
        requires detail::HasInterpretedAccess<FieldTag>
    {
        assert(present_ && "FieldProxy::value() called on field that is not present");
        assert(packet_ && "FieldProxy::value() called on invalid proxy");

        using Trait = detail::FieldTraits<FieldTag::cif, FieldTag::bit>;
        return Trait::to_interpreted(encoded());
    }

    /**
     * Diagnostic fallback: Helpful error when .value() called without interpreted support
     */
    template <typename T = void>
    auto value() const noexcept -> void
        requires(!detail::HasInterpretedAccess<FieldTag>)
    {
        static_assert(detail::always_false<T>,
                      "Field does not have interpreted support. "
                      "Use .encoded() to access the on-wire format, "
                      "or add interpreted_type/to_interpreted()/from_interpreted() "
                      "to the FieldTraits specialization to enable .value().");
    }

    /**
     * Write interpreted value to field (mutable packets only)
     *
     * Converts from interpreted value (Hz, dBm, etc.) to on-wire format and writes it.
     * Only available for fields with interpreted support.
     *
     * @param v Interpreted value to write (e.g., double Hz)
     *
     * Note: Only available for fixed-size fields with interpreted support.
     */
    void set_value(detail::interpreted_type_or_dummy_t<FieldTag> v) noexcept
        requires detail::HasInterpretedAccess<FieldTag> &&
                 requires(Packet& p) {
                     { p.mutable_context_buffer() } -> std::same_as<uint8_t*>;
                 } && detail::FixedFieldTrait<detail::FieldTraits<FieldTag::cif, FieldTag::bit>>
    {
        if (!present_ || !packet_) {
            return;
        }

        using Trait = detail::FieldTraits<FieldTag::cif, FieldTag::bit>;
        set_encoded(Trait::from_interpreted(v));
    }

    /**
     * Diagnostic fallback: Helpful error when .set_value() called without interpreted support
     */
    template <typename T = void>
    void set_value(const auto&) const noexcept
        requires(!detail::HasInterpretedAccess<FieldTag>)
    {
        static_assert(detail::always_false<T>,
                      "Field does not have interpreted support. "
                      "Use .set_encoded() to write the on-wire format, "
                      "or add interpreted_type/to_interpreted()/from_interpreted() "
                      "to the FieldTraits specialization to enable .set_value().");
    }

    /**
     * Dereference operator for interpreted value access
     *
     * Returns the interpreted field value (like .value()).
     * Only available for fields with interpreted support.
     *
     * Precondition: has_value() must be true (checked by assertion in debug builds)
     *
     * Usage: if (auto bw = pkt[field::bandwidth]) { std::cout << *bw << " Hz\n"; }
     */
    [[nodiscard]] auto operator*() const noexcept -> detail::interpreted_type_or_dummy_t<FieldTag>
        requires detail::HasInterpretedAccess<FieldTag>
    {
        return value();
    }

    /**
     * Diagnostic fallback: Helpful error when operator* called without interpreted support
     */
    template <typename T = void>
    auto operator*() const noexcept -> void
        requires(!detail::HasInterpretedAccess<FieldTag>)
    {
        static_assert(detail::always_false<T>,
                      "Cannot dereference field proxy without interpreted support. "
                      "Use .encoded() instead, or add interpreted support to enable operator*.");
    }
};

} // namespace vrtio
