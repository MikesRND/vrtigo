#pragma once

#include <span>

#include <cstdio>
#include <vrtigo/class_id.hpp>
#include <vrtigo/types.hpp>

#include "../detail/packet_header_accessor.hpp"
#include "../detail/prologue.hpp"
#include "../detail/timestamp_traits.hpp"

namespace vrtigo::typed::detail {

/**
 * CRTP base class for compile-time packets
 *
 * Provides common prologue field accessors (header, stream_id, class_id,
 * timestamp, packet_count) and buffer access. Derived classes implement
 * packet-specific functionality (payload for DataPacket, CIF for ContextPacket).
 *
 * @tparam Derived The derived packet class (CRTP)
 * @tparam PrologueType The Prologue<...> instantiation for this packet
 */
template <typename Derived, typename PrologueType>
class PacketBase {
public:
    // Expose prologue type for derived classes
    using prologue_type = PrologueType;
    using timestamp_type = typename prologue_type::timestamp_type;

protected:
    uint8_t* buffer_;
    mutable prologue_type prologue_;

    // Protected constructor - only derived classes can construct
    explicit PacketBase(uint8_t* buffer) noexcept : buffer_(buffer), prologue_(buffer) {}

public:
    // Prevent copying (packet is a view over buffer)
    PacketBase(const PacketBase&) = delete;
    PacketBase& operator=(const PacketBase&) = delete;

    // Allow moving
    PacketBase(PacketBase&&) noexcept = default;
    PacketBase& operator=(PacketBase&&) noexcept = default;

    // ========================================================================
    // Static property functions (delegate to prologue_type)
    // ========================================================================

    static constexpr bool has_stream_id() noexcept { return prologue_type::has_stream_id(); }
    static constexpr bool has_class_id() noexcept { return prologue_type::has_class_id(); }
    static constexpr bool has_timestamp() noexcept { return prologue_type::has_timestamp(); }

    // Note: has_trailer(), type(), size_words(), size_bytes() remain in derived
    // classes since they have packet-specific logic

    // ========================================================================
    // Header accessors
    // ========================================================================

    vrtigo::detail::MutableHeaderView header() noexcept {
        return vrtigo::detail::MutableHeaderView{&prologue_.header_word()};
    }

    vrtigo::detail::HeaderView header() const noexcept {
        return vrtigo::detail::HeaderView{&prologue_.header_word()};
    }

    // ========================================================================
    // Packet count (4-bit field: valid range 0-15)
    // ========================================================================

    uint8_t packet_count() const noexcept { return prologue_.packet_count(); }

    void set_packet_count(uint8_t count) noexcept {
#ifndef NDEBUG
        if (count > 15) {
            std::fprintf(stderr,
                         "WARNING: packet_count value %u exceeds 4-bit limit (15). "
                         "Value will be wrapped modulo 16 to %u.\n",
                         static_cast<unsigned>(count), static_cast<unsigned>(count & 0x0F));
        }
#endif
        prologue_.set_packet_count(count);
    }

    // ========================================================================
    // Stream ID accessors (conditional on has_stream_id)
    // ========================================================================

    uint32_t stream_id() const noexcept
        requires(has_stream_id())
    {
        return prologue_.stream_id();
    }

    void set_stream_id(uint32_t id) noexcept
        requires(has_stream_id())
    {
        prologue_.set_stream_id(id);
    }

    // ========================================================================
    // Class ID accessors (conditional on has_class_id)
    // ========================================================================

    ClassIdValue class_id() const noexcept
        requires(has_class_id())
    {
        return prologue_.class_id();
    }

    void set_class_id(ClassIdValue cid) noexcept
        requires(has_class_id())
    {
        prologue_.set_class_id(cid);
    }

    // ========================================================================
    // Timestamp accessors (conditional on has_timestamp)
    // ========================================================================

    timestamp_type timestamp() const noexcept
        requires(has_timestamp())
    {
        return prologue_.timestamp();
    }

    void set_timestamp(timestamp_type ts) noexcept
        requires(has_timestamp())
    {
        prologue_.set_timestamp(ts);
    }

    // ========================================================================
    // Buffer access
    // Uses Derived::safe_size_words() via CRTP for dynamic extent (clamped for safety)
    // Note: auto return type required because Derived is incomplete at base instantiation
    // ========================================================================

    auto as_bytes() noexcept {
        return std::span<uint8_t>(buffer_,
                                  static_cast<Derived*>(this)->safe_size_words() * vrt_word_size);
    }

    auto as_bytes() const noexcept {
        return std::span<const uint8_t>(
            buffer_, static_cast<const Derived*>(this)->safe_size_words() * vrt_word_size);
    }
};

} // namespace vrtigo::typed::detail
