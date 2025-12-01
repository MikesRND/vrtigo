#pragma once

#include <type_traits>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vrtigo/class_id.hpp>
#include <vrtigo/timestamp.hpp>
#include <vrtigo/types.hpp>

#include "buffer_io.hpp"
#include "header.hpp"
#include "header_decode.hpp"
#include "header_init.hpp"
#include "timestamp_traits.hpp"

namespace vrtigo {

/**
 * @brief Consolidates the common prologue fields for VRT packets
 *
 * The Prologue manages the variable header components that appear at the
 * beginning of all VRT packet types:
 * - Header (always present, 1 word)
 * - Stream ID (optional/mandatory based on packet type)
 * - Class ID (optional, 2 words when present)
 * - Timestamp (optional, TSI: 0-1 word, TSF: 0-2 words)
 *
 * This class eliminates code duplication between DataPacket and ContextPacket
 * while maintaining zero runtime overhead through compile-time calculations.
 *
 * @tparam Type The packet type (determines stream ID presence for data packets)
 * @tparam ClassIdType NoClassId or WithClassId marker type
 * @tparam TimestampType NoTimestamp or Timestamp<TSI,TSF> type
 * @tparam IsContext True for context packets (stream ID always present)
 */
template <PacketType Type = PacketType::signal_data_no_id, typename ClassIdType = NoClassId,
          typename TimestampType = NoTimestamp, bool IsContext = false>
class Prologue {
    // ========================================================================
    // Private implementation details (internal word counts)
    // These must be declared first so size_words() can use them
    // ========================================================================
    static constexpr size_t header_words_ = 1;
    static constexpr size_t stream_id_words_ =
        (IsContext || Type == PacketType::signal_data || Type == PacketType::extension_data) ? 1
                                                                                             : 0;
    static constexpr size_t class_id_words_ = ClassIdTraits<ClassIdType>::size_words;
    static constexpr TsiType tsi_ = TimestampTraits<TimestampType>::tsi;
    static constexpr TsfType tsf_ = TimestampTraits<TimestampType>::tsf;
    static constexpr size_t tsi_words_ = (tsi_ == TsiType::none) ? 0 : 1;
    static constexpr size_t tsf_words_ = (tsf_ == TsfType::none) ? 0 : 2;
    static constexpr size_t timestamp_words_ = tsi_words_ + tsf_words_;

public:
    // Type traits
    static constexpr PacketType packet_type = Type;
    using class_id_type = ClassIdType;
    using timestamp_type = TimestampType;

    // ========================================================================
    // Static constexpr property functions (public API)
    // These can be called as T::has_stream_id() or pkt.has_stream_id()
    // ========================================================================

    /// Check if this is a context packet
    static constexpr bool is_context_packet() noexcept { return IsContext; }

    /// Check if packet has stream ID field
    static constexpr bool has_stream_id() noexcept {
        return IsContext || (Type == PacketType::signal_data || Type == PacketType::extension_data);
    }

    /// Check if packet has class ID field
    static constexpr bool has_class_id() noexcept {
        return ClassIdTraits<ClassIdType>::has_class_id;
    }

    /// Check if packet has timestamp fields
    static constexpr bool has_timestamp() noexcept {
        return TimestampTraits<TimestampType>::has_timestamp;
    }

    /// Get prologue size in 32-bit words
    static constexpr size_t size_words() noexcept {
        return header_words_ + stream_id_words_ + class_id_words_ + timestamp_words_;
    }

    /// Get prologue size in bytes
    static constexpr size_t size_bytes() noexcept { return size_words() * vrt_word_size; }

    // Extract timestamp components (public - needed for validation)
    static constexpr TsiType tsi = tsi_;
    static constexpr TsfType tsf = tsf_;

    // Field offsets in words (from start of packet) - public for DataPacket/ContextPacket
    static constexpr size_t header_offset = 0;
    static constexpr size_t stream_id_offset = header_offset + header_words_;
    static constexpr size_t class_id_offset = stream_id_offset + stream_id_words_;
    static constexpr size_t tsi_offset = class_id_offset + class_id_words_;
    static constexpr size_t tsf_offset = tsi_offset + tsi_words_;
    static constexpr size_t payload_offset = tsf_offset + tsf_words_;

    /**
     * @brief Construct a Prologue with a buffer
     * @param buffer The packet buffer
     */
    explicit Prologue(uint8_t* buffer) noexcept : buffer_(buffer) {}

    /**
     * @brief Initialize the header field
     * @param packet_size Total packet size in words
     * @param packet_count Packet sequence count
     * @param has_trailer Whether packet has trailer (data packets only)
     */
    void init_header(uint16_t packet_size, uint8_t packet_count = 0,
                     bool has_trailer = false) noexcept {
        const uint32_t header =
            detail::build_header(static_cast<uint8_t>(Type), // Cast enum to uint8_t
                                 has_class_id(),             // Class ID indicator
                                 has_trailer,                // Bit 26: Trailer indicator
                                 false,                      // Bit 25: Nd0 indicator
                                 false,                      // Bit 24: Spectrum/Time
                                 tsi,                        // TSI field
                                 tsf,                        // TSF field
                                 packet_count,               // Packet count
                                 packet_size                 // Packet size in words
            );
        detail::write_u32(buffer_, header_offset * vrt_word_size, header);
    }

    /**
     * @brief Initialize the stream ID field (if present)
     */
    void init_stream_id() noexcept
        requires(has_stream_id())
    {
        detail::write_u32(buffer_, stream_id_offset * vrt_word_size, 0);
    }

    /**
     * @brief Initialize the class ID fields (if present)
     */
    void init_class_id() noexcept
        requires(has_class_id())
    {
        std::memset(buffer_ + class_id_offset * vrt_word_size, 0, class_id_words_ * vrt_word_size);
    }

    /**
     * @brief Initialize timestamp fields (if present)
     */
    void init_timestamps() noexcept
        requires(has_timestamp())
    {
        if constexpr (tsi != TsiType::none) {
            detail::write_u32(buffer_, tsi_offset * vrt_word_size, 0);
        }
        if constexpr (tsf != TsfType::none) {
            std::memset(buffer_ + tsf_offset * vrt_word_size, 0, tsf_words_ * vrt_word_size);
        }
    }

    /**
     * @brief Get the packet header
     */
    [[nodiscard]] uint32_t header() const noexcept {
        return detail::read_u32(buffer_, header_offset * vrt_word_size);
    }

    /**
     * @brief Get reference to raw header word (mutable)
     *
     * Used by MutableHeaderView for direct manipulation.
     */
    [[nodiscard]] uint32_t& header_word() noexcept {
        return *reinterpret_cast<uint32_t*>(buffer_ + header_offset * vrt_word_size);
    }

    /**
     * @brief Get reference to raw header word (const)
     *
     * Used by HeaderView for reading.
     */
    [[nodiscard]] const uint32_t& header_word() const noexcept {
        return *reinterpret_cast<const uint32_t*>(buffer_ + header_offset * vrt_word_size);
    }

    /**
     * @brief Get the packet size from header
     */
    [[nodiscard]] uint16_t packet_size() const noexcept { return header() & header::size_mask; }

    /**
     * @brief Set the packet size in header
     */
    void set_packet_size(uint16_t size) noexcept {
        uint32_t h = header();
        h = (h & ~header::size_mask) | (size & header::size_mask);
        detail::write_u32(buffer_, header_offset * vrt_word_size, h);
    }

    /**
     * @brief Get the packet count from header
     */
    [[nodiscard]] uint8_t packet_count() const noexcept {
        return static_cast<uint8_t>((header() >> header::packet_count_shift) &
                                    header::packet_count_mask);
    }

    /**
     * @brief Set the packet count in header
     */
    void set_packet_count(uint8_t count) noexcept {
        uint32_t h = header();
        h = (h & ~(header::packet_count_mask << header::packet_count_shift)) |
            ((count & header::packet_count_mask) << header::packet_count_shift);
        detail::write_u32(buffer_, header_offset * vrt_word_size, h);
    }

    /**
     * @brief Get the stream ID (if present)
     */
    [[nodiscard]] uint32_t stream_id() const noexcept
        requires(has_stream_id())
    {
        return detail::read_u32(buffer_, stream_id_offset * vrt_word_size);
    }

    /**
     * @brief Set the stream ID (if present)
     */
    void set_stream_id(uint32_t id) noexcept
        requires(has_stream_id())
    {
        detail::write_u32(buffer_, stream_id_offset * vrt_word_size, id);
    }

    /**
     * @brief Get the class ID (if present)
     */
    ClassIdValue class_id() const noexcept
        requires(has_class_id())
    {
        uint32_t word0 = detail::read_u32(buffer_, class_id_offset * vrt_word_size);
        uint32_t word1 = detail::read_u32(buffer_, (class_id_offset + 1) * vrt_word_size);
        return ClassIdValue::fromWords(word0, word1);
    }

    /**
     * @brief Set the class ID (if present)
     */
    void set_class_id(ClassIdValue cid) noexcept
        requires(has_class_id())
    {
        detail::write_u32(buffer_, class_id_offset * vrt_word_size, cid.word0());
        detail::write_u32(buffer_, (class_id_offset + 1) * vrt_word_size, cid.word1());
    }

    /**
     * @brief Get the timestamp (if present)
     */
    TimestampType timestamp() const noexcept
        requires(has_timestamp())
    {
        uint32_t tsi_val =
            (tsi != TsiType::none) ? detail::read_u32(buffer_, tsi_offset * vrt_word_size) : 0;
        uint64_t tsf_val =
            (tsf != TsfType::none) ? detail::read_u64(buffer_, tsf_offset * vrt_word_size) : 0;
        return TimestampType(tsi_val, tsf_val);
    }

    /**
     * @brief Set the timestamp (if present)
     */
    void set_timestamp(TimestampType ts) noexcept
        requires(has_timestamp())
    {
        if constexpr (tsi != TsiType::none) {
            detail::write_u32(buffer_, tsi_offset * vrt_word_size, ts.tsi());
        }
        if constexpr (tsf != TsfType::none) {
            detail::write_u64(buffer_, tsf_offset * vrt_word_size, ts.tsf());
        }
    }

    /**
     * @brief Get pointer to start of payload/CIF/context data
     */
    [[nodiscard]] uint8_t* payload_data() noexcept {
        return buffer_ + payload_offset * vrt_word_size;
    }

    /**
     * @brief Get const pointer to start of payload/CIF/context data
     */
    [[nodiscard]] const uint8_t* payload_data() const noexcept {
        return buffer_ + payload_offset * vrt_word_size;
    }

    /**
     * @brief Get the underlying buffer
     */
    [[nodiscard]] uint8_t* data() noexcept { return buffer_; }
    [[nodiscard]] const uint8_t* data() const noexcept { return buffer_; }

private:
    uint8_t* buffer_;
};

} // namespace vrtigo
