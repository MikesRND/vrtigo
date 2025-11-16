#pragma once

#include <variant>

#include <cstdint>
#include <vrtio/types.hpp>

#include "header.hpp"
#include "header_decode.hpp"
#include "header_indicators.hpp"

namespace vrtio::detail {

/**
 * @brief Read-only accessor for VRT packet header word (first 32 bits)
 *
 * Provides a unified interface to access header fields from either:
 * - A raw uint32_t word (for compile-time packets)
 * - A decoded DecodedHeader struct (for runtime packet views)
 *
 * This class consolidates header field access and eliminates duplication
 * between compile-time and runtime packet implementations.
 */
class HeaderView {
protected:
    // Tagged union - exactly one of these is active
    std::variant<const uint32_t*, const DecodedHeader*> source_;

    // Helper: extract packet type from raw word
    static PacketType extract_packet_type(uint32_t word) noexcept {
        return static_cast<PacketType>((word >> header::packet_type_shift) &
                                       header::packet_type_mask);
    }

    // Helper: extract packet count from raw word
    static uint8_t extract_packet_count(uint32_t word) noexcept {
        return static_cast<uint8_t>((word >> header::packet_count_shift) &
                                    header::packet_count_mask);
    }

    // Helper: extract packet size from raw word
    static uint16_t extract_packet_size(uint32_t word) noexcept {
        return static_cast<uint16_t>((word >> header::size_shift) & header::size_mask);
    }

    // Helper: extract class ID indicator from raw word
    static bool extract_has_class_id(uint32_t word) noexcept {
        return (word >> header::class_id_shift) & header::class_id_mask;
    }

    // Helper: extract TSI from raw word
    static TsiType extract_tsi(uint32_t word) noexcept {
        return static_cast<TsiType>((word >> header::tsi_shift) & header::tsi_mask);
    }

    // Helper: extract TSF from raw word
    static TsfType extract_tsf(uint32_t word) noexcept {
        return static_cast<TsfType>((word >> header::tsf_shift) & header::tsf_mask);
    }

public:
    /**
     * @brief Construct accessor from raw header word (compile-time packets)
     */
    explicit HeaderView(const uint32_t* header_word) noexcept : source_(header_word) {}

    /**
     * @brief Construct accessor from decoded header (runtime views)
     */
    explicit HeaderView(const DecodedHeader* decoded) noexcept : source_(decoded) {}

    /**
     * @brief Get packet type (4-bit field)
     */
    [[nodiscard]] PacketType packet_type() const noexcept {
        if (auto decoded = std::get_if<const DecodedHeader*>(&source_)) {
            return (*decoded)->type;
        }
        auto word = std::get<const uint32_t*>(source_);
        return extract_packet_type(*word);
    }

    /**
     * @brief Get packet count (4-bit sequence number)
     */
    [[nodiscard]] uint8_t packet_count() const noexcept {
        if (auto decoded = std::get_if<const DecodedHeader*>(&source_)) {
            return (*decoded)->packet_count;
        }
        auto word = std::get<const uint32_t*>(source_);
        return extract_packet_count(*word);
    }

    /**
     * @brief Get packet size in 32-bit words (16-bit field)
     */
    [[nodiscard]] uint16_t packet_size() const noexcept {
        if (auto decoded = std::get_if<const DecodedHeader*>(&source_)) {
            return (*decoded)->size_words;
        }
        auto word = std::get<const uint32_t*>(source_);
        return extract_packet_size(*word);
    }

    /**
     * @brief Check if class ID field is present (C bit)
     */
    [[nodiscard]] bool has_class_id() const noexcept {
        if (auto decoded = std::get_if<const DecodedHeader*>(&source_)) {
            return (*decoded)->has_class_id;
        }
        auto word = std::get<const uint32_t*>(source_);
        return extract_has_class_id(*word);
    }

    /**
     * @brief Check if trailer field is present (T bit, only valid for signal/ext data packets)
     */
    [[nodiscard]] bool has_trailer() const noexcept {
        if (auto decoded = std::get_if<const DecodedHeader*>(&source_)) {
            return decode_data_indicators(**decoded).has_trailer;
        }
        auto word = std::get<const uint32_t*>(source_);
        return decode_data_indicators(*word).has_trailer;
    }

    /**
     * @brief Get timestamp integer format type (TSI field, 2 bits)
     *
     * Returns the format of the integer timestamp component.
     * This is metadata ABOUT the timestamp, not the timestamp data itself.
     */
    [[nodiscard]] TsiType tsi_type() const noexcept {
        if (auto decoded = std::get_if<const DecodedHeader*>(&source_)) {
            return (*decoded)->tsi;
        }
        auto word = std::get<const uint32_t*>(source_);
        return extract_tsi(*word);
    }

    /**
     * @brief Get timestamp fractional format type (TSF field, 2 bits)
     *
     * Returns the format of the fractional timestamp component.
     * This is metadata ABOUT the timestamp, not the timestamp data itself.
     */
    [[nodiscard]] TsfType tsf_type() const noexcept {
        if (auto decoded = std::get_if<const DecodedHeader*>(&source_)) {
            return (*decoded)->tsf;
        }
        auto word = std::get<const uint32_t*>(source_);
        return extract_tsf(*word);
    }

    /**
     * @brief Check if integer timestamp component is present
     *
     * Derived from TSI field - true if TSI != none.
     */
    [[nodiscard]] bool has_timestamp_integer() const noexcept {
        return tsi_type() != TsiType::none;
    }

    /**
     * @brief Check if fractional timestamp component is present
     *
     * Derived from TSF field - true if TSF != none.
     */
    [[nodiscard]] bool has_timestamp_fractional() const noexcept {
        return tsf_type() != TsfType::none;
    }

    /**
     * @brief Get interpreted indicator bits for data packets (types 0-3)
     */
    [[nodiscard]] DataIndicators data_indicators() const noexcept {
        if (auto decoded = std::get_if<const DecodedHeader*>(&source_)) {
            return decode_data_indicators(**decoded);
        }
        auto word = std::get<const uint32_t*>(source_);
        return decode_data_indicators(*word);
    }

    /**
     * @brief Get interpreted indicator bits for context packets (types 4-5)
     */
    [[nodiscard]] ContextIndicators context_indicators() const noexcept {
        if (auto decoded = std::get_if<const DecodedHeader*>(&source_)) {
            return decode_context_indicators(**decoded);
        }
        auto word = std::get<const uint32_t*>(source_);
        return decode_context_indicators(*word);
    }

    /**
     * @brief Get interpreted indicator bits for command packets (types 6-7)
     */
    [[nodiscard]] CommandIndicators command_indicators() const noexcept {
        if (auto decoded = std::get_if<const DecodedHeader*>(&source_)) {
            return decode_command_indicators(**decoded);
        }
        auto word = std::get<const uint32_t*>(source_);
        return decode_command_indicators(*word);
    }
};

/**
 * @brief Mutable accessor for VRT packet header word (compile-time packets only)
 *
 * Extends HeaderView with setter methods for modifying header fields.
 * Only available for compile-time packets that operate on raw buffer words.
 */
class MutableHeaderView : public HeaderView {
    uint32_t* mutable_word_; // Separate mutable pointer for setters

    // Helper: update packet count in raw word
    static void update_packet_count(uint32_t& word, uint8_t count) noexcept {
        word = (word & ~(header::packet_count_mask << header::packet_count_shift)) |
               ((count & header::packet_count_mask) << header::packet_count_shift);
    }

    // Helper: update packet size in raw word
    static void update_packet_size(uint32_t& word, uint16_t size) noexcept {
        word = (word & ~(header::size_mask << header::size_shift)) |
               ((size & header::size_mask) << header::size_shift);
    }

public:
    /**
     * @brief Construct mutable accessor from raw header word
     *
     * Only available for compile-time packets. Runtime views are always const.
     */
    explicit MutableHeaderView(uint32_t* header_word) noexcept
        : HeaderView(const_cast<const uint32_t*>(header_word)),
          mutable_word_(header_word) {}

    /**
     * @brief Set packet count (4-bit sequence number)
     */
    void set_packet_count(uint8_t count) noexcept { update_packet_count(*mutable_word_, count); }

    /**
     * @brief Set packet size in 32-bit words (16-bit field)
     */
    void set_packet_size(uint16_t size) noexcept { update_packet_size(*mutable_word_, size); }

    // Note: Other setters (packet_type, class_id, tsi, tsf, etc.) are typically set
    // during packet construction and don't need to be modified afterward.
    // If needed, they can be added following the same pattern.
};

} // namespace vrtio::detail

// Public aliases
namespace vrtio {
using HeaderView = detail::HeaderView;
using MutableHeaderView = detail::MutableHeaderView;
} // namespace vrtio
