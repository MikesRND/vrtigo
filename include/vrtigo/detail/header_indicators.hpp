#pragma once

#include <cstdint>
#include <vrtigo/types.hpp>

#include "header.hpp"
#include "header_decode.hpp"

namespace vrtigo::detail {

struct DataIndicators {
    bool has_trailer = false;
    bool spectrum = false;
    bool nd0 = false;
};

struct ContextIndicators {
    bool timestamp_mode = false;
    bool nd0 = false;
};

struct CommandIndicators {
    bool acknowledge = false;
    bool cancel = false;
};

inline DataIndicators decode_data_indicators(uint32_t header_word) noexcept {
    DataIndicators indicators{};
    uint8_t type =
        static_cast<uint8_t>((header_word >> header::packet_type_shift) & header::packet_type_mask);
    if (type <= 3) {
        indicators.has_trailer =
            (header_word >> header::indicator_bit_26_shift) & header::indicator_bit_mask;
        indicators.nd0 =
            (header_word >> header::indicator_bit_25_shift) & header::indicator_bit_mask;
        indicators.spectrum =
            (header_word >> header::indicator_bit_24_shift) & header::indicator_bit_mask;
    }
    return indicators;
}

inline ContextIndicators decode_context_indicators(uint32_t header_word) noexcept {
    ContextIndicators indicators{};
    uint8_t type =
        static_cast<uint8_t>((header_word >> header::packet_type_shift) & header::packet_type_mask);
    if (type == 4 || type == 5) {
        indicators.nd0 =
            (header_word >> header::indicator_bit_25_shift) & header::indicator_bit_mask;
        indicators.timestamp_mode =
            (header_word >> header::indicator_bit_24_shift) & header::indicator_bit_mask;
    }
    return indicators;
}

inline CommandIndicators decode_command_indicators(uint32_t header_word) noexcept {
    CommandIndicators indicators{};
    uint8_t type =
        static_cast<uint8_t>((header_word >> header::packet_type_shift) & header::packet_type_mask);
    if (type == 6 || type == 7) {
        indicators.acknowledge =
            (header_word >> header::indicator_bit_26_shift) & header::indicator_bit_mask;
        indicators.cancel =
            (header_word >> header::indicator_bit_24_shift) & header::indicator_bit_mask;
    }
    return indicators;
}

inline DataIndicators decode_data_indicators(const DecodedHeader& header) noexcept {
    DataIndicators indicators{};
    if (static_cast<uint8_t>(header.type) <= 3) {
        indicators.has_trailer = header.trailer_included;
        indicators.spectrum = header.signal_spectrum;
        indicators.nd0 = header.nd0;
    }
    return indicators;
}

inline ContextIndicators decode_context_indicators(const DecodedHeader& header) noexcept {
    ContextIndicators indicators{};
    if (header.type == PacketType::context || header.type == PacketType::extension_context) {
        indicators.timestamp_mode = header.context_tsm;
        indicators.nd0 = header.nd0;
    }
    return indicators;
}

inline CommandIndicators decode_command_indicators(const DecodedHeader& header) noexcept {
    CommandIndicators indicators{};
    if (header.type == PacketType::command || header.type == PacketType::extension_command) {
        indicators.acknowledge = header.command_ack;
        indicators.cancel = header.command_cancel;
    }
    return indicators;
}

} // namespace vrtigo::detail
