#pragma once

#include <span>

#include <cstdint>
#include <vrtigo/types.hpp>

#include "header_decode.hpp"

namespace vrtigo {

/**
 * @brief Error information from failed packet parsing
 *
 * Contains all context needed to diagnose parsing failures, including
 * the validation error, packet type from header, decoded header fields,
 * and the raw bytes that failed to parse.
 *
 * This is a trivially copyable type (span is just pointer + size).
 */
struct ParseError {
    ValidationError code;         ///< The validation error that occurred
    PacketType attempted_type;    ///< Packet type detected from header
    detail::DecodedHeader header; ///< Decoded header information (may be partial on early failures)
    std::span<const uint8_t> raw_bytes; ///< Raw packet bytes for debugging

    /**
     * @brief Get a human-readable error message
     * @return Static string describing the validation error
     */
    [[nodiscard]] const char* message() const noexcept { return validation_error_string(code); }
};

} // namespace vrtigo
