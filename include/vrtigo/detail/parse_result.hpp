#pragma once

#include "../expected.hpp"
#include "parse_error.hpp"

namespace vrtigo {

/**
 * @brief Result type for packet parsing operations
 *
 * Alias for expected<T, ParseError>. Holds either a successfully parsed packet
 * view or a ParseError with details about what went wrong.
 *
 * Usage:
 * @code
 *   auto result = parse_data_packet(buffer);
 *   if (result.has_value()) {
 *       auto payload = result->payload();  // Access via operator->
 *   } else {
 *       std::cerr << result.error().message() << "\n";
 *   }
 * @endcode
 *
 * Monadic operations:
 * @code
 *   parse_packet(buffer)
 *       .and_then(process_packet)
 *       .or_else(handle_error);
 * @endcode
 *
 * @tparam T The type of the successfully parsed packet view
 */
template <typename T>
using ParseResult = expected<T, ParseError>;

/**
 * @brief Factory function for creating parse errors
 *
 * Provides a concise way to create unexpected<ParseError> values
 * with designated initializers.
 *
 * Usage:
 * @code
 *   return make_parse_error(ValidationError::buffer_too_small,
 *                           PacketType::signal_data, header, buffer);
 * @endcode
 *
 * @param code The validation error code
 * @param type The packet type that was being parsed
 * @param header The decoded header (may be partial)
 * @param bytes The raw bytes that failed to parse
 * @return unexpected<ParseError> suitable for returning from parse functions
 */
inline auto make_parse_error(ValidationError code, PacketType type,
                             const detail::DecodedHeader& header,
                             std::span<const uint8_t> bytes) noexcept {
    return unexpected(
        ParseError{.code = code, .attempted_type = type, .header = header, .raw_bytes = bytes});
}

} // namespace vrtigo
