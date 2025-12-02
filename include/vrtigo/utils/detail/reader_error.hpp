#pragma once

#include <span>
#include <variant>

#include <cstdint>

#include "../../detail/header_decode.hpp"
#include "../../detail/parse_error.hpp"
#include "../../types.hpp"

namespace vrtigo::utils {

/**
 * @brief Represents end-of-stream (no error, just no more data)
 *
 * Returned when a reader reaches EOF during normal operation.
 */
struct EndOfStream {};

/**
 * @brief I/O-level error (distinct from parse errors)
 *
 * Represents system-level read failures, truncated data, etc.
 * Includes diagnostic context for debugging corrupt files.
 */
struct IOError {
    enum class Kind : uint8_t {
        read_error,       ///< File/socket read failed
        truncated_header, ///< Incomplete header read
        truncated_payload ///< Incomplete payload read
    };

    Kind kind;
    int errno_value{0};

    // Diagnostic context (matches ParseError fields for consistency)
    PacketType attempted_type{};            ///< From partial header read
    vrtigo::detail::DecodedHeader header{}; ///< Decoded partial header
    std::span<const uint8_t> raw_bytes{};   ///< Partial bytes for debugging

    /**
     * @brief Get human-readable error message
     */
    [[nodiscard]] const char* message() const noexcept {
        switch (kind) {
            case Kind::read_error:
                return "I/O read error";
            case Kind::truncated_header:
                return "Truncated header";
            case Kind::truncated_payload:
                return "Truncated payload";
        }
        return "Unknown I/O error";
    }
};

/**
 * @brief Unified reader error type
 *
 * A variant that can represent:
 * - EndOfStream: Normal end of data
 * - IOError: System-level I/O failure
 * - ParseError: Valid read but invalid packet data
 */
using ReaderError = std::variant<EndOfStream, IOError, vrtigo::ParseError>;

/**
 * @brief Check if error represents end of stream
 */
[[nodiscard]] inline bool is_eof(const ReaderError& e) noexcept {
    return std::holds_alternative<EndOfStream>(e);
}

/**
 * @brief Check if error is an I/O error
 */
[[nodiscard]] inline bool is_io_error(const ReaderError& e) noexcept {
    return std::holds_alternative<IOError>(e);
}

/**
 * @brief Check if error is a parse error
 */
[[nodiscard]] inline bool is_parse_error(const ReaderError& e) noexcept {
    return std::holds_alternative<vrtigo::ParseError>(e);
}

/**
 * @brief Get human-readable error message from any ReaderError
 */
[[nodiscard]] inline const char* error_message(const ReaderError& e) noexcept {
    return std::visit(
        [](auto&& err) -> const char* {
            using T = std::decay_t<decltype(err)>;
            if constexpr (std::is_same_v<T, EndOfStream>) {
                return "End of stream";
            } else if constexpr (std::is_same_v<T, IOError>) {
                return err.message();
            } else if constexpr (std::is_same_v<T, vrtigo::ParseError>) {
                return err.message();
            }
            return "Unknown error";
        },
        e);
}

} // namespace vrtigo::utils
