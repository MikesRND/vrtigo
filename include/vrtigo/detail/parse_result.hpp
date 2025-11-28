#pragma once

#include <variant>

#include <cassert>

#include "parse_error.hpp"

namespace vrtigo {

/**
 * @brief Result type for packet parsing operations
 *
 * A type-safe wrapper that holds either a successfully parsed packet view
 * or a ParseError with details about what went wrong. Similar to std::expected
 * but available in C++20.
 *
 * Usage:
 * @code
 *   auto result = parse_data_packet(buffer);
 *   if (result) {
 *       auto payload = result->payload();  // Access via operator->
 *   } else {
 *       std::cerr << result.error().message() << "\n";
 *   }
 * @endcode
 *
 * @tparam T The type of the successfully parsed packet view
 */
template <typename T>
class [[nodiscard]] ParseResult {
    std::variant<T, ParseError> storage_;

public:
    /// Type of the success value
    using value_type = T;
    /// Type of the error value
    using error_type = ParseError;

    /**
     * @brief Construct a successful result
     * @param value The successfully parsed packet view
     */
    ParseResult(T value) noexcept : storage_(std::move(value)) {}

    /**
     * @brief Construct an error result
     * @param error The parse error (trivially copyable, no move needed)
     */
    ParseResult(const ParseError& error) noexcept : storage_(error) {}

    // ==========
    // Observers
    // ==========

    /**
     * @brief Check if parsing succeeded
     * @return true if this holds a valid packet view, false if it holds an error
     */
    [[nodiscard]] bool ok() const noexcept { return std::holds_alternative<T>(storage_); }

    /**
     * @brief Boolean conversion for use in conditionals
     * @return true if parsing succeeded
     */
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }

    // ==========
    // Value Accessors
    // ==========

    /**
     * @brief Get the parsed packet view (const lvalue reference)
     *
     * Precondition: ok() == true
     * In debug builds, asserts if called on an error result.
     *
     * @return Const reference to the packet view
     */
    [[nodiscard]] const T& value() const& {
        assert(ok() && "value() called on error result");
        return std::get<T>(storage_);
    }

    /**
     * @brief Move the parsed packet view out (rvalue reference)
     *
     * Precondition: ok() == true
     * In debug builds, asserts if called on an error result.
     *
     * @return Rvalue reference to the packet view
     */
    [[nodiscard]] T&& value() && {
        assert(ok() && "value() called on error result");
        return std::get<T>(std::move(storage_));
    }

    // ==========
    // Error Accessor
    // ==========

    /**
     * @brief Get the parse error
     *
     * Precondition: ok() == false
     * In debug builds, asserts if called on a success result.
     *
     * @return Const reference to the ParseError
     */
    [[nodiscard]] const ParseError& error() const& {
        assert(!ok() && "error() called on success result");
        return std::get<ParseError>(storage_);
    }

    // ==========
    // Pointer-like Access
    // ==========

    /**
     * @brief Arrow operator for convenient member access
     *
     * Precondition: ok() == true
     *
     * @return Const pointer to the packet view
     */
    [[nodiscard]] const T* operator->() const {
        assert(ok() && "operator-> called on error result");
        return &std::get<T>(storage_);
    }

    /**
     * @brief Dereference operator
     *
     * Precondition: ok() == true
     *
     * @return Const reference to the packet view
     */
    [[nodiscard]] const T& operator*() const& {
        assert(ok() && "operator* called on error result");
        return std::get<T>(storage_);
    }
};

} // namespace vrtigo
