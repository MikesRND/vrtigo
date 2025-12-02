#pragma once

// VRTIGO Expected Type
//
// Exposes tl::expected in the vrtigo namespace for consistent error handling.
// This provides a std::expected-compatible API (C++23) using the TartanLlama
// implementation for C++20 compatibility.
//
// Usage:
//   vrtigo::expected<T, E> result = some_operation();
//   if (result.has_value()) {
//       process(*result);
//   } else {
//       handle(result.error());
//   }
//
// Monadic operations:
//   result.and_then(f)  - chain on success
//   result.map(f)       - transform value
//   result.or_else(f)   - chain on error
//   result.map_error(f) - transform error

#include <tl/expected.hpp>

namespace vrtigo {

using tl::expected;
using tl::make_unexpected;
using tl::unexpect;
using tl::unexpect_t;
using tl::unexpected;

} // namespace vrtigo
