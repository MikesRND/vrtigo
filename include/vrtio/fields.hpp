#pragma once

/// @file fields.hpp
/// @brief Complete field access API for VITA 49.2 Context Information Fields
///
/// This header provides a unified, type-safe interface for accessing all supported
/// CIF fields in context packets. Use this header to access any of the 70 supported
/// fields across CIF0, CIF1, and CIF2.
///
/// Example usage:
/// @code
///     using namespace vrtio;
///     using namespace vrtio::field;
///
///     // Reading fields
///     auto bw = get(packet, bandwidth);
///     if (bw) {
///         std::cout << "Bandwidth: " << *bw << " Hz\n";
///     }
///
///     // Writing fields (mutable packets only)
///     set(packet, sample_rate, 1'000'000);
///
///     // Check field presence
///     if (has(packet, gain)) {
///         auto gain_val = get_unchecked(packet, gain);
///     }
/// @endcode

// Core field types and views
#include "vrtio/core/field_values.hpp"

// Field tag definitions (all 70 supported fields)
#include "vrtio/core/fields.hpp"

// Field trait specializations (internal)
#include "vrtio/core/detail/field_traits.hpp"

// Public API functions: get(), set(), has(), get_unchecked()
#include "vrtio/core/field_access.hpp"

// Packet types that work with field access
#include "vrtio/packet/context_packet.hpp"
#include "vrtio/packet/context_packet_view.hpp"
