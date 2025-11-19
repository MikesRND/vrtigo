#pragma once

// VRTIO - Lightweight VITA Radio Transport Library
//
// A header-only C++20 library for creating and parsing VITA 49.2 VRT packets
// with zero allocation and compile-time validation.
//
// Data Packet Features:
// - Signal Data packets (types 0-1)
// - Extension Data packets (types 2-3)
// - Integer and fractional timestamps
// - Optional trailer
// - Compile-time size calculation
// - Zero-copy operations on user buffers
// - Builder pattern for fluent construction
//
// Context Packet Features:
// - Context packets (types 4-5)
// - Runtime parsing with RuntimeContextPacket
// - Compile-time creation with ContextPacket template
// - CIF0, CIF1, CIF2, CIF3 field support (70+ fields)
// - Variable-length field handling (GPS ASCII, Context Association Lists)
// - Full Class ID support with 24-bit OUI and 32-bit PCC
// - Unified field access API via operator[]

// ====================
// Public API
// ====================

// Core types and enums (users need these for parameters and return values)
#include "vrtio/types.hpp"

// Timestamp types (users instantiate these directly)
#include "vrtio/timestamp.hpp"

// ClassId types (users instantiate these directly)
#include "vrtio/class_id.hpp"

// Field tags for context packet field access
#include "vrtio/field_tags.hpp"

// ====================
// Implementation
// ====================

// Packet implementations (exposed via this header but users don't include detail/ directly)
#include "vrtio/detail/builder.hpp"
#include "vrtio/detail/context_packet.hpp"
#include "vrtio/detail/data_packet.hpp"
#include "vrtio/detail/runtime_context_packet.hpp"
#include "vrtio/detail/runtime_data_packet.hpp"

// ====================
// Convenience Aliases
// ====================

namespace vrtio {

// Convenient aliases for common packet types
template <typename ClassIdType, typename TimeStampType, Trailer TrailerType, size_t PayloadWords>
using SignalDataPacket =
    DataPacket<PacketType::signal_data, ClassIdType, TimeStampType, TrailerType, PayloadWords>;

template <typename ClassIdType, typename TimeStampType, Trailer TrailerType, size_t PayloadWords>
using ExtensionDataPacket =
    DataPacket<PacketType::extension_data, ClassIdType, TimeStampType, TrailerType, PayloadWords>;

} // namespace vrtio
