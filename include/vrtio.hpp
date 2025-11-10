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
// - Runtime parsing with ContextPacketView
// - Compile-time creation with ContextPacket template
// - CIF0, CIF1, CIF2, CIF3 field support (70+ fields)
// - Variable-length field handling (GPS ASCII, Context Association Lists)
// - Full Class ID support with 24-bit OUI and 32-bit PCC
// - Unified field access API (get/set/has)

// Core types and utilities
#include "vrtio/core/cif.hpp"
#include "vrtio/core/class_id.hpp"
#include "vrtio/core/concepts.hpp"
#include "vrtio/core/endian.hpp"
#include "vrtio/core/header.hpp"
#include "vrtio/core/types.hpp"

// Packet implementation
#include "vrtio/packet/builder.hpp"
#include "vrtio/packet/context_packet.hpp"
#include "vrtio/packet/context_packet_view.hpp"
#include "vrtio/packet/data_packet.hpp"

// Field access API (get, set, has for context packet fields)
#include "vrtio/fields.hpp"
