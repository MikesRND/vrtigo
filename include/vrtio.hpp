#pragma once

// VRTIO - Lightweight VITA Radio Transport Library
//
// A header-only C++20 library for creating and parsing VITA 49.2 VRT packets
// with zero allocation and compile-time validation.
//
// Phase 1 Features:
// - Signal Data packets (types 0-1)
// - Integer and fractional timestamps
// - Optional trailer
// - Compile-time size calculation
// - Zero-copy operations on user buffers
//
// Phase 2 Features:
    // - Context packets (type 4)
// - Runtime parsing with ContextPacketView
// - Compile-time creation with ContextPacket template
// - CIF0, CIF1, CIF2 field support
// - Variable-length field handling (GPS ASCII, Context Lists)
// - Full Class ID support with 32-bit PCC

// Core types and utilities
#include "vrtio/core/types.hpp"
#include "vrtio/core/header.hpp"
#include "vrtio/core/endian.hpp"
#include "vrtio/core/concepts.hpp"
#include "vrtio/core/class_id.hpp"
#include "vrtio/core/cif.hpp"

// Packet implementation
#include "vrtio/packet/data_packet.hpp"
#include "vrtio/packet/builder.hpp"
#include "vrtio/packet/context_packet_view.hpp"
#include "vrtio/packet/context_packet.hpp"

// Field access API (get, set, has for context packet fields)
#include "vrtio/fields.hpp"
