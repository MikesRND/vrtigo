#pragma once

// VRTIO - Lightweight VITA Radio Transport Library
// Version: 1.0.0 (Phase 1)
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

// Core types and utilities
#include "vrtio/core/types.hpp"
#include "vrtio/core/endian.hpp"
#include "vrtio/core/concepts.hpp"

// Packet implementation
#include "vrtio/packet/signal_packet.hpp"
#include "vrtio/packet/builder.hpp"
