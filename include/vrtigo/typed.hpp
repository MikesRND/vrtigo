#pragma once

/**
 * @file typed.hpp
 * @brief Compile-time typed packet builders (high-performance, static structure)
 *
 * This header aggregates the compile-time packet builder types for creating packets
 * when the packet structure is known at compile time.
 *
 * Types provided:
 * - typed::DataPacketBuilder - Compile-time data packet builder template
 * - typed::ContextPacketBuilder - Compile-time context packet builder template (field-based)
 * - typed::ContextPacketBuilderBase - Low-level context packet builder template (CIF bitmask-based)
 *
 * Convenience aliases:
 * - typed::SignalDataPacketBuilder
 * - typed::SignalDataPacketBuilderNoId
 * - typed::ExtensionDataPacketBuilder
 * - typed::ExtensionDataPacketBuilderNoId
 *
 * Use these types when:
 * - Creating packets for transmission
 * - Packet structure is fixed and known at compile time
 * - You need zero runtime overhead for field access
 *
 * For runtime packet parsing (receive side), see dynamic.hpp instead.
 */

#include "typed/context_packet.hpp"
#include "typed/data_packet.hpp"
