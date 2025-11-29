#pragma once

/**
 * @file typed.hpp
 * @brief Compile-time typed packets (high-performance, static structure)
 *
 * This header aggregates the compile-time packet types for creating packets
 * when the packet structure is known at compile time.
 *
 * Types provided:
 * - typed::DataPacket - Compile-time data packet template
 * - typed::ContextPacket - Compile-time context packet template (field-based)
 * - typed::ContextPacketBase - Low-level context packet template (CIF bitmask-based)
 *
 * Convenience aliases:
 * - typed::SignalDataPacket
 * - typed::SignalDataPacketNoId
 * - typed::ExtensionDataPacket
 * - typed::ExtensionDataPacketNoId
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
