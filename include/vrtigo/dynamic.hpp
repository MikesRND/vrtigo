#pragma once

/**
 * @file dynamic.hpp
 * @brief Runtime packet parsing (flexible, dynamic structure)
 *
 * This header aggregates the runtime packet types for parsing received packets
 * when the packet structure is not known at compile time.
 *
 * Types provided:
 * - dynamic::DataPacketView - Runtime parser for signal/extension data packets
 * - dynamic::ContextPacketView - Runtime parser for context packets
 *
 * Use these types when:
 * - Receiving packets from external sources
 * - Packet structure varies or is determined at runtime
 * - You need automatic validation on parse
 *
 * For compile-time typed packets (transmit side), see typed.hpp instead.
 */

#include "dynamic/context_packet.hpp"
#include "dynamic/data_packet.hpp"
