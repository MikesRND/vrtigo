#pragma once
// Owning packet bindings: DataPacket, ContextPacket

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include <vrtigo/dynamic.hpp>

#include "py_types.hpp"

#include <sstream>

namespace nb = nanobind;
using namespace nb::literals;

namespace vrtigo_python {

inline void bind_owning_packets(nb::module_& m) {
    // =========================================================================
    // DataPacket - owns its data, safe to hold across reader iterations
    // =========================================================================

    nb::class_<PyDataPacket>(
        m, "DataPacket",
        "VRT data packet (owns its data). Safe to hold across reader iterations.")
        .def_static(
            "from_bytes",
            [](nb::bytes data) {
                const auto* ptr = reinterpret_cast<const uint8_t*>(data.c_str());
                size_t size = data.size();
                return PyDataPacket::from_bytes({ptr, size});
            },
            "Create an owning data packet from bytes. Raises ParseError on failure.",
            "data"_a)
        // Forward packet metadata to underlying view
        .def_prop_ro("type", [](const PyDataPacket& p) { return p.view.type(); },
                     "Packet type")
        .def_prop_ro("packet_count",
                     [](const PyDataPacket& p) { return p.view.packet_count(); },
                     "Packet count (0-15)")
        .def_prop_ro("size_bytes",
                     [](const PyDataPacket& p) { return p.view.size_bytes(); },
                     "Packet size in bytes")
        .def_prop_ro("size_words",
                     [](const PyDataPacket& p) { return p.view.size_words(); },
                     "Packet size in 32-bit words")
        // Field presence
        .def_prop_ro("has_stream_id",
                     [](const PyDataPacket& p) { return p.view.has_stream_id(); },
                     "True if packet has stream ID")
        .def_prop_ro("has_class_id",
                     [](const PyDataPacket& p) { return p.view.has_class_id(); },
                     "True if packet has class ID")
        .def_prop_ro("has_timestamp",
                     [](const PyDataPacket& p) { return p.view.has_timestamp(); },
                     "True if packet has timestamp")
        .def_prop_ro("has_trailer",
                     [](const PyDataPacket& p) { return p.view.has_trailer(); },
                     "True if packet has trailer")
        // Field accessors
        .def_prop_ro("stream_id",
                     [](const PyDataPacket& p) { return p.view.stream_id(); },
                     "Stream ID if present, else None")
        .def_prop_ro("class_id",
                     [](const PyDataPacket& p) { return p.view.class_id(); },
                     "ClassId if present, else None")
        .def_prop_ro("timestamp",
                     [](const PyDataPacket& p) { return p.view.timestamp(); },
                     "Timestamp if present, else None")
        // Payload access
        .def_prop_ro("payload_size_bytes",
                     [](const PyDataPacket& p) {
                         return p.view.payload_size_bytes();
                     },
                     "Payload size in bytes")
        .def_prop_ro("payload_size_words",
                     [](const PyDataPacket& p) {
                         return p.view.payload_size_words();
                     },
                     "Payload size in 32-bit words")
        .def_prop_ro(
            "payload",
            [](const PyDataPacket& p) {
                auto span = p.view.payload();
                return nb::bytes(reinterpret_cast<const char*>(span.data()), span.size());
            },
            "Payload data as bytes")
        .def("__repr__", [](const PyDataPacket& p) {
            std::ostringstream oss;
            oss << "DataPacket(type=" << vrtigo::packet_type_string(p.view.type())
                << ", size=" << p.view.size_bytes() << " bytes";
            if (p.view.has_stream_id()) {
                oss << ", stream_id=0x" << std::hex << p.view.stream_id().value();
            }
            oss << ", payload=" << std::dec << p.view.payload_size_bytes() << " bytes)";
            return oss.str();
        });

    // =========================================================================
    // ContextPacket - owns its data, safe to hold across reader iterations
    // =========================================================================

    nb::class_<PyContextPacket>(
        m, "ContextPacket",
        "VRT context packet (owns its data). Safe to hold across reader iterations.")
        .def_static(
            "from_bytes",
            [](nb::bytes data) {
                const auto* ptr = reinterpret_cast<const uint8_t*>(data.c_str());
                size_t size = data.size();
                return PyContextPacket::from_bytes({ptr, size});
            },
            "Create an owning context packet from bytes. Raises ParseError on failure.",
            "data"_a)
        // Forward packet metadata to underlying view
        .def_prop_ro("type",
                     [](const PyContextPacket& p) { return p.view.type(); },
                     "Packet type")
        .def_prop_ro("packet_count",
                     [](const PyContextPacket& p) { return p.view.packet_count(); },
                     "Packet count (0-15)")
        .def_prop_ro("size_bytes",
                     [](const PyContextPacket& p) { return p.view.size_bytes(); },
                     "Packet size in bytes")
        .def_prop_ro("size_words",
                     [](const PyContextPacket& p) { return p.view.size_words(); },
                     "Packet size in 32-bit words")
        // Field presence
        .def_prop_ro("has_stream_id",
                     [](const PyContextPacket& p) { return p.view.has_stream_id(); },
                     "True if packet has stream ID")
        .def_prop_ro("has_class_id",
                     [](const PyContextPacket& p) { return p.view.has_class_id(); },
                     "True if packet has class ID")
        .def_prop_ro("has_timestamp",
                     [](const PyContextPacket& p) {
                         return p.view.has_timestamp();
                     },
                     "True if packet has timestamp")
        .def_prop_ro("has_trailer",
                     [](const PyContextPacket& p) { return p.view.has_trailer(); },
                     "True if packet has trailer")
        // Field accessors
        .def_prop_ro("stream_id",
                     [](const PyContextPacket& p) { return p.view.stream_id(); },
                     "Stream ID if present, else None")
        .def_prop_ro("class_id",
                     [](const PyContextPacket& p) { return p.view.class_id(); },
                     "ClassId if present, else None")
        .def_prop_ro("timestamp",
                     [](const PyContextPacket& p) { return p.view.timestamp(); },
                     "Timestamp if present, else None")
        // CIF accessors (context-specific)
        .def_prop_ro("cif0", [](const PyContextPacket& p) { return p.view.cif0(); },
                     "Context Indicator Field 0")
        .def_prop_ro("cif1", [](const PyContextPacket& p) { return p.view.cif1(); },
                     "Context Indicator Field 1 (0 if not present)")
        .def_prop_ro("cif2", [](const PyContextPacket& p) { return p.view.cif2(); },
                     "Context Indicator Field 2 (0 if not present)")
        .def_prop_ro("cif3", [](const PyContextPacket& p) { return p.view.cif3(); },
                     "Context Indicator Field 3 (0 if not present)")
        .def_prop_ro("change_indicator",
                     [](const PyContextPacket& p) {
                         return p.view.change_indicator();
                     },
                     "True if at least one context field has changed since last packet")
        .def("__repr__", [](const PyContextPacket& p) {
            std::ostringstream oss;
            oss << "ContextPacket(type=" << vrtigo::packet_type_string(p.view.type())
                << ", size=" << p.view.size_bytes() << " bytes";
            if (p.view.has_stream_id()) {
                oss << ", stream_id=0x" << std::hex << p.view.stream_id().value();
            }
            oss << ", cif0=0x" << std::hex << p.view.cif0() << ")";
            return oss.str();
        });
}

} // namespace vrtigo_python
