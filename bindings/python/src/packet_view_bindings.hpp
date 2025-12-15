#pragma once
// Packet view bindings: DataPacketView, ContextPacketView

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include <vrtigo/dynamic.hpp>

#include <sstream>
#include <span>

namespace nb = nanobind;
using namespace nb::literals;

namespace vrtigo_python {

inline void bind_packet_views(nb::module_& m) {
    // =========================================================================
    // DataPacketView
    // =========================================================================

    nb::class_<vrtigo::dynamic::DataPacketView>(
        m, "DataPacketView", "Runtime parser for VRT data packets")
        .def_static(
            "parse",
            [](nb::bytes data) {
                const auto* ptr = reinterpret_cast<const uint8_t*>(data.c_str());
                size_t size = data.size();
                std::span<const uint8_t> buffer(ptr, size);

                auto result = vrtigo::dynamic::DataPacketView::parse(buffer);
                if (!result.has_value()) {
                    throw std::runtime_error(result.error().message());
                }
                return result.value();
            },
            "Parse a data packet from bytes. Raises ParseError on failure.\n\n"
            "WARNING: The returned view references the input bytes. Do not let\n"
            "the bytes object be garbage collected while using the view.",
            "data"_a,
            nb::keep_alive<0, 1>())
        .def_prop_ro("type", &vrtigo::dynamic::DataPacketView::type, "Packet type")
        .def_prop_ro("packet_count", &vrtigo::dynamic::DataPacketView::packet_count,
                     "Packet count (0-15)")
        .def_prop_ro("size_bytes", &vrtigo::dynamic::DataPacketView::size_bytes,
                     "Packet size in bytes")
        .def_prop_ro("size_words", &vrtigo::dynamic::DataPacketView::size_words,
                     "Packet size in 32-bit words")
        .def_prop_ro("has_stream_id", &vrtigo::dynamic::DataPacketView::has_stream_id,
                     "True if packet has stream ID")
        .def_prop_ro("has_class_id", &vrtigo::dynamic::DataPacketView::has_class_id,
                     "True if packet has class ID")
        .def_prop_ro("has_timestamp", &vrtigo::dynamic::DataPacketView::has_timestamp,
                     "True if packet has timestamp")
        .def_prop_ro("has_trailer", &vrtigo::dynamic::DataPacketView::has_trailer,
                     "True if packet has trailer")
        .def_prop_ro("stream_id", &vrtigo::dynamic::DataPacketView::stream_id,
                     "Stream ID if present, else None")
        .def_prop_ro("class_id", &vrtigo::dynamic::DataPacketView::class_id,
                     "ClassId if present, else None")
        .def_prop_ro("timestamp", &vrtigo::dynamic::DataPacketView::timestamp,
                     "Timestamp if present, else None")
        .def_prop_ro("payload_size_bytes",
                     &vrtigo::dynamic::DataPacketView::payload_size_bytes,
                     "Payload size in bytes")
        .def_prop_ro("payload_size_words",
                     &vrtigo::dynamic::DataPacketView::payload_size_words,
                     "Payload size in 32-bit words")
        .def_prop_ro(
            "payload",
            [](const vrtigo::dynamic::DataPacketView& pkt) {
                auto span = pkt.payload();
                return nb::bytes(reinterpret_cast<const char*>(span.data()), span.size());
            },
            "Payload data as bytes")
        .def("__repr__", [](const vrtigo::dynamic::DataPacketView& pkt) {
            std::ostringstream oss;
            oss << "DataPacketView(type=" << vrtigo::packet_type_string(pkt.type())
                << ", size=" << pkt.size_bytes() << " bytes";
            if (pkt.has_stream_id()) {
                oss << ", stream_id=0x" << std::hex << pkt.stream_id().value();
            }
            oss << ", payload=" << std::dec << pkt.payload_size_bytes() << " bytes)";
            return oss.str();
        });

    // =========================================================================
    // ContextPacketView
    // =========================================================================

    nb::class_<vrtigo::dynamic::ContextPacketView>(
        m, "ContextPacketView", "Runtime parser for VRT context packets")
        .def_static(
            "parse",
            [](nb::bytes data) {
                const auto* ptr = reinterpret_cast<const uint8_t*>(data.c_str());
                size_t size = data.size();
                std::span<const uint8_t> buffer(ptr, size);

                auto result = vrtigo::dynamic::ContextPacketView::parse(buffer);
                if (!result.has_value()) {
                    throw std::runtime_error(result.error().message());
                }
                return result.value();
            },
            "Parse a context packet from bytes. Raises ParseError on failure.\n\n"
            "WARNING: The returned view references the input bytes. Do not let\n"
            "the bytes object be garbage collected while using the view.",
            "data"_a,
            nb::keep_alive<0, 1>())
        .def_prop_ro("type", &vrtigo::dynamic::ContextPacketView::type, "Packet type")
        .def_prop_ro("packet_count", &vrtigo::dynamic::ContextPacketView::packet_count,
                     "Packet count (0-15)")
        .def_prop_ro("size_bytes", &vrtigo::dynamic::ContextPacketView::size_bytes,
                     "Packet size in bytes")
        .def_prop_ro("size_words", &vrtigo::dynamic::ContextPacketView::size_words,
                     "Packet size in 32-bit words")
        .def_prop_ro("has_stream_id", &vrtigo::dynamic::ContextPacketView::has_stream_id,
                     "True if packet has stream ID")
        .def_prop_ro("has_class_id", &vrtigo::dynamic::ContextPacketView::has_class_id,
                     "True if packet has class ID")
        .def_prop_ro("has_timestamp", &vrtigo::dynamic::ContextPacketView::has_timestamp,
                     "True if packet has timestamp")
        .def_prop_ro("has_trailer", &vrtigo::dynamic::ContextPacketView::has_trailer,
                     "True if packet has trailer")
        .def_prop_ro("stream_id", &vrtigo::dynamic::ContextPacketView::stream_id,
                     "Stream ID if present, else None")
        .def_prop_ro("class_id", &vrtigo::dynamic::ContextPacketView::class_id,
                     "ClassId if present, else None")
        .def_prop_ro("timestamp", &vrtigo::dynamic::ContextPacketView::timestamp,
                     "Timestamp if present, else None")
        .def_prop_ro("cif0", &vrtigo::dynamic::ContextPacketView::cif0,
                     "Context Indicator Field 0")
        .def_prop_ro("cif1", &vrtigo::dynamic::ContextPacketView::cif1,
                     "Context Indicator Field 1 (0 if not present)")
        .def_prop_ro("cif2", &vrtigo::dynamic::ContextPacketView::cif2,
                     "Context Indicator Field 2 (0 if not present)")
        .def_prop_ro("cif3", &vrtigo::dynamic::ContextPacketView::cif3,
                     "Context Indicator Field 3 (0 if not present)")
        .def_prop_ro("change_indicator", &vrtigo::dynamic::ContextPacketView::change_indicator,
                     "True if at least one context field has changed since last packet")
        .def("__repr__", [](const vrtigo::dynamic::ContextPacketView& pkt) {
            std::ostringstream oss;
            oss << "ContextPacketView(type=" << vrtigo::packet_type_string(pkt.type())
                << ", size=" << pkt.size_bytes() << " bytes";
            if (pkt.has_stream_id()) {
                oss << ", stream_id=0x" << std::hex << pkt.stream_id().value();
            }
            oss << ", cif0=0x" << std::hex << pkt.cif0() << ")";
            return oss.str();
        });
}

} // namespace vrtigo_python
