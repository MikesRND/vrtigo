#pragma once
// Core bindings: Enums, ClassId, Timestamp, constants

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <vrtigo/class_id.hpp>
#include <vrtigo/timestamp.hpp>
#include <vrtigo/types.hpp>

#include <sstream>

namespace nb = nanobind;
using namespace nb::literals;

namespace vrtigo_python {

inline void bind_core(nb::module_& m) {
    // =========================================================================
    // Enums
    // =========================================================================

    nb::enum_<vrtigo::PacketType>(m, "PacketType", "VRT packet types (VITA 49.2 standard)")
        .value("signal_data_no_id", vrtigo::PacketType::signal_data_no_id,
               "Signal data without stream identifier")
        .value("signal_data", vrtigo::PacketType::signal_data,
               "Signal data with stream identifier")
        .value("extension_data_no_id", vrtigo::PacketType::extension_data_no_id,
               "Extension data without stream identifier")
        .value("extension_data", vrtigo::PacketType::extension_data,
               "Extension data with stream identifier")
        .value("context", vrtigo::PacketType::context, "Context packet")
        .value("extension_context", vrtigo::PacketType::extension_context,
               "Extension context packet")
        .value("command", vrtigo::PacketType::command, "Command packet (VITA 49.2)")
        .value("extension_command", vrtigo::PacketType::extension_command,
               "Extension command packet (VITA 49.2)")
        .def("__str__",
             [](vrtigo::PacketType t) { return std::string(vrtigo::packet_type_string(t)); })
        .def("__repr__", [](vrtigo::PacketType t) {
            return std::string("PacketType.") + vrtigo::packet_type_string(t);
        });

    nb::enum_<vrtigo::TsiType>(m, "TsiType", "Integer timestamp types (TSI field)")
        .value("none", vrtigo::TsiType::none, "No integer timestamp")
        .value("utc", vrtigo::TsiType::utc, "UTC time")
        .value("gps", vrtigo::TsiType::gps, "GPS time")
        .value("other", vrtigo::TsiType::other, "Other/application-defined")
        .def("__str__",
             [](vrtigo::TsiType t) { return std::string(vrtigo::tsi_type_string(t)); })
        .def("__repr__", [](vrtigo::TsiType t) {
            return std::string("TsiType.") + vrtigo::tsi_type_string(t);
        });

    nb::enum_<vrtigo::TsfType>(m, "TsfType", "Fractional timestamp types (TSF field)")
        .value("none", vrtigo::TsfType::none, "No fractional timestamp")
        .value("sample_count", vrtigo::TsfType::sample_count, "Sample count timestamp")
        .value("real_time", vrtigo::TsfType::real_time, "Real-time picosecond timestamp")
        .value("free_running", vrtigo::TsfType::free_running, "Free-running count")
        .def("__str__",
             [](vrtigo::TsfType t) { return std::string(vrtigo::tsf_type_string(t)); })
        .def("__repr__", [](vrtigo::TsfType t) {
            return std::string("TsfType.") + vrtigo::tsf_type_string(t);
        });

    nb::enum_<vrtigo::ValidationError>(m, "ValidationError",
                                       "Validation error codes for packet parsing")
        .value("none", vrtigo::ValidationError::none, "No error, packet is valid")
        .value("buffer_too_small", vrtigo::ValidationError::buffer_too_small,
               "Buffer size smaller than declared packet size")
        .value("packet_type_mismatch", vrtigo::ValidationError::packet_type_mismatch,
               "Packet type in header doesn't match template")
        .value("class_id_bit_mismatch", vrtigo::ValidationError::class_id_bit_mismatch,
               "Class ID indicator doesn't match template")
        .value("tsi_mismatch", vrtigo::ValidationError::tsi_mismatch,
               "TSI field doesn't match template")
        .value("tsf_mismatch", vrtigo::ValidationError::tsf_mismatch,
               "TSF field doesn't match template")
        .value("trailer_bit_mismatch", vrtigo::ValidationError::trailer_bit_mismatch,
               "Trailer indicator doesn't match template")
        .value("size_field_mismatch", vrtigo::ValidationError::size_field_mismatch,
               "Size field doesn't match expected packet size")
        .value("invalid_packet_type", vrtigo::ValidationError::invalid_packet_type,
               "Reserved or unsupported packet type value")
        .value("unsupported_field", vrtigo::ValidationError::unsupported_field,
               "Packet contains fields not supported by this implementation")
        .def("__str__",
             [](vrtigo::ValidationError e) {
                 return std::string(vrtigo::validation_error_string(e));
             })
        .def("__repr__", [](vrtigo::ValidationError e) {
            const char* names[] = {"none",
                                   "buffer_too_small",
                                   "packet_type_mismatch",
                                   "class_id_bit_mismatch",
                                   "tsi_mismatch",
                                   "tsf_mismatch",
                                   "trailer_bit_mismatch",
                                   "size_field_mismatch",
                                   "invalid_packet_type",
                                   "unsupported_field"};
            auto idx = static_cast<size_t>(e);
            if (idx < sizeof(names) / sizeof(names[0])) {
                return std::string("ValidationError.") + names[idx];
            }
            return std::string("ValidationError.unknown");
        });

    // Helper functions
    m.def("is_signal_data", &vrtigo::is_signal_data, "Check if packet type is signal data",
          "type"_a);
    m.def("has_stream_identifier", &vrtigo::has_stream_identifier,
          "Check if packet type includes stream ID", "type"_a);

    // Constants
    m.attr("VRT_WORD_SIZE") = vrtigo::vrt_word_size;
    m.attr("VRT_WORD_BITS") = vrtigo::vrt_word_bits;
    m.attr("MAX_PACKET_WORDS") = vrtigo::max_packet_words;
    m.attr("MAX_PACKET_BYTES") = vrtigo::max_packet_bytes;
    m.attr("PICOSECONDS_PER_SECOND") = vrtigo::picoseconds_per_second;

    // =========================================================================
    // ClassId
    // =========================================================================

    nb::class_<vrtigo::ClassIdValue>(m, "ClassId", "VRT Class Identifier (OUI, ICC, PCC)")
        .def(nb::init<uint32_t, uint16_t, uint16_t, uint8_t>(),
             "Create a ClassId from OUI, ICC, PCC, and PBC values", "oui"_a, "icc"_a,
             "pcc"_a = 0, "pbc"_a = 0)
        .def_static(
            "from_words", &vrtigo::ClassIdValue::fromWords,
            "Create a ClassId from two 32-bit words as stored in a packet", "word0"_a, "word1"_a)
        .def_prop_ro("oui", &vrtigo::ClassIdValue::oui,
                     "Organizationally Unique Identifier (24 bits)")
        .def_prop_ro("icc", &vrtigo::ClassIdValue::icc, "Information Class Code (16 bits)")
        .def_prop_ro("pcc", &vrtigo::ClassIdValue::pcc, "Packet Class Code (16 bits)")
        .def_prop_ro("pbc", &vrtigo::ClassIdValue::pbc, "Pad Bit Count (5 bits)")
        .def_prop_ro("word0", &vrtigo::ClassIdValue::word0, "First 32-bit word encoding")
        .def_prop_ro("word1", &vrtigo::ClassIdValue::word1, "Second 32-bit word encoding")
        .def("__eq__",
             [](const vrtigo::ClassIdValue& a, const vrtigo::ClassIdValue& b) {
                 return a.oui() == b.oui() && a.icc() == b.icc() && a.pcc() == b.pcc() &&
                        a.pbc() == b.pbc();
             })
        .def("__repr__", [](const vrtigo::ClassIdValue& c) {
            std::ostringstream oss;
            oss << "ClassId(oui=0x" << std::hex << c.oui() << ", icc=0x" << c.icc()
                << ", pcc=0x" << c.pcc() << ")";
            return oss.str();
        });

    // =========================================================================
    // Timestamp
    // =========================================================================

    nb::class_<vrtigo::TimestampValue>(m, "Timestamp",
                                       "Type-erased VRT timestamp (TSI + TSF)")
        .def_prop_ro("tsi", &vrtigo::TimestampValue::tsi,
                     "Integer timestamp value (seconds)")
        .def_prop_ro("tsf", &vrtigo::TimestampValue::tsf,
                     "Fractional timestamp value (picoseconds for real_time)")
        .def_prop_ro("tsi_kind", &vrtigo::TimestampValue::tsi_kind,
                     "Integer timestamp type (TsiType)")
        .def_prop_ro("tsf_kind", &vrtigo::TimestampValue::tsf_kind,
                     "Fractional timestamp type (TsfType)")
        .def_prop_ro("has_tsi", &vrtigo::TimestampValue::has_tsi,
                     "True if integer timestamp is present")
        .def_prop_ro("has_tsf", &vrtigo::TimestampValue::has_tsf,
                     "True if fractional timestamp is present")
        .def("__eq__",
             [](const vrtigo::TimestampValue& a, const vrtigo::TimestampValue& b) {
                 return a == b;
             })
        .def("__repr__", [](const vrtigo::TimestampValue& ts) {
            std::ostringstream oss;
            oss << "Timestamp(tsi=" << ts.tsi() << ", tsf=" << ts.tsf()
                << ", tsi_kind=" << vrtigo::tsi_type_string(ts.tsi_kind())
                << ", tsf_kind=" << vrtigo::tsf_type_string(ts.tsf_kind()) << ")";
            return oss.str();
        });
}

} // namespace vrtigo_python
