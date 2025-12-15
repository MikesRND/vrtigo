// VRTIGO Python Bindings
// Main module entry point - includes component bindings

#include <nanobind/nanobind.h>

// Binding components
#include "core_bindings.hpp"
#include "error_bindings.hpp"
#include "owning_packet_bindings.hpp"
#include "packet_view_bindings.hpp"
#include "reader_bindings.hpp"
#include "sample_framer_bindings.hpp"

namespace nb = nanobind;

// Define the exception type pointers (declared extern in py_types.hpp)
namespace vrtigo_python {
PyObject* parse_error_type = nullptr;
PyObject* vrt_io_error_type = nullptr;
} // namespace vrtigo_python

NB_MODULE(vrtigo, m) {
    m.doc() = "VRTIGO - VRT (VITA 49.2) packet library";

    // Bind components in dependency order:
    // 1. Core types (enums, ClassId, Timestamp) - no dependencies
    vrtigo_python::bind_core(m);

    // 2. Error types (sets parse_error_type, vrt_io_error_type) - needs core enums
    vrtigo_python::bind_errors(m);

    // 3. Packet views (DataPacketView, ContextPacketView) - needs core types
    vrtigo_python::bind_packet_views(m);

    // 4. Owning packets (DataPacket, ContextPacket) - needs parse_error_type
    vrtigo_python::bind_owning_packets(m);

    // 5. Readers (VRTFileReader, UDPVRTReader) - needs owning packets, error types
    vrtigo_python::bind_readers(m);

    // 6. SampleFramer - needs owning packets, error types
    vrtigo_python::bind_sample_framer(m);
}
