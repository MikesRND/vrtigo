#pragma once
// Error bindings: FrameError, IOErrorKind, VRTIOError, PayloadAlignmentError, IngestResult

#include <nanobind/nanobind.h>

#include <vrtigo/utils/detail/reader_error.hpp>
#include <vrtigo/utils/sample_framer.hpp>

#include "py_types.hpp"

#include <sstream>

namespace nb = nanobind;
using namespace nb::literals;

namespace vrtigo_python {

inline void bind_errors(nb::module_& m) {
    // =========================================================================
    // FrameError enum (from SampleFramer)
    // =========================================================================

    nb::enum_<vrtigo::utils::FrameError>(m, "FrameError",
                                         "Errors from sample framing operations")
        .value("payload_not_aligned", vrtigo::utils::FrameError::payload_not_aligned,
               "Payload size not a multiple of sample size")
        .value("stop_requested", vrtigo::utils::FrameError::stop_requested,
               "Callback returned false to stop processing")
        .def("__str__", [](vrtigo::utils::FrameError e) {
            return std::string(vrtigo::utils::frame_error_string(e));
        });

    // =========================================================================
    // IOErrorKind enum (from reader errors)
    // =========================================================================

    nb::enum_<vrtigo::utils::IOError::Kind>(m, "IOErrorKind",
                                            "I/O error types for packet reading")
        .value("read_error", vrtigo::utils::IOError::Kind::read_error,
               "File/socket read failed")
        .value("truncated_header", vrtigo::utils::IOError::Kind::truncated_header,
               "Incomplete header read")
        .value("truncated_payload", vrtigo::utils::IOError::Kind::truncated_payload,
               "Incomplete payload read");

    // =========================================================================
    // Custom Exceptions
    // =========================================================================

    // ParseError - for packet parsing failures
    auto parse_error =
        nb::exception<std::runtime_error>(m, "ParseError", PyExc_ValueError);
    parse_error_type = parse_error.ptr();

    // VRTIOError - custom exception with diagnostic context
    // Inherits from OSError to match Python conventions for I/O errors
    auto vrt_io_error = nb::exception<std::runtime_error>(m, "VRTIOError", PyExc_OSError);
    vrt_io_error_type = vrt_io_error.ptr();

    // PayloadAlignmentError - custom exception for framing alignment issues
    nb::exception<std::runtime_error>(m, "PayloadAlignmentError", PyExc_ValueError);

    // =========================================================================
    // IngestResult - result class for SampleFramer operations
    // =========================================================================

    nb::class_<IngestResult>(m, "IngestResult",
                             "Result from SampleFramer ingest operations")
        .def(nb::init<>())
        .def_rw("frames_emitted", &IngestResult::frames_emitted,
                "Number of frames emitted during this operation")
        .def_rw("stopped", &IngestResult::stopped,
                "True if callback returned False to stop processing")
        .def("__repr__", [](const IngestResult& r) {
            std::ostringstream oss;
            oss << "IngestResult(frames_emitted=" << r.frames_emitted
                << ", stopped=" << (r.stopped ? "True" : "False") << ")";
            return oss.str();
        })
        .def("__bool__", [](const IngestResult& r) {
            // Truthy if frames were emitted and not stopped
            return r.frames_emitted > 0 || !r.stopped;
        });
}

} // namespace vrtigo_python
