#pragma once
// Reader bindings: VRTFileReader, UDPVRTReader

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <vrtigo/dynamic.hpp>
#include <vrtigo/utils/detail/reader_error.hpp>
#include <vrtigo/utils/fileio/vrt_file_reader.hpp>
#include <vrtigo/utils/netio/udp_vrt_reader.hpp>

#include "py_types.hpp"

#include <cerrno>
#include <sstream>
#include <variant>

namespace nb = nanobind;
using namespace nb::literals;

namespace vrtigo_python {

inline void bind_readers(nb::module_& m) {
    // =========================================================================
    // VRTFileReader
    // =========================================================================

    nb::class_<PyVRTFileReader>(m, "VRTFileReader",
                                "VRT file reader with automatic validation")
        .def(nb::init<const std::string&>(),
             "Open a VRT file for reading. Raises IOError if file cannot be opened.",
             "filepath"_a)
        // read_next_packet - returns owning DataPacket or ContextPacket
        .def(
            "read_next_packet",
            [](PyVRTFileReader& r, bool strict) -> nb::object {
                auto result = [&]() {
                    nb::gil_scoped_release release;
                    return r.reader.read_next_packet();
                }();

                // Handle errors
                if (!result.has_value()) {
                    const auto& err = result.error();

                    // EOF -> return None (signals iteration end)
                    if (vrtigo::utils::is_eof(err)) {
                        return nb::none();
                    }

                    // IOError -> always raise VRTIOError
                    if (vrtigo::utils::is_io_error(err)) {
                        auto& io_err = std::get<vrtigo::utils::IOError>(err);
                        PyErr_SetString(vrt_io_error_type, io_err.message());
                        throw nb::python_error();
                    }

                    // ParseError
                    if (strict) {
                        // Strict mode: raise ParseError
                        auto& parse_err = std::get<vrtigo::ParseError>(err);
                        PyErr_SetString(parse_error_type, parse_err.message());
                        throw nb::python_error();
                    } else {
                        // Non-strict: return None (caller handles skip)
                        return nb::none();
                    }
                }

                // Success - convert view to owning packet
                const auto& variant = result.value();
                if (vrtigo::is_data_packet(variant)) {
                    const auto& view = std::get<vrtigo::dynamic::DataPacketView>(variant);
                    auto bytes = view.as_bytes();
                    nb::bytes owned(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                    const auto* ptr = reinterpret_cast<const uint8_t*>(owned.c_str());
                    auto parsed = vrtigo::dynamic::DataPacketView::parse({ptr, bytes.size()});
                    return nb::cast(PyDataPacket{std::move(owned), *parsed});
                } else {
                    const auto& view = std::get<vrtigo::dynamic::ContextPacketView>(variant);
                    auto bytes = view.as_bytes();
                    nb::bytes owned(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                    const auto* ptr = reinterpret_cast<const uint8_t*>(owned.c_str());
                    auto parsed =
                        vrtigo::dynamic::ContextPacketView::parse({ptr, bytes.size()});
                    return nb::cast(PyContextPacket{std::move(owned), *parsed});
                }
            },
            "Read next packet. Returns DataPacket, ContextPacket, or None (EOF/parse error).\n\n"
            "If strict=True, raises ParseError on invalid packets instead of returning None.",
            "strict"_a = false)
        // Iterator protocol
        .def("__iter__", [](PyVRTFileReader& r) -> PyVRTFileReader& {
            return r;
        }, nb::rv_policy::reference)
        .def(
            "__next__",
            [](PyVRTFileReader& r) -> nb::object {
                while (true) {
                    auto result = [&]() {
                        nb::gil_scoped_release release;
                        return r.reader.read_next_packet();
                    }();

                    if (!result.has_value()) {
                        const auto& err = result.error();

                        // EOF -> StopIteration
                        if (vrtigo::utils::is_eof(err)) {
                            throw nb::stop_iteration();
                        }

                        // IOError -> raise VRTIOError
                        if (vrtigo::utils::is_io_error(err)) {
                            auto& io_err = std::get<vrtigo::utils::IOError>(err);
                            PyErr_SetString(vrt_io_error_type, io_err.message());
                            throw nb::python_error();
                        }

                        // ParseError -> skip, increment counter, continue
                        r.skipped_count++;
                        continue;
                    }

                    // Success - convert view to owning packet
                    const auto& variant = result.value();
                    if (vrtigo::is_data_packet(variant)) {
                        const auto& view = std::get<vrtigo::dynamic::DataPacketView>(variant);
                        auto bytes = view.as_bytes();
                        nb::bytes owned(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                        const auto* ptr = reinterpret_cast<const uint8_t*>(owned.c_str());
                        auto parsed =
                            vrtigo::dynamic::DataPacketView::parse({ptr, bytes.size()});
                        return nb::cast(PyDataPacket{std::move(owned), *parsed});
                    } else {
                        const auto& view = std::get<vrtigo::dynamic::ContextPacketView>(variant);
                        auto bytes = view.as_bytes();
                        nb::bytes owned(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                        const auto* ptr = reinterpret_cast<const uint8_t*>(owned.c_str());
                        auto parsed =
                            vrtigo::dynamic::ContextPacketView::parse({ptr, bytes.size()});
                        return nb::cast(PyContextPacket{std::move(owned), *parsed});
                    }
                }
            })
        // Methods
        .def("rewind", [](PyVRTFileReader& r) { r.reader.rewind(); },
             "Rewind file to beginning")
        .def("tell", [](PyVRTFileReader& r) { return r.reader.tell(); },
             "Get current file position in bytes")
        .def("size", [](PyVRTFileReader& r) { return r.reader.size(); },
             "Get total file size in bytes")
        // Properties
        .def_prop_ro("packets_read",
                     [](PyVRTFileReader& r) { return r.reader.packets_read(); },
                     "Number of packets read so far")
        .def_prop_ro("skipped_count", [](PyVRTFileReader& r) {
            return r.skipped_count;
        }, "Number of packets skipped due to parse errors during iteration")
        .def_prop_ro("is_open",
                     [](PyVRTFileReader& r) { return r.reader.is_open(); },
                     "True if file is still open")
        .def("__repr__", [](PyVRTFileReader& r) {
            std::ostringstream oss;
            oss << "VRTFileReader(packets_read=" << r.reader.packets_read()
                << ", position=" << r.reader.tell() << "/" << r.reader.size() << " bytes";
            if (r.skipped_count > 0) {
                oss << ", skipped=" << r.skipped_count;
            }
            oss << ")";
            return oss.str();
        });

    // =========================================================================
    // UDPVRTReader
    // =========================================================================

    nb::class_<PyUDPVRTReader>(m, "UDPVRTReader",
                               "UDP VRT packet reader with automatic validation")
        .def(nb::init<uint16_t>(),
             "Create UDP reader listening on specified port. Raises RuntimeError if binding fails.",
             "port"_a)
        // read_next_packet - returns owning DataPacket or ContextPacket
        .def(
            "read_next_packet",
            [](PyUDPVRTReader& r, bool strict) -> nb::object {
                auto result = [&]() {
                    nb::gil_scoped_release release;
                    return r.reader.read_next_packet();
                }();

                if (!result.has_value()) {
                    const auto& err = result.error();

                    // EOF (socket closed) -> return None
                    if (vrtigo::utils::is_eof(err)) {
                        return nb::none();
                    }

                    // IOError - check if it's a timeout
                    if (vrtigo::utils::is_io_error(err)) {
                        auto& io_err = std::get<vrtigo::utils::IOError>(err);
                        // EAGAIN/EWOULDBLOCK indicates timeout
                        if (io_err.errno_value == EAGAIN || io_err.errno_value == EWOULDBLOCK) {
                            PyErr_SetString(PyExc_TimeoutError, "UDP receive timeout");
                            throw nb::python_error();
                        }
                        // Other I/O errors
                        PyErr_SetString(vrt_io_error_type, io_err.message());
                        throw nb::python_error();
                    }

                    // ParseError
                    if (strict) {
                        auto& parse_err = std::get<vrtigo::ParseError>(err);
                        PyErr_SetString(parse_error_type, parse_err.message());
                        throw nb::python_error();
                    } else {
                        return nb::none();
                    }
                }

                // Success - convert view to owning packet
                const auto& variant = result.value();
                if (vrtigo::is_data_packet(variant)) {
                    const auto& view = std::get<vrtigo::dynamic::DataPacketView>(variant);
                    auto bytes = view.as_bytes();
                    nb::bytes owned(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                    const auto* ptr = reinterpret_cast<const uint8_t*>(owned.c_str());
                    auto parsed = vrtigo::dynamic::DataPacketView::parse({ptr, bytes.size()});
                    return nb::cast(PyDataPacket{std::move(owned), *parsed});
                } else {
                    const auto& view = std::get<vrtigo::dynamic::ContextPacketView>(variant);
                    auto bytes = view.as_bytes();
                    nb::bytes owned(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                    const auto* ptr = reinterpret_cast<const uint8_t*>(owned.c_str());
                    auto parsed =
                        vrtigo::dynamic::ContextPacketView::parse({ptr, bytes.size()});
                    return nb::cast(PyContextPacket{std::move(owned), *parsed});
                }
            },
            "Read next packet (blocks until data arrives or timeout).\n\n"
            "Returns DataPacket, ContextPacket, or None (socket closed/parse error).\n"
            "Raises TimeoutError on receive timeout.\n"
            "If strict=True, raises ParseError on invalid packets.",
            "strict"_a = false)
        // Iterator protocol
        .def("__iter__",
             [](PyUDPVRTReader& r) -> PyUDPVRTReader& { return r; },
             nb::rv_policy::reference)
        .def(
            "__next__",
            [](PyUDPVRTReader& r) -> nb::object {
                while (true) {
                    auto result = [&]() {
                        nb::gil_scoped_release release;
                        return r.reader.read_next_packet();
                    }();

                    if (!result.has_value()) {
                        const auto& err = result.error();

                        // EOF -> StopIteration
                        if (vrtigo::utils::is_eof(err)) {
                            throw nb::stop_iteration();
                        }

                        // IOError - check if it's a timeout (stopping condition)
                        if (vrtigo::utils::is_io_error(err)) {
                            auto& io_err = std::get<vrtigo::utils::IOError>(err);
                            if (io_err.errno_value == EAGAIN || io_err.errno_value == EWOULDBLOCK) {
                                PyErr_SetString(PyExc_TimeoutError, "UDP receive timeout");
                                throw nb::python_error();
                            }
                            PyErr_SetString(vrt_io_error_type, io_err.message());
                            throw nb::python_error();
                        }

                        // ParseError -> skip, increment counter, continue
                        r.skipped_count++;
                        continue;
                    }

                    // Success - convert view to owning packet
                    const auto& variant = result.value();
                    if (vrtigo::is_data_packet(variant)) {
                        const auto& view = std::get<vrtigo::dynamic::DataPacketView>(variant);
                        auto bytes = view.as_bytes();
                        nb::bytes owned(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                        const auto* ptr = reinterpret_cast<const uint8_t*>(owned.c_str());
                        auto parsed =
                            vrtigo::dynamic::DataPacketView::parse({ptr, bytes.size()});
                        return nb::cast(PyDataPacket{std::move(owned), *parsed});
                    } else {
                        const auto& view = std::get<vrtigo::dynamic::ContextPacketView>(variant);
                        auto bytes = view.as_bytes();
                        nb::bytes owned(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                        const auto* ptr = reinterpret_cast<const uint8_t*>(owned.c_str());
                        auto parsed =
                            vrtigo::dynamic::ContextPacketView::parse({ptr, bytes.size()});
                        return nb::cast(PyContextPacket{std::move(owned), *parsed});
                    }
                }
            })
        // Methods
        .def(
            "set_timeout",
            [](PyUDPVRTReader& r, int timeout_ms) {
                if (!r.reader.try_set_timeout(std::chrono::milliseconds(timeout_ms))) {
                    throw std::runtime_error("Failed to set socket timeout");
                }
            },
            "Set receive timeout in milliseconds (0 = infinite blocking)", "timeout_ms"_a)
        .def(
            "set_receive_buffer_size",
            [](PyUDPVRTReader& r, size_t bytes) {
                if (!r.reader.try_set_receive_buffer_size(bytes)) {
                    throw std::runtime_error("Failed to set receive buffer size");
                }
            },
            "Set socket receive buffer size in bytes", "bytes"_a)
        // Properties
        .def_prop_ro("socket_port",
                     [](PyUDPVRTReader& r) { return r.reader.socket_port(); },
                     "Port the socket is bound to")
        .def_prop_ro("socket_fd",
                     [](PyUDPVRTReader& r) { return r.reader.socket_fd(); },
                     "Underlying socket file descriptor")
        .def_prop_ro("skipped_count",
                     [](PyUDPVRTReader& r) { return r.skipped_count; },
                     "Number of packets skipped due to parse errors during iteration")
        .def_prop_ro("is_open",
                     [](PyUDPVRTReader& r) { return r.reader.is_open(); },
                     "True if socket is still valid")
        .def("__repr__", [](PyUDPVRTReader& r) {
            std::ostringstream oss;
            oss << "UDPVRTReader(port=" << r.reader.socket_port();
            if (r.skipped_count > 0) {
                oss << ", skipped=" << r.skipped_count;
            }
            oss << ")";
            return oss.str();
        });
}

} // namespace vrtigo_python
