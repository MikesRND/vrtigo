#pragma once
// Python wrapper types for VRTIGO bindings

#include <nanobind/nanobind.h>

#include <vrtigo/dynamic.hpp>
#include <vrtigo/utils/fileio/vrt_file_reader.hpp>
#include <vrtigo/utils/netio/udp_vrt_reader.hpp>

#include <span>
#include <string>

namespace nb = nanobind;

namespace vrtigo_python {

// Exception type pointers (set during module init)
extern PyObject* parse_error_type;
extern PyObject* vrt_io_error_type;

/**
 * @brief Result from SampleFramer::ingest() operations
 */
struct IngestResult {
    size_t frames_emitted{0};
    bool stopped{false};
};

/**
 * @brief Owning data packet wrapper for Python
 */
struct PyDataPacket {
    nb::bytes owned;
    vrtigo::dynamic::DataPacketView view;

    static PyDataPacket from_bytes(std::span<const uint8_t> buf);
};

/**
 * @brief Owning context packet wrapper for Python
 */
struct PyContextPacket {
    nb::bytes owned;
    vrtigo::dynamic::ContextPacketView view;

    static PyContextPacket from_bytes(std::span<const uint8_t> buf);
};

/**
 * @brief Python wrapper for VRTFileReader
 */
struct PyVRTFileReader {
    vrtigo::utils::fileio::VRTFileReader<> reader;
    size_t skipped_count{0};

    explicit PyVRTFileReader(const std::string& filepath) : reader(filepath) {}

    PyVRTFileReader(const PyVRTFileReader&) = delete;
    PyVRTFileReader& operator=(const PyVRTFileReader&) = delete;
    PyVRTFileReader(PyVRTFileReader&&) = default;
    PyVRTFileReader& operator=(PyVRTFileReader&&) = default;
};

/**
 * @brief Python wrapper for UDPVRTReader
 */
struct PyUDPVRTReader {
    vrtigo::utils::netio::UDPVRTReader<> reader;
    size_t skipped_count{0};

    explicit PyUDPVRTReader(uint16_t port) : reader(port) {}

    PyUDPVRTReader(const PyUDPVRTReader&) = delete;
    PyUDPVRTReader& operator=(const PyUDPVRTReader&) = delete;
    PyUDPVRTReader(PyUDPVRTReader&&) = default;
    PyUDPVRTReader& operator=(PyUDPVRTReader&&) = default;
};

// Factory implementations
inline PyDataPacket PyDataPacket::from_bytes(std::span<const uint8_t> buf) {
    nb::bytes owned(reinterpret_cast<const char*>(buf.data()), buf.size());
    const auto* ptr = reinterpret_cast<const uint8_t*>(owned.c_str());
    auto parsed = vrtigo::dynamic::DataPacketView::parse({ptr, buf.size()});
    if (!parsed.has_value()) {
        PyErr_SetString(parse_error_type, parsed.error().message());
        throw nb::python_error();
    }
    return PyDataPacket{std::move(owned), *parsed};
}

inline PyContextPacket PyContextPacket::from_bytes(std::span<const uint8_t> buf) {
    nb::bytes owned(reinterpret_cast<const char*>(buf.data()), buf.size());
    const auto* ptr = reinterpret_cast<const uint8_t*>(owned.c_str());
    auto parsed = vrtigo::dynamic::ContextPacketView::parse({ptr, buf.size()});
    if (!parsed.has_value()) {
        PyErr_SetString(parse_error_type, parsed.error().message());
        throw nb::python_error();
    }
    return PyContextPacket{std::move(owned), *parsed};
}

} // namespace vrtigo_python
