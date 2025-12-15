#pragma once
// SampleFramer bindings with numpy support

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>

#include <vrtigo/dynamic.hpp>
#include <vrtigo/utils/sample_framer.hpp>

#include "py_types.hpp"

#include <complex>
#include <sstream>
#include <variant>

namespace nb = nanobind;
using namespace nb::literals;

namespace vrtigo_python {

/**
 * Python wrapper for SampleFramer with runtime dtype dispatch.
 *
 * Uses std::variant to hold the appropriate SimpleSampleFramer<T> instantiation
 * based on the numpy array's dtype at construction time.
 */
class PySampleFramer {
public:
    // All supported framer types via variant
    using FramerVariant = std::variant<
        vrtigo::utils::SimpleSampleFramer<int8_t>,
        vrtigo::utils::SimpleSampleFramer<int16_t>,
        vrtigo::utils::SimpleSampleFramer<int32_t>,
        vrtigo::utils::SimpleSampleFramer<float>,
        vrtigo::utils::SimpleSampleFramer<double>,
        vrtigo::utils::SimpleSampleFramer<std::complex<float>>,
        vrtigo::utils::SimpleSampleFramer<std::complex<double>>>;

    // Numpy array type we accept (contiguous, any dtype)
    using NumpyArray = nb::ndarray<nb::numpy, nb::c_contig>;

    /**
     * Construct framer from numpy buffer.
     *
     * @param buffer Numpy array for frame accumulation (determines dtype)
     * @param samples_per_frame Number of samples per emitted frame
     * @param callback Python callable: fn(frame: ndarray) -> bool
     * @param copy_frames If true, callback receives copies; if false, views
     */
    PySampleFramer(NumpyArray buffer, size_t samples_per_frame, nb::callable callback)
        : py_callback_(std::move(callback)),
          samples_per_frame_(samples_per_frame) {

        // Store buffer as Python object to prevent GC
        buffer_obj_ = nb::cast(buffer);

        // Dispatch based on dtype
        auto dtype = buffer.dtype();

        if (dtype == nb::dtype<int8_t>()) {
            init_framer<int8_t>(buffer, samples_per_frame);
        } else if (dtype == nb::dtype<int16_t>()) {
            init_framer<int16_t>(buffer, samples_per_frame);
        } else if (dtype == nb::dtype<int32_t>()) {
            init_framer<int32_t>(buffer, samples_per_frame);
        } else if (dtype == nb::dtype<float>()) {
            init_framer<float>(buffer, samples_per_frame);
        } else if (dtype == nb::dtype<double>()) {
            init_framer<double>(buffer, samples_per_frame);
        } else if (dtype == nb::dtype<std::complex<float>>()) {
            init_framer<std::complex<float>>(buffer, samples_per_frame);
        } else if (dtype == nb::dtype<std::complex<double>>()) {
            init_framer<std::complex<double>>(buffer, samples_per_frame);
        } else {
            throw std::invalid_argument(
                "Unsupported dtype. Supported: int8, int16, int32, float32, float64, "
                "complex64, complex128");
        }
    }

    /**
     * Ingest samples from a DataPacket.
     */
    IngestResult ingest(const PyDataPacket& packet) {
        return ingest_payload_bytes(packet.view.payload());
    }

    /**
     * Ingest samples from a DataPacketView.
     */
    IngestResult ingest_view(const vrtigo::dynamic::DataPacketView& view) {
        return ingest_payload_bytes(view.payload());
    }

    /**
     * Ingest raw payload bytes.
     */
    IngestResult ingest_payload(nb::bytes payload) {
        const auto* ptr = reinterpret_cast<const uint8_t*>(payload.c_str());
        size_t size = payload.size();
        return ingest_payload_bytes({ptr, size});
    }

    /**
     * Flush any partial frame.
     */
    IngestResult flush_partial() {
        return std::visit(
            [](auto& framer) -> IngestResult {
                auto result = framer.flush_partial();
                if (!result.has_value()) {
                    return IngestResult{0, result.error() == vrtigo::utils::FrameError::stop_requested};
                }
                return IngestResult{*result, false};
            },
            *framer_);
    }

    /**
     * Reset framer state.
     */
    void reset() {
        std::visit([](auto& framer) { framer.reset(); }, *framer_);
    }

    // Accessors
    size_t samples_per_frame() const noexcept { return samples_per_frame_; }

    size_t emitted_frames() const noexcept {
        return std::visit([](const auto& framer) { return framer.emitted_frames(); }, *framer_);
    }

    size_t buffered_samples() const noexcept {
        return std::visit([](const auto& framer) { return framer.buffered_samples(); }, *framer_);
    }

    bool is_ping_pong() const noexcept {
        return std::visit([](const auto& framer) { return framer.is_ping_pong(); }, *framer_);
    }

private:
    nb::object buffer_obj_;        // Keep buffer alive
    nb::callable py_callback_;     // Python callback
    size_t samples_per_frame_;     // Cached for accessor
    std::optional<FramerVariant> framer_;  // Initialized by init_framer()

    template <typename SampleT>
    void init_framer(NumpyArray& buffer, size_t samples_per_frame) {
        // Get raw pointer to buffer data
        auto* data = static_cast<SampleT*>(buffer.data());
        // For 1D arrays, shape(0) gives element count
        size_t count = 1;
        for (size_t i = 0; i < buffer.ndim(); ++i) {
            count *= buffer.shape(i);
        }
        std::span<SampleT> buf_span(data, count);

        // Create callback that bridges to Python
        auto cpp_callback = [this](std::span<const SampleT> frame) -> bool {
            nb::gil_scoped_acquire gil;

            // Create numpy array view for the frame
            // Note: Users should call .copy() if they need to retain the data
            size_t shape[1] = {frame.size()};
            auto arr = nb::ndarray<nb::numpy, const SampleT, nb::shape<-1>>(
                frame.data(), 1, shape, buffer_obj_);
            nb::object py_frame = nb::cast(arr);

            // Call Python callback
            nb::object result = py_callback_(py_frame);

            // Interpret result as bool (None or missing return = True)
            if (result.is_none()) {
                return true;
            }
            return nb::cast<bool>(result);
        };

        // Construct the framer and assign to optional
        framer_ = FramerVariant{
            std::in_place_type<vrtigo::utils::SimpleSampleFramer<SampleT>>,
            buf_span, samples_per_frame, std::move(cpp_callback)};
    }

    IngestResult ingest_payload_bytes(std::span<const uint8_t> payload) {
        return std::visit(
            [&payload](auto& framer) -> IngestResult {
                auto result = framer.ingest_payload(payload);
                if (!result.has_value()) {
                    if (result.error() == vrtigo::utils::FrameError::stop_requested) {
                        return IngestResult{0, true};
                    }
                    // payload_not_aligned - throw
                    throw std::invalid_argument("Payload size not aligned to sample size");
                }
                return IngestResult{*result, false};
            },
            *framer_);
    }
};

inline void bind_sample_framer(nb::module_& m) {
    nb::class_<PySampleFramer>(
        m, "SampleFramer",
        "Accumulates VRT payload samples into fixed-size frames.\n\n"
        "The framer reads samples from packet payloads, performs endian conversion,\n"
        "and calls your callback when a complete frame is ready.\n\n"
        "Example::\n\n"
        "    import numpy as np\n"
        "    import vrtigo\n\n"
        "    buffer = np.zeros(1024, dtype=np.int16)\n\n"
        "    def on_frame(frame):\n"
        "        # frame is a read-only view into the buffer\n"
        "        # Call frame.copy() if you need to retain the data\n"
        "        print(f'Got frame with {len(frame)} samples')\n"
        "        return True  # Continue processing\n\n"
        "    framer = vrtigo.SampleFramer(buffer, samples_per_frame=512, callback=on_frame)\n\n"
        "    for packet in vrtigo.VRTFileReader('data.vrt'):\n"
        "        if isinstance(packet, vrtigo.DataPacket):\n"
        "            result = framer.ingest(packet)\n"
        "            if result.stopped:\n"
        "                break\n\n"
        "    framer.flush_partial()  # Emit any remaining samples")
        .def(
            nb::init<PySampleFramer::NumpyArray, size_t, nb::callable>(),
            "Create a sample framer.\n\n"
            "Args:\n"
            "    buffer: Numpy array for frame accumulation. Dtype determines sample type.\n"
            "            Supported: int8, int16, int32, float32, float64, complex64, complex128\n"
            "    samples_per_frame: Number of samples per emitted frame\n"
            "    callback: Function called with each complete frame: fn(frame) -> bool\n"
            "              Return False to stop processing, True to continue.\n"
            "              Frame is a view - call frame.copy() to retain data.",
            "buffer"_a, "samples_per_frame"_a, "callback"_a)
        .def("ingest", &PySampleFramer::ingest,
             "Ingest samples from a DataPacket.\n\n"
             "Returns IngestResult with frames_emitted and stopped flag.",
             "packet"_a)
        .def("ingest_view", &PySampleFramer::ingest_view,
             "Ingest samples from a DataPacketView.\n\n"
             "Returns IngestResult with frames_emitted and stopped flag.",
             "view"_a)
        .def("ingest_payload", &PySampleFramer::ingest_payload,
             "Ingest raw payload bytes.\n\n"
             "Raises ValueError if payload size is not aligned to sample size.",
             "payload"_a)
        .def("flush_partial", &PySampleFramer::flush_partial,
             "Flush any partial frame (emits shorter frame if samples are buffered).\n\n"
             "Returns IngestResult.")
        .def("reset", &PySampleFramer::reset,
             "Reset framer state, clearing accumulated samples and frame count.")
        .def_prop_ro("samples_per_frame", &PySampleFramer::samples_per_frame,
                     "Number of samples per complete frame")
        .def_prop_ro("emitted_frames", &PySampleFramer::emitted_frames,
                     "Total frames emitted since construction or last reset")
        .def_prop_ro("buffered_samples", &PySampleFramer::buffered_samples,
                     "Number of samples currently buffered (waiting for more data)")
        .def_prop_ro("is_ping_pong", &PySampleFramer::is_ping_pong,
                     "True if using ping-pong (double buffer) mode")
        .def("__repr__", [](const PySampleFramer& f) {
            std::ostringstream oss;
            oss << "SampleFramer(samples_per_frame=" << f.samples_per_frame()
                << ", emitted=" << f.emitted_frames()
                << ", buffered=" << f.buffered_samples() << ")";
            return oss.str();
        });
}

} // namespace vrtigo_python
