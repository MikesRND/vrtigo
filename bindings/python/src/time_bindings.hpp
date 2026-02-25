#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <vrtigo/duration.hpp>
#include <vrtigo/timestamp.hpp>
#include <vrtigo/types.hpp>
#include <vrtigo/utils/sample_clock.hpp>
#include <vrtigo/utils/start_time.hpp>

#include <chrono>
#include <sstream>

namespace nb = nanobind;
using namespace nb::literals;

namespace vrtigo_python {

namespace {

// ---------------------------------------------------------------------------
// Dispatch helper: construct TimestampValue from (tsi, tsf, tsi_kind, tsf_kind)
// by instantiating the correct Timestamp<TSI, TSF> and relying on implicit
// conversion to TimestampValue.
// ---------------------------------------------------------------------------
template <vrtigo::TsiType TSI>
inline vrtigo::TimestampValue make_timestamp_value_tsf(uint32_t tsi, uint64_t tsf,
                                                       vrtigo::TsfType tsf_kind) {
    switch (tsf_kind) {
        case vrtigo::TsfType::none:
            return vrtigo::Timestamp<TSI, vrtigo::TsfType::none>(tsi, tsf);
        case vrtigo::TsfType::sample_count:
            return vrtigo::Timestamp<TSI, vrtigo::TsfType::sample_count>(tsi, tsf);
        case vrtigo::TsfType::real_time:
            return vrtigo::Timestamp<TSI, vrtigo::TsfType::real_time>(tsi, tsf);
        case vrtigo::TsfType::free_running:
            return vrtigo::Timestamp<TSI, vrtigo::TsfType::free_running>(tsi, tsf);
    }
    throw nb::value_error("Invalid tsf_kind value");
}

inline vrtigo::TimestampValue make_timestamp_value(uint32_t tsi, uint64_t tsf,
                                                   vrtigo::TsiType tsi_kind,
                                                   vrtigo::TsfType tsf_kind) {
    switch (tsi_kind) {
        case vrtigo::TsiType::none:
            return make_timestamp_value_tsf<vrtigo::TsiType::none>(tsi, tsf, tsf_kind);
        case vrtigo::TsiType::utc:
            return make_timestamp_value_tsf<vrtigo::TsiType::utc>(tsi, tsf, tsf_kind);
        case vrtigo::TsiType::gps:
            return make_timestamp_value_tsf<vrtigo::TsiType::gps>(tsi, tsf, tsf_kind);
        case vrtigo::TsiType::other:
            return make_timestamp_value_tsf<vrtigo::TsiType::other>(tsi, tsf, tsf_kind);
    }
    throw nb::value_error("Invalid tsi_kind value");
}

// ---------------------------------------------------------------------------
// Arithmetic dispatch: Timestamp + Duration (requires TSF == real_time)
// ---------------------------------------------------------------------------
inline vrtigo::TimestampValue timestamp_add_duration(const vrtigo::TimestampValue& ts,
                                                     const vrtigo::Duration& d) {
    if (ts.tsf_kind() != vrtigo::TsfType::real_time) {
        throw nb::type_error("Timestamp arithmetic requires tsf_kind == real_time");
    }
    switch (ts.tsi_kind()) {
        case vrtigo::TsiType::none: {
            auto typed = *ts.as<vrtigo::TsiType::none, vrtigo::TsfType::real_time>();
            return typed + d;
        }
        case vrtigo::TsiType::utc: {
            auto typed = *ts.as<vrtigo::TsiType::utc, vrtigo::TsfType::real_time>();
            return typed + d;
        }
        case vrtigo::TsiType::gps: {
            auto typed = *ts.as<vrtigo::TsiType::gps, vrtigo::TsfType::real_time>();
            return typed + d;
        }
        case vrtigo::TsiType::other: {
            auto typed = *ts.as<vrtigo::TsiType::other, vrtigo::TsfType::real_time>();
            return typed + d;
        }
    }
    throw nb::value_error("Invalid tsi_kind value");
}

// ---------------------------------------------------------------------------
// Arithmetic dispatch: Timestamp - Duration (requires TSF == real_time)
// ---------------------------------------------------------------------------
inline vrtigo::TimestampValue timestamp_sub_duration(const vrtigo::TimestampValue& ts,
                                                     const vrtigo::Duration& d) {
    if (ts.tsf_kind() != vrtigo::TsfType::real_time) {
        throw nb::type_error("Timestamp arithmetic requires tsf_kind == real_time");
    }
    switch (ts.tsi_kind()) {
        case vrtigo::TsiType::none: {
            auto typed = *ts.as<vrtigo::TsiType::none, vrtigo::TsfType::real_time>();
            return typed - d;
        }
        case vrtigo::TsiType::utc: {
            auto typed = *ts.as<vrtigo::TsiType::utc, vrtigo::TsfType::real_time>();
            return typed - d;
        }
        case vrtigo::TsiType::gps: {
            auto typed = *ts.as<vrtigo::TsiType::gps, vrtigo::TsfType::real_time>();
            return typed - d;
        }
        case vrtigo::TsiType::other: {
            auto typed = *ts.as<vrtigo::TsiType::other, vrtigo::TsfType::real_time>();
            return typed - d;
        }
    }
    throw nb::value_error("Invalid tsi_kind value");
}

// ---------------------------------------------------------------------------
// Arithmetic dispatch: Timestamp - Timestamp → Duration
// Requires both have TSF == real_time and same TSI kind.
// ---------------------------------------------------------------------------
inline vrtigo::Duration timestamp_difference(const vrtigo::TimestampValue& lhs,
                                             const vrtigo::TimestampValue& rhs) {
    if (lhs.tsf_kind() != vrtigo::TsfType::real_time ||
        rhs.tsf_kind() != vrtigo::TsfType::real_time) {
        throw nb::type_error("Timestamp difference requires both tsf_kind == real_time");
    }
    if (lhs.tsi_kind() != rhs.tsi_kind()) {
        throw nb::type_error("Timestamp difference requires both timestamps to have the same tsi_kind");
    }
    switch (lhs.tsi_kind()) {
        case vrtigo::TsiType::none: {
            auto a = *lhs.as<vrtigo::TsiType::none, vrtigo::TsfType::real_time>();
            auto b = *rhs.as<vrtigo::TsiType::none, vrtigo::TsfType::real_time>();
            return a - b;
        }
        case vrtigo::TsiType::utc: {
            auto a = *lhs.as<vrtigo::TsiType::utc, vrtigo::TsfType::real_time>();
            auto b = *rhs.as<vrtigo::TsiType::utc, vrtigo::TsfType::real_time>();
            return a - b;
        }
        case vrtigo::TsiType::gps: {
            auto a = *lhs.as<vrtigo::TsiType::gps, vrtigo::TsfType::real_time>();
            auto b = *rhs.as<vrtigo::TsiType::gps, vrtigo::TsfType::real_time>();
            return a - b;
        }
        case vrtigo::TsiType::other: {
            auto a = *lhs.as<vrtigo::TsiType::other, vrtigo::TsfType::real_time>();
            auto b = *rhs.as<vrtigo::TsiType::other, vrtigo::TsfType::real_time>();
            return a - b;
        }
    }
    throw nb::value_error("Invalid tsi_kind value");
}

} // anonymous namespace

inline void bind_time(nb::module_& m) {

    // -----------------------------------------------------------------------
    // Duration
    // -----------------------------------------------------------------------
    nb::class_<vrtigo::Duration>(m, "Duration")
        // Static factories
        .def_static("from_picoseconds", &vrtigo::Duration::from_picoseconds,
                     "picos"_a)
        .def_static("from_nanoseconds", &vrtigo::Duration::from_nanoseconds,
                     "nanos"_a)
        .def_static("from_microseconds", &vrtigo::Duration::from_microseconds,
                     "micros"_a)
        .def_static("from_milliseconds", &vrtigo::Duration::from_milliseconds,
                     "millis"_a)
        .def_static("from_seconds", [](int64_t s) {
            return vrtigo::Duration::from_seconds(s);
        }, "seconds"_a)
        .def_static("from_seconds_float", [](double s) -> vrtigo::Duration {
            auto result = vrtigo::Duration::from_seconds(s);
            if (!result) {
                throw nb::value_error("Cannot represent the given seconds as a Duration");
            }
            return *result;
        }, "seconds"_a)
        // Named constants
        .def_static("zero", &vrtigo::Duration::zero)
        .def_static("min", &vrtigo::Duration::min)
        .def_static("max", &vrtigo::Duration::max)
        // Read-only properties
        .def_prop_ro("seconds", &vrtigo::Duration::seconds)
        .def_prop_ro("picoseconds", &vrtigo::Duration::picoseconds)
        .def_prop_ro("total_picoseconds", &vrtigo::Duration::total_picoseconds)
        .def_prop_ro("to_seconds", &vrtigo::Duration::to_seconds)
        // Predicates
        .def_prop_ro("is_zero", &vrtigo::Duration::is_zero)
        .def_prop_ro("is_negative", &vrtigo::Duration::is_negative)
        .def_prop_ro("is_positive", &vrtigo::Duration::is_positive)
        // abs
        .def("__abs__", &vrtigo::Duration::abs)
        .def("abs", &vrtigo::Duration::abs)
        // Arithmetic operators
        .def("__add__", [](const vrtigo::Duration& a, const vrtigo::Duration& b) {
            return a + b;
        }, nb::is_operator())
        .def("__sub__", [](const vrtigo::Duration& a, const vrtigo::Duration& b) {
            return a - b;
        }, nb::is_operator())
        .def("__neg__", [](const vrtigo::Duration& a) {
            return -a;
        })
        .def("__mul__", [](const vrtigo::Duration& a, int64_t b) {
            return a * b;
        }, nb::is_operator())
        .def("__rmul__", [](const vrtigo::Duration& a, int64_t b) {
            return a * b;
        }, nb::is_operator())
        .def("__floordiv__", [](const vrtigo::Duration& a, int64_t b) {
            return a / b;
        }, nb::is_operator())
        .def("__truediv__", [](const vrtigo::Duration& a, int64_t b) {
            return a / b;
        }, nb::is_operator())
        // Duration / Duration → int64_t ratio
        .def("ratio", [](const vrtigo::Duration& a, const vrtigo::Duration& b) -> int64_t {
            return a / b;
        }, "other"_a)
        // Comparison operators
        .def("__eq__", [](const vrtigo::Duration& a, const vrtigo::Duration& b) {
            return a == b;
        }, nb::is_operator())
        .def("__lt__", [](const vrtigo::Duration& a, const vrtigo::Duration& b) {
            return a < b;
        }, nb::is_operator())
        .def("__le__", [](const vrtigo::Duration& a, const vrtigo::Duration& b) {
            return a <= b;
        }, nb::is_operator())
        .def("__gt__", [](const vrtigo::Duration& a, const vrtigo::Duration& b) {
            return a > b;
        }, nb::is_operator())
        .def("__ge__", [](const vrtigo::Duration& a, const vrtigo::Duration& b) {
            return a >= b;
        }, nb::is_operator())
        // __repr__
        .def("__repr__", [](const vrtigo::Duration& d) {
            std::ostringstream oss;
            oss << "Duration(seconds=" << d.seconds()
                << ", picoseconds=" << d.picoseconds() << ")";
            return oss.str();
        });

    // -----------------------------------------------------------------------
    // SamplePeriod
    // -----------------------------------------------------------------------
    nb::class_<vrtigo::SamplePeriod>(m, "SamplePeriod")
        // Static factories
        .def_static("from_picoseconds", &vrtigo::SamplePeriod::from_picoseconds,
                     "picos"_a)
        .def_static("from_rate_hz", [](double hz) -> vrtigo::SamplePeriod {
            auto result = vrtigo::SamplePeriod::from_rate_hz(hz);
            if (!result) {
                throw nb::value_error("Cannot create SamplePeriod from the given rate");
            }
            return *result;
        }, "hz"_a)
        .def_static("from_seconds", [](double s) -> vrtigo::SamplePeriod {
            auto result = vrtigo::SamplePeriod::from_seconds(s);
            if (!result) {
                throw nb::value_error("Cannot create SamplePeriod from the given seconds value");
            }
            return *result;
        }, "seconds"_a)
        .def_static("from_ratio", [](uint64_t num, uint64_t den) -> vrtigo::SamplePeriod {
            auto result = vrtigo::SamplePeriod::from_ratio(num, den);
            if (!result) {
                throw nb::value_error("Cannot create SamplePeriod from the given ratio");
            }
            return *result;
        }, "numerator"_a, "denominator"_a)
        // Read-only properties
        .def_prop_ro("picoseconds", &vrtigo::SamplePeriod::picoseconds)
        .def_prop_ro("rate_hz", &vrtigo::SamplePeriod::rate_hz)
        .def_prop_ro("seconds", &vrtigo::SamplePeriod::seconds)
        .def_prop_ro("is_exact", &vrtigo::SamplePeriod::is_exact)
        .def_prop_ro("error_picoseconds", &vrtigo::SamplePeriod::error_picoseconds)
        .def_prop_ro("error_ppm", &vrtigo::SamplePeriod::error_ppm)
        // Conversion
        .def("to_duration", &vrtigo::SamplePeriod::to_duration)
        // Comparison operators
        .def("__eq__", [](const vrtigo::SamplePeriod& a, const vrtigo::SamplePeriod& b) {
            return a == b;
        }, nb::is_operator())
        .def("__lt__", [](const vrtigo::SamplePeriod& a, const vrtigo::SamplePeriod& b) {
            return a < b;
        }, nb::is_operator())
        .def("__le__", [](const vrtigo::SamplePeriod& a, const vrtigo::SamplePeriod& b) {
            return a <= b;
        }, nb::is_operator())
        .def("__gt__", [](const vrtigo::SamplePeriod& a, const vrtigo::SamplePeriod& b) {
            return a > b;
        }, nb::is_operator())
        .def("__ge__", [](const vrtigo::SamplePeriod& a, const vrtigo::SamplePeriod& b) {
            return a >= b;
        }, nb::is_operator())
        // __repr__
        .def("__repr__", [](const vrtigo::SamplePeriod& sp) {
            std::ostringstream oss;
            oss << "SamplePeriod(picoseconds=" << sp.picoseconds()
                << ", rate_hz=" << sp.rate_hz()
                << ", exact=" << (sp.is_exact() ? "True" : "False") << ")";
            return oss.str();
        });

    // -----------------------------------------------------------------------
    // Timestamp (wrapping TimestampValue)
    // -----------------------------------------------------------------------
    nb::class_<vrtigo::TimestampValue>(m, "Timestamp")
        // Constructor via dispatch helper
        .def("__init__", [](vrtigo::TimestampValue* self, uint32_t tsi, uint64_t tsf,
                            vrtigo::TsiType tsi_kind, vrtigo::TsfType tsf_kind) {
            new (self) vrtigo::TimestampValue(make_timestamp_value(tsi, tsf, tsi_kind, tsf_kind));
        }, "tsi"_a, "tsf"_a, "tsi_kind"_a, "tsf_kind"_a)
        // Read-only properties
        .def_prop_ro("tsi", &vrtigo::TimestampValue::tsi)
        .def_prop_ro("tsf", &vrtigo::TimestampValue::tsf)
        .def_prop_ro("tsi_kind", &vrtigo::TimestampValue::tsi_kind)
        .def_prop_ro("tsf_kind", &vrtigo::TimestampValue::tsf_kind)
        .def_prop_ro("has_tsi", &vrtigo::TimestampValue::has_tsi)
        .def_prop_ro("has_tsf", &vrtigo::TimestampValue::has_tsf)
        // Equality
        .def("__eq__", [](const vrtigo::TimestampValue& a, const vrtigo::TimestampValue& b) {
            return a == b;
        }, nb::is_operator())
        // Timestamp + Duration
        .def("__add__", [](const vrtigo::TimestampValue& self, const vrtigo::Duration& d) {
            return timestamp_add_duration(self, d);
        }, nb::is_operator())
        // Timestamp - Duration or Timestamp - Timestamp
        .def("__sub__", [](const vrtigo::TimestampValue& self, nb::object other) -> nb::object {
            if (nb::isinstance<vrtigo::Duration>(other)) {
                auto d = nb::cast<vrtigo::Duration>(other);
                return nb::cast(timestamp_sub_duration(self, d));
            }
            if (nb::isinstance<vrtigo::TimestampValue>(other)) {
                auto rhs = nb::cast<vrtigo::TimestampValue>(other);
                return nb::cast(timestamp_difference(self, rhs));
            }
            return nb::not_implemented();
        }, nb::is_operator())
        // Static: now() — returns UTC real_time timestamp
        .def_static("now", []() -> vrtigo::TimestampValue {
            return vrtigo::UtcRealTimestamp::now();
        })
        // Static: from_utc_seconds
        .def_static("from_utc_seconds", [](uint32_t sec) -> vrtigo::TimestampValue {
            return vrtigo::UtcRealTimestamp::from_utc_seconds(sec);
        }, "seconds"_a)
        // to_datetime — only for UTC + real_time
        .def("to_datetime", [](const vrtigo::TimestampValue& ts) {
            if (ts.tsi_kind() != vrtigo::TsiType::utc ||
                ts.tsf_kind() != vrtigo::TsfType::real_time) {
                throw nb::type_error("to_datetime() requires UTC real_time timestamp");
            }
            auto typed = *ts.as<vrtigo::TsiType::utc, vrtigo::TsfType::real_time>();
            auto tp = typed.to_chrono();
            auto duration = tp.time_since_epoch();
            auto secs = std::chrono::duration_cast<std::chrono::seconds>(duration);
            auto micros = std::chrono::duration_cast<std::chrono::microseconds>(duration - secs);

            nb::object datetime_mod = nb::module_::import_("datetime");
            nb::object datetime_cls = datetime_mod.attr("datetime");
            nb::object tz_cls = datetime_mod.attr("timezone");
            nb::object utc = tz_cls.attr("utc");

            std::time_t t = static_cast<std::time_t>(secs.count());
            std::tm* gm = std::gmtime(&t);
            return datetime_cls(
                gm->tm_year + 1900, gm->tm_mon + 1, gm->tm_mday,
                gm->tm_hour, gm->tm_min, gm->tm_sec,
                static_cast<int>(micros.count()), utc
            );
        })
        // __repr__
        .def("__repr__", [](const vrtigo::TimestampValue& ts) {
            std::ostringstream oss;
            oss << "Timestamp(tsi=" << ts.tsi()
                << ", tsf=" << ts.tsf()
                << ", tsi_kind=" << vrtigo::tsi_type_string(ts.tsi_kind())
                << ", tsf_kind=" << vrtigo::tsf_type_string(ts.tsf_kind()) << ")";
            return oss.str();
        });

    // -----------------------------------------------------------------------
    // StartTime::Base enum
    // -----------------------------------------------------------------------
    nb::enum_<vrtigo::utils::StartTime::Base>(m, "StartTimeBase", "StartTime base reference type")
        .value("now", vrtigo::utils::StartTime::Base::now, "Capture wall clock at resolve()")
        .value("next_second", vrtigo::utils::StartTime::Base::next_second, "Next second boundary")
        .value("absolute", vrtigo::utils::StartTime::Base::absolute, "Explicit timestamp")
        .value("zero", vrtigo::utils::StartTime::Base::zero, "Epoch (0, 0)");

    // -----------------------------------------------------------------------
    // StartTime
    // -----------------------------------------------------------------------
    nb::class_<vrtigo::utils::StartTime>(m, "StartTime")
        .def_static("now", &vrtigo::utils::StartTime::now)
        .def_static("now_plus", &vrtigo::utils::StartTime::now_plus, "offset"_a)
        .def_static("absolute", [](const vrtigo::TimestampValue& ts) {
            if (ts.tsi_kind() != vrtigo::TsiType::utc ||
                ts.tsf_kind() != vrtigo::TsfType::real_time) {
                throw nb::type_error("StartTime.absolute() requires a UTC real_time timestamp");
            }
            auto typed = *ts.as<vrtigo::TsiType::utc, vrtigo::TsfType::real_time>();
            return vrtigo::utils::StartTime::absolute(typed);
        }, "timestamp"_a)
        .def_static("zero", &vrtigo::utils::StartTime::zero)
        .def_static("at_next_second", &vrtigo::utils::StartTime::at_next_second)
        .def_static("at_next_second_plus", &vrtigo::utils::StartTime::at_next_second_plus,
                     "offset"_a)
        .def("resolve", [](const vrtigo::utils::StartTime& st) -> vrtigo::TimestampValue {
            return st.resolve();
        })
        .def_prop_ro("base", &vrtigo::utils::StartTime::base)
        .def_prop_ro("offset", &vrtigo::utils::StartTime::offset)
        .def("__repr__", [](const vrtigo::utils::StartTime& st) {
            std::ostringstream oss;
            oss << "StartTime(base=" << static_cast<int>(st.base())
                << ", offset=" << st.offset().to_seconds() << "s)";
            return oss.str();
        });

    // -----------------------------------------------------------------------
    // SampleClock (UTC real_time specialization)
    // -----------------------------------------------------------------------
    using PySampleClock = vrtigo::utils::SampleClock<vrtigo::TsiType::utc, vrtigo::TsfType::real_time>;

    nb::class_<PySampleClock>(m, "SampleClock")
        .def(nb::init<double, vrtigo::utils::StartTime>(),
             "sample_period_seconds"_a, "start"_a = vrtigo::utils::StartTime::zero())
        .def("now", [](PySampleClock& self) -> vrtigo::TimestampValue {
            return self.now();
        })
        .def("tick", [](PySampleClock& self, int64_t samples) -> vrtigo::TimestampValue {
            if (samples < 0) {
                throw nb::value_error("sample count must be non-negative");
            }
            return self.tick(static_cast<uint64_t>(samples));
        }, "samples"_a = 1)
        .def("advance", [](PySampleClock& self, int64_t samples) {
            if (samples < 0) {
                throw nb::value_error("sample count must be non-negative");
            }
            self.advance(static_cast<uint64_t>(samples));
        }, "samples"_a)
        .def("reset", &PySampleClock::reset)
        .def_prop_ro("period", &PySampleClock::period)
        .def_prop_ro("elapsed_samples", &PySampleClock::elapsed_samples)
        .def("__repr__", [](const PySampleClock& sc) {
            std::ostringstream oss;
            oss << "SampleClock(period=" << sc.period().picoseconds()
                << "ps, elapsed_samples=" << sc.elapsed_samples() << ")";
            return oss.str();
        });
}

} // namespace vrtigo_python
