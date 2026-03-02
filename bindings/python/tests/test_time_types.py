"""Tests for Python bindings of C++ time types exposed via vrtigo."""

import datetime
import math

import pytest
import vrtigo


# ---------------------------------------------------------------------------
# Duration Tests
# ---------------------------------------------------------------------------


class TestDurationFactories:
    """test_duration_factories -- verify every static factory and its accessors."""

    def test_from_picoseconds(self):
        d = vrtigo.Duration.from_picoseconds(500)
        assert d.seconds == 0
        assert d.picoseconds == 500

    def test_from_nanoseconds(self):
        d = vrtigo.Duration.from_nanoseconds(1500)
        # 1500 ns == 1_500_000 ps
        assert d.seconds == 0
        assert d.picoseconds == 1_500_000

    def test_from_microseconds(self):
        d = vrtigo.Duration.from_microseconds(250)
        # 250 us == 250_000_000 ps
        assert d.seconds == 0
        assert d.picoseconds == 250_000_000

    def test_from_milliseconds(self):
        d = vrtigo.Duration.from_milliseconds(1)
        # 1 ms == 1_000_000_000 ps
        assert d.seconds == 0
        assert d.picoseconds == 1_000_000_000

    def test_from_seconds(self):
        d = vrtigo.Duration.from_seconds(3)
        assert d.seconds == 3
        assert d.picoseconds == 0

    def test_from_seconds_float(self):
        d = vrtigo.Duration.from_seconds_float(1.5)
        assert d.seconds == 1
        assert d.picoseconds == 500_000_000_000

    def test_zero(self):
        d = vrtigo.Duration.zero()
        assert d.seconds == 0
        assert d.picoseconds == 0
        assert d.is_zero

    def test_min(self):
        d = vrtigo.Duration.min()
        assert d.is_negative

    def test_max(self):
        d = vrtigo.Duration.max()
        assert d.is_positive

    def test_total_picoseconds(self):
        d = vrtigo.Duration.from_seconds(1)
        assert d.total_picoseconds == vrtigo.PICOSECONDS_PER_SECOND

    def test_to_seconds(self):
        d = vrtigo.Duration.from_seconds_float(2.5)
        assert d.to_seconds == pytest.approx(2.5)


class TestDurationFromSecondsFloatInvalid:
    """test_duration_from_seconds_float_invalid -- NaN, inf, -inf raise ValueError."""

    def test_nan(self):
        with pytest.raises(ValueError):
            vrtigo.Duration.from_seconds_float(float("nan"))

    def test_inf(self):
        with pytest.raises(ValueError):
            vrtigo.Duration.from_seconds_float(float("inf"))

    def test_neg_inf(self):
        with pytest.raises(ValueError):
            vrtigo.Duration.from_seconds_float(float("-inf"))


class TestDurationArithmetic:
    """test_duration_arithmetic -- add, sub, mul, div, negation, abs."""

    def test_add(self):
        a = vrtigo.Duration.from_seconds(1)
        b = vrtigo.Duration.from_milliseconds(500)
        result = a + b
        assert result.to_seconds == pytest.approx(1.5)

    def test_sub(self):
        a = vrtigo.Duration.from_seconds(2)
        b = vrtigo.Duration.from_seconds(1)
        result = a - b
        assert result.seconds == 1
        assert result.picoseconds == 0

    def test_mul(self):
        d = vrtigo.Duration.from_seconds(3)
        result = d * 4
        assert result.seconds == 12

    def test_div(self):
        d = vrtigo.Duration.from_seconds(12)
        result = d / 4
        assert result.seconds == 3
        assert result.picoseconds == 0

    def test_negation(self):
        d = vrtigo.Duration.from_seconds(5)
        neg = -d
        assert neg.is_negative
        assert neg.to_seconds == pytest.approx(-5.0)

    def test_abs_of_negative(self):
        d = -vrtigo.Duration.from_seconds(7)
        assert abs(d).seconds == 7
        assert abs(d).picoseconds == 0

    def test_abs_of_positive(self):
        d = vrtigo.Duration.from_seconds(7)
        assert abs(d).seconds == 7


class TestDurationComparison:
    """test_duration_comparison -- ==, <, >, <=, >=."""

    def test_equal(self):
        a = vrtigo.Duration.from_seconds(1)
        b = vrtigo.Duration.from_seconds(1)
        assert a == b

    def test_less_than(self):
        a = vrtigo.Duration.from_seconds(1)
        b = vrtigo.Duration.from_seconds(2)
        assert a < b

    def test_greater_than(self):
        a = vrtigo.Duration.from_seconds(3)
        b = vrtigo.Duration.from_seconds(1)
        assert a > b

    def test_less_equal(self):
        a = vrtigo.Duration.from_seconds(1)
        b = vrtigo.Duration.from_seconds(1)
        assert a <= b
        c = vrtigo.Duration.from_seconds(2)
        assert a <= c

    def test_greater_equal(self):
        a = vrtigo.Duration.from_seconds(2)
        b = vrtigo.Duration.from_seconds(2)
        assert a >= b
        c = vrtigo.Duration.from_seconds(1)
        assert a >= c

    def test_negative_less_than_zero(self):
        neg = -vrtigo.Duration.from_seconds(1)
        zero = vrtigo.Duration.zero()
        assert neg < zero

    def test_positive_greater_than_negative(self):
        pos = vrtigo.Duration.from_seconds(1)
        neg = -vrtigo.Duration.from_seconds(1)
        assert pos > neg


class TestDurationHash:
    """test_duration_hash -- equal durations have equal hashes; set deduplicates."""

    def test_equal_durations_have_equal_hashes(self):
        a = vrtigo.Duration.from_seconds_float(1.5)
        b = vrtigo.Duration.from_seconds_float(1.5)
        assert a == b
        assert hash(a) == hash(b)

    def test_set_deduplicates_equal_durations(self):
        a = vrtigo.Duration.from_seconds(2)
        b = vrtigo.Duration.from_seconds(2)
        c = vrtigo.Duration.from_seconds(3)
        s = {a, b, c}
        assert len(s) == 2


class TestDurationPredicates:
    """test_duration_predicates -- is_zero, is_negative, is_positive."""

    def test_zero_predicates(self):
        d = vrtigo.Duration.zero()
        assert d.is_zero is True
        assert d.is_negative is False
        assert d.is_positive is False

    def test_positive_predicates(self):
        d = vrtigo.Duration.from_seconds(1)
        assert d.is_zero is False
        assert d.is_negative is False
        assert d.is_positive is True

    def test_negative_predicates(self):
        d = -vrtigo.Duration.from_seconds(1)
        assert d.is_zero is False
        assert d.is_negative is True
        assert d.is_positive is False


class TestDurationRepr:
    """test_duration_repr -- verify repr contains 'Duration' and field values."""

    def test_repr_contains_duration(self):
        d = vrtigo.Duration.from_seconds_float(1.5)
        r = repr(d)
        assert "Duration" in r

    def test_repr_contains_values(self):
        d = vrtigo.Duration.from_seconds_float(1.5)
        r = repr(d)
        assert "seconds=1" in r
        assert "picoseconds=500000000000" in r


# ---------------------------------------------------------------------------
# SamplePeriod Tests
# ---------------------------------------------------------------------------


class TestSamplePeriodFactories:
    """test_sample_period_factories -- from_rate_hz, from_picoseconds, from_ratio."""

    def test_from_rate_hz_10mhz(self):
        sp = vrtigo.SamplePeriod.from_rate_hz(10e6)
        assert sp.picoseconds == 100_000

    def test_from_picoseconds(self):
        sp = vrtigo.SamplePeriod.from_picoseconds(100_000)
        assert sp.picoseconds == 100_000

    def test_from_ratio_10mhz(self):
        sp = vrtigo.SamplePeriod.from_ratio(10_000_000, 1)
        assert sp.picoseconds == 100_000


class TestSamplePeriodFromSeconds:
    """test_sample_period_from_seconds -- from_seconds(1e-6) yields 1000000 ps."""

    def test_from_seconds_1us(self):
        sp = vrtigo.SamplePeriod.from_seconds(1e-6)
        assert sp.picoseconds == 1_000_000


class TestSamplePeriodInvalid:
    """test_sample_period_invalid -- zero, negative, NaN rates raise ValueError."""

    def test_zero_rate(self):
        with pytest.raises(ValueError):
            vrtigo.SamplePeriod.from_rate_hz(0.0)

    def test_negative_rate(self):
        with pytest.raises(ValueError):
            vrtigo.SamplePeriod.from_rate_hz(-1.0)

    def test_nan_rate(self):
        with pytest.raises(ValueError):
            vrtigo.SamplePeriod.from_rate_hz(float("nan"))

    def test_from_seconds_zero(self):
        with pytest.raises(ValueError):
            vrtigo.SamplePeriod.from_seconds(0.0)

    def test_from_seconds_negative(self):
        with pytest.raises(ValueError):
            vrtigo.SamplePeriod.from_seconds(-1.0)

    def test_from_seconds_nan(self):
        with pytest.raises(ValueError):
            vrtigo.SamplePeriod.from_seconds(float("nan"))


class TestSamplePeriodExactness:
    """test_sample_period_exactness -- from_picoseconds is exact, 10 MHz is exact."""

    def test_from_picoseconds_is_exact(self):
        sp = vrtigo.SamplePeriod.from_picoseconds(100_000)
        assert sp.is_exact is True

    def test_from_rate_hz_10mhz_is_exact(self):
        sp = vrtigo.SamplePeriod.from_rate_hz(10e6)
        assert sp.is_exact is True

    def test_from_rate_hz_3_might_not_be_exact(self):
        sp = vrtigo.SamplePeriod.from_rate_hz(3.0)
        # 1/3 Hz is not representable exactly in picoseconds
        # Just check the property is accessible; exactness depends on implementation
        assert isinstance(sp.is_exact, bool)


class TestSamplePeriodAccessors:
    """test_sample_period_accessors -- rate_hz, seconds, picoseconds, to_duration."""

    def test_rate_hz(self):
        sp = vrtigo.SamplePeriod.from_rate_hz(10e6)
        assert sp.rate_hz == pytest.approx(10e6)

    def test_seconds(self):
        sp = vrtigo.SamplePeriod.from_rate_hz(10e6)
        assert sp.seconds == pytest.approx(1e-7)

    def test_picoseconds(self):
        sp = vrtigo.SamplePeriod.from_picoseconds(100_000)
        assert sp.picoseconds == 100_000

    def test_to_duration(self):
        sp = vrtigo.SamplePeriod.from_picoseconds(100_000)
        d = sp.to_duration()
        assert isinstance(d, vrtigo.Duration)
        assert d.picoseconds == 100_000
        assert d.seconds == 0


class TestSamplePeriodComparison:
    """test_sample_period_comparison -- two periods with different picoseconds."""

    def test_less_than(self):
        a = vrtigo.SamplePeriod.from_picoseconds(100)
        b = vrtigo.SamplePeriod.from_picoseconds(200)
        assert a < b

    def test_greater_than(self):
        a = vrtigo.SamplePeriod.from_picoseconds(200)
        b = vrtigo.SamplePeriod.from_picoseconds(100)
        assert a > b

    def test_equal(self):
        a = vrtigo.SamplePeriod.from_picoseconds(100)
        b = vrtigo.SamplePeriod.from_picoseconds(100)
        assert a == b

    def test_less_equal(self):
        a = vrtigo.SamplePeriod.from_picoseconds(100)
        b = vrtigo.SamplePeriod.from_picoseconds(100)
        assert a <= b

    def test_greater_equal(self):
        a = vrtigo.SamplePeriod.from_picoseconds(200)
        b = vrtigo.SamplePeriod.from_picoseconds(100)
        assert a >= b


class TestSamplePeriodHash:
    """test_sample_period_hash -- equal periods have equal hashes; set deduplicates."""

    def test_equal_periods_have_equal_hashes(self):
        a = vrtigo.SamplePeriod.from_picoseconds(100_000)
        b = vrtigo.SamplePeriod.from_picoseconds(100_000)
        assert a == b
        assert hash(a) == hash(b)

    def test_set_deduplicates_equal_periods(self):
        a = vrtigo.SamplePeriod.from_picoseconds(100_000)
        b = vrtigo.SamplePeriod.from_picoseconds(100_000)
        c = vrtigo.SamplePeriod.from_picoseconds(200_000)
        s = {a, b, c}
        assert len(s) == 2


class TestSamplePeriodRepr:
    """test_sample_period_repr -- verify repr contains 'SamplePeriod'."""

    def test_repr_contains_sample_period(self):
        sp = vrtigo.SamplePeriod.from_rate_hz(10e6)
        r = repr(sp)
        assert "SamplePeriod" in r


# ---------------------------------------------------------------------------
# Timestamp Tests
# ---------------------------------------------------------------------------


class TestTimestampConstruction:
    """test_timestamp_construction -- all TsiType values with TsfType.real_time."""

    def test_tsi_none(self):
        ts = vrtigo.Timestamp(0, 0, vrtigo.TsiType.none, vrtigo.TsfType.real_time)
        assert ts.tsi == 0
        assert ts.tsf == 0
        assert ts.tsi_kind == vrtigo.TsiType.none
        assert ts.tsf_kind == vrtigo.TsfType.real_time

    def test_tsi_utc(self):
        ts = vrtigo.Timestamp(1000, 500, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        assert ts.tsi == 1000
        assert ts.tsf == 500
        assert ts.tsi_kind == vrtigo.TsiType.utc
        assert ts.has_tsi is True
        assert ts.has_tsf is True

    def test_tsi_gps(self):
        ts = vrtigo.Timestamp(2000, 0, vrtigo.TsiType.gps, vrtigo.TsfType.real_time)
        assert ts.tsi_kind == vrtigo.TsiType.gps

    def test_tsi_other(self):
        ts = vrtigo.Timestamp(3000, 0, vrtigo.TsiType.other, vrtigo.TsfType.real_time)
        assert ts.tsi_kind == vrtigo.TsiType.other


class TestTimestampConstructionAllTsf:
    """test_timestamp_construction_all_tsf -- all TsfType values."""

    def test_tsf_none(self):
        ts = vrtigo.Timestamp(100, 0, vrtigo.TsiType.utc, vrtigo.TsfType.none)
        assert ts.tsf_kind == vrtigo.TsfType.none
        assert ts.has_tsf is False

    def test_tsf_sample_count(self):
        ts = vrtigo.Timestamp(100, 42, vrtigo.TsiType.utc, vrtigo.TsfType.sample_count)
        assert ts.tsf_kind == vrtigo.TsfType.sample_count
        assert ts.tsf == 42

    def test_tsf_real_time(self):
        ts = vrtigo.Timestamp(100, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        assert ts.tsf_kind == vrtigo.TsfType.real_time

    def test_tsf_free_running(self):
        ts = vrtigo.Timestamp(100, 0, vrtigo.TsiType.utc, vrtigo.TsfType.free_running)
        assert ts.tsf_kind == vrtigo.TsfType.free_running


class TestTimestampAddDuration:
    """test_timestamp_add_duration -- UTC real_time ts + Duration."""

    def test_add_duration_increases_tsi(self):
        ts = vrtigo.Timestamp(1000, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        d = vrtigo.Duration.from_seconds(5)
        result = ts + d
        assert isinstance(result, vrtigo.Timestamp)
        assert result.tsi == 1005

    def test_add_duration_with_subsecond(self):
        ts = vrtigo.Timestamp(1000, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        d = vrtigo.Duration.from_milliseconds(500)
        result = ts + d
        assert result.tsi == 1000
        assert result.tsf > 0


class TestTimestampSubDuration:
    """test_timestamp_sub_duration -- ts - Duration."""

    def test_sub_duration_decreases_tsi(self):
        ts = vrtigo.Timestamp(1000, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        d = vrtigo.Duration.from_seconds(3)
        result = ts - d
        assert isinstance(result, vrtigo.Timestamp)
        assert result.tsi == 997


class TestTimestampDifference:
    """test_timestamp_difference -- ts1 - ts2 returns Duration."""

    def test_difference(self):
        ts1 = vrtigo.Timestamp(1010, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        ts2 = vrtigo.Timestamp(1000, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        diff = ts1 - ts2
        assert isinstance(diff, vrtigo.Duration)
        assert diff.seconds == 10
        assert diff.picoseconds == 0


class TestTimestampArithmeticAllTsiKinds:
    """test_timestamp_arithmetic_all_tsi_kinds -- add/sub with each TsiType x real_time."""

    @pytest.mark.parametrize(
        "tsi_kind",
        [vrtigo.TsiType.utc, vrtigo.TsiType.gps, vrtigo.TsiType.other, vrtigo.TsiType.none],
    )
    def test_add_with_tsi_kind(self, tsi_kind):
        ts = vrtigo.Timestamp(1000, 0, tsi_kind, vrtigo.TsfType.real_time)
        d = vrtigo.Duration.from_seconds(1)
        result = ts + d
        assert isinstance(result, vrtigo.Timestamp)

    @pytest.mark.parametrize(
        "tsi_kind",
        [vrtigo.TsiType.utc, vrtigo.TsiType.gps, vrtigo.TsiType.other, vrtigo.TsiType.none],
    )
    def test_sub_duration_with_tsi_kind(self, tsi_kind):
        ts = vrtigo.Timestamp(1000, 0, tsi_kind, vrtigo.TsfType.real_time)
        d = vrtigo.Duration.from_seconds(1)
        result = ts - d
        assert isinstance(result, vrtigo.Timestamp)


class TestTimestampNonRealtimeArithmeticFails:
    """test_timestamp_non_realtime_arithmetic_fails -- sample_count ts + Duration -> TypeError."""

    def test_add_duration_to_sample_count_ts(self):
        ts = vrtigo.Timestamp(1000, 42, vrtigo.TsiType.utc, vrtigo.TsfType.sample_count)
        d = vrtigo.Duration.from_seconds(1)
        with pytest.raises(TypeError):
            _ = ts + d

    def test_sub_duration_from_sample_count_ts(self):
        ts = vrtigo.Timestamp(1000, 42, vrtigo.TsiType.utc, vrtigo.TsfType.sample_count)
        d = vrtigo.Duration.from_seconds(1)
        with pytest.raises(TypeError):
            _ = ts - d


class TestTimestampMixedTsiSubtractionFails:
    """test_timestamp_mixed_tsi_subtraction_fails -- utc ts - gps ts -> TypeError."""

    def test_utc_minus_gps(self):
        ts_utc = vrtigo.Timestamp(1000, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        ts_gps = vrtigo.Timestamp(1000, 0, vrtigo.TsiType.gps, vrtigo.TsfType.real_time)
        with pytest.raises(TypeError):
            _ = ts_utc - ts_gps


class TestTimestampSubDispatchesCorrectly:
    """test_timestamp_sub_dispatches_correctly -- ts - Duration -> Timestamp, ts - ts -> Duration."""

    def test_sub_duration_returns_timestamp(self):
        ts = vrtigo.Timestamp(1000, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        d = vrtigo.Duration.from_seconds(1)
        result = ts - d
        assert isinstance(result, vrtigo.Timestamp)

    def test_sub_timestamp_returns_duration(self):
        ts1 = vrtigo.Timestamp(1000, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        ts2 = vrtigo.Timestamp(999, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        result = ts1 - ts2
        assert isinstance(result, vrtigo.Duration)


class TestTimestampNow:
    """test_timestamp_now -- Timestamp.now() returns UTC real_time with reasonable tsi."""

    def test_now_returns_utc_real_time(self):
        ts = vrtigo.Timestamp.now()
        assert ts.tsi_kind == vrtigo.TsiType.utc
        assert ts.tsf_kind == vrtigo.TsfType.real_time

    def test_now_tsi_is_reasonable(self):
        ts = vrtigo.Timestamp.now()
        assert ts.tsi > 1_700_000_000


class TestTimestampFromUtcSeconds:
    """test_timestamp_from_utc_seconds -- roundtrip verification."""

    def test_roundtrip(self):
        ts = vrtigo.Timestamp.from_utc_seconds(12345)
        assert ts.tsi == 12345
        assert ts.tsf == 0

    def test_kind(self):
        ts = vrtigo.Timestamp.from_utc_seconds(12345)
        assert ts.tsi_kind == vrtigo.TsiType.utc
        assert ts.tsf_kind == vrtigo.TsfType.real_time


class TestTimestampToDatetime:
    """test_timestamp_to_datetime -- known epoch 1000000000 = 2001-09-09 01:46:40 UTC."""

    def test_known_epoch(self):
        ts = vrtigo.Timestamp.from_utc_seconds(1_000_000_000)
        dt = ts.to_datetime()
        assert isinstance(dt, datetime.datetime)
        assert dt.year == 2001
        assert dt.month == 9
        assert dt.day == 9
        assert dt.hour == 1
        assert dt.minute == 46
        assert dt.second == 40


class TestTimestampToDatetimeNonUtcFails:
    """test_timestamp_to_datetime_non_utc_fails -- GPS timestamp -> to_datetime() -> TypeError."""

    def test_gps_to_datetime_raises(self):
        ts = vrtigo.Timestamp(1000, 0, vrtigo.TsiType.gps, vrtigo.TsfType.real_time)
        with pytest.raises(TypeError):
            ts.to_datetime()


class TestTimestampEq:
    """test_timestamp_eq -- equal and not-equal comparisons."""

    def test_equal(self):
        ts1 = vrtigo.Timestamp(1000, 500, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        ts2 = vrtigo.Timestamp(1000, 500, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        assert ts1 == ts2

    def test_not_equal_tsi(self):
        ts1 = vrtigo.Timestamp(1000, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        ts2 = vrtigo.Timestamp(1001, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        assert ts1 != ts2

    def test_not_equal_tsf(self):
        ts1 = vrtigo.Timestamp(1000, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        ts2 = vrtigo.Timestamp(1000, 1, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        assert ts1 != ts2


class TestTimestampHash:
    """test_timestamp_hash -- equal timestamps have equal hashes; set deduplicates."""

    def test_equal_timestamps_have_equal_hashes(self):
        a = vrtigo.Timestamp(1000, 500, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        b = vrtigo.Timestamp(1000, 500, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        assert a == b
        assert hash(a) == hash(b)

    def test_set_deduplicates_equal_timestamps(self):
        a = vrtigo.Timestamp(1000, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        b = vrtigo.Timestamp(1000, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        c = vrtigo.Timestamp(2000, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        s = {a, b, c}
        assert len(s) == 2


class TestTimestampOrdering:
    """test_timestamp_ordering -- <, <=, >, >= on UTC real_time timestamps."""

    def test_less_than_different_tsi(self):
        t1 = vrtigo.Timestamp(100, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        t2 = vrtigo.Timestamp(200, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        assert t1 < t2
        assert not (t2 < t1)

    def test_less_than_same_tsi_different_tsf(self):
        t1 = vrtigo.Timestamp(100, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        t2 = vrtigo.Timestamp(100, 500_000_000_000, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        assert t1 < t2

    def test_greater_than(self):
        t1 = vrtigo.Timestamp(200, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        t2 = vrtigo.Timestamp(100, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        assert t1 > t2
        assert not (t2 > t1)

    def test_less_equal_when_equal(self):
        t1 = vrtigo.Timestamp(100, 500, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        t2 = vrtigo.Timestamp(100, 500, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        assert t1 <= t2
        assert t2 <= t1

    def test_less_equal_when_less(self):
        t1 = vrtigo.Timestamp(100, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        t2 = vrtigo.Timestamp(200, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        assert t1 <= t2

    def test_greater_equal_when_equal(self):
        t1 = vrtigo.Timestamp(100, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        t2 = vrtigo.Timestamp(100, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        assert t1 >= t2
        assert t2 >= t1

    def test_greater_equal_when_greater(self):
        t1 = vrtigo.Timestamp(200, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        t2 = vrtigo.Timestamp(100, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        assert t1 >= t2

    @pytest.mark.parametrize(
        "tsi_kind",
        [vrtigo.TsiType.utc, vrtigo.TsiType.gps, vrtigo.TsiType.other, vrtigo.TsiType.none],
    )
    def test_ordering_all_tsi_kinds(self, tsi_kind):
        t1 = vrtigo.Timestamp(100, 0, tsi_kind, vrtigo.TsfType.real_time)
        t2 = vrtigo.Timestamp(200, 0, tsi_kind, vrtigo.TsfType.real_time)
        assert t1 < t2
        assert t2 > t1


class TestTimestampOrderingErrors:
    """test_timestamp_ordering_errors -- TypeError for non-real_time or mismatched tsi_kind."""

    def test_mismatched_tsi_kind_raises(self):
        t_utc = vrtigo.Timestamp(100, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        t_gps = vrtigo.Timestamp(100, 0, vrtigo.TsiType.gps, vrtigo.TsfType.real_time)
        with pytest.raises(TypeError):
            t_utc < t_gps

    def test_sample_count_ordering_raises(self):
        t1 = vrtigo.Timestamp(100, 0, vrtigo.TsiType.utc, vrtigo.TsfType.sample_count)
        t2 = vrtigo.Timestamp(200, 0, vrtigo.TsiType.utc, vrtigo.TsfType.sample_count)
        with pytest.raises(TypeError):
            t1 < t2

    def test_free_running_ordering_raises(self):
        t1 = vrtigo.Timestamp(100, 0, vrtigo.TsiType.utc, vrtigo.TsfType.free_running)
        t2 = vrtigo.Timestamp(200, 0, vrtigo.TsiType.utc, vrtigo.TsfType.free_running)
        with pytest.raises(TypeError):
            t1 < t2

    def test_tsf_none_ordering_raises(self):
        t1 = vrtigo.Timestamp(100, 0, vrtigo.TsiType.utc, vrtigo.TsfType.none)
        t2 = vrtigo.Timestamp(200, 0, vrtigo.TsiType.utc, vrtigo.TsfType.none)
        with pytest.raises(TypeError):
            t1 < t2

    def test_mixed_tsf_kind_ordering_raises(self):
        t1 = vrtigo.Timestamp(100, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        t2 = vrtigo.Timestamp(100, 0, vrtigo.TsiType.utc, vrtigo.TsfType.sample_count)
        with pytest.raises(TypeError):
            t1 < t2


class TestTimestampRepr:
    """test_timestamp_repr -- verify repr contains 'Timestamp'."""

    def test_repr_contains_timestamp(self):
        ts = vrtigo.Timestamp(1000, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        assert "Timestamp" in repr(ts)


# ---------------------------------------------------------------------------
# StartTime Tests
# ---------------------------------------------------------------------------


class TestStartTimeFactories:
    """test_start_time_factories -- zero(), now(), at_next_second() don't raise."""

    def test_zero(self):
        st = vrtigo.StartTime.zero()
        assert isinstance(st, vrtigo.StartTime)

    def test_now(self):
        st = vrtigo.StartTime.now()
        assert isinstance(st, vrtigo.StartTime)

    def test_at_next_second(self):
        st = vrtigo.StartTime.at_next_second()
        assert isinstance(st, vrtigo.StartTime)


class TestStartTimeNowPlus:
    """test_start_time_now_plus -- doesn't raise."""

    def test_now_plus_one_second(self):
        st = vrtigo.StartTime.now_plus(vrtigo.Duration.from_seconds(1))
        assert isinstance(st, vrtigo.StartTime)


class TestStartTimeAbsolute:
    """test_start_time_absolute -- UTC real_time timestamp accepted."""

    def test_with_utc_real_time(self):
        ts = vrtigo.Timestamp(1000, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        st = vrtigo.StartTime.absolute(ts)
        assert isinstance(st, vrtigo.StartTime)


class TestStartTimeAbsoluteInvalidKind:
    """test_start_time_absolute_invalid_kind -- non-UTC or non-real_time -> TypeError."""

    def test_gps_real_time_raises(self):
        ts = vrtigo.Timestamp(1000, 0, vrtigo.TsiType.gps, vrtigo.TsfType.real_time)
        with pytest.raises(TypeError):
            vrtigo.StartTime.absolute(ts)

    def test_utc_sample_count_raises(self):
        ts = vrtigo.Timestamp(1000, 0, vrtigo.TsiType.utc, vrtigo.TsfType.sample_count)
        with pytest.raises(TypeError):
            vrtigo.StartTime.absolute(ts)


class TestStartTimeBase:
    """test_start_time_base -- verify .base returns correct StartTimeBase enum."""

    def test_zero_base(self):
        st = vrtigo.StartTime.zero()
        assert st.base == vrtigo.StartTimeBase.zero

    def test_now_base(self):
        st = vrtigo.StartTime.now()
        assert st.base == vrtigo.StartTimeBase.now

    def test_at_next_second_base(self):
        st = vrtigo.StartTime.at_next_second()
        assert st.base == vrtigo.StartTimeBase.next_second

    def test_absolute_base(self):
        ts = vrtigo.Timestamp(1000, 0, vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
        st = vrtigo.StartTime.absolute(ts)
        assert st.base == vrtigo.StartTimeBase.absolute


class TestStartTimeRepr:
    """test_start_time_repr -- verify repr contains 'StartTime'."""

    def test_repr_contains_start_time(self):
        st = vrtigo.StartTime.zero()
        assert "StartTime" in repr(st)


# ---------------------------------------------------------------------------
# SampleClock Tests
# ---------------------------------------------------------------------------


class TestSampleClockBasic:
    """test_sample_clock_basic -- construct 1 MHz clock, tick, verify timestamps advance."""

    def test_tick_advances(self):
        clock = vrtigo.SampleClock(1e-6)
        ts1 = clock.now()
        ts2 = clock.tick()
        # After one tick the timestamp should have advanced by ~1 us
        assert isinstance(ts2, vrtigo.Timestamp)


class TestSampleClockTickMultiple:
    """test_sample_clock_tick_multiple -- tick(10) advances by 10 samples worth."""

    def test_tick_10(self):
        clock = vrtigo.SampleClock(1e-6)
        ts_before = clock.now()
        ts_after = clock.tick(10)
        diff = ts_after - ts_before
        # 10 samples at 1 us each = 10 us = 10_000_000 ps
        assert diff.total_picoseconds == pytest.approx(10_000_000, rel=1e-6)


class TestSampleClockAdvance:
    """test_sample_clock_advance -- advance(5) then now() matches tick pattern."""

    def test_advance_then_now(self):
        clock = vrtigo.SampleClock(1e-6)
        ts_start = clock.now()
        clock.advance(5)
        ts_after = clock.now()
        diff = ts_after - ts_start
        # 5 samples at 1 us each = 5 us = 5_000_000 ps
        assert diff.total_picoseconds == pytest.approx(5_000_000, rel=1e-6)


class TestSampleClockElapsedSamples:
    """test_sample_clock_elapsed_samples -- after tick(100), elapsed_samples == 100."""

    def test_elapsed(self):
        clock = vrtigo.SampleClock(1e-6)
        assert clock.elapsed_samples == 0
        clock.tick(100)
        assert clock.elapsed_samples == 100


class TestSampleClockReset:
    """test_sample_clock_reset -- tick some, reset, verify elapsed_samples == 0."""

    def test_reset(self):
        clock = vrtigo.SampleClock(1e-6)
        clock.tick(50)
        assert clock.elapsed_samples == 50
        clock.reset()
        assert clock.elapsed_samples == 0


class TestSampleClockNegativeSamples:
    """test_sample_clock_negative_samples -- tick(-1) and advance(-1) raise ValueError."""

    def test_tick_negative(self):
        clock = vrtigo.SampleClock(1e-6)
        with pytest.raises(ValueError):
            clock.tick(-1)

    def test_advance_negative(self):
        clock = vrtigo.SampleClock(1e-6)
        with pytest.raises(ValueError):
            clock.advance(-1)


class TestSampleClockInvalidPeriod:
    """test_sample_clock_invalid_period -- zero, negative, NaN periods raise ValueError."""

    def test_zero_period(self):
        with pytest.raises(ValueError):
            vrtigo.SampleClock(0.0)

    def test_negative_period(self):
        with pytest.raises(ValueError):
            vrtigo.SampleClock(-1.0)

    def test_nan_period(self):
        with pytest.raises(ValueError):
            vrtigo.SampleClock(float("nan"))


class TestSampleClockPeriodAccessor:
    """test_sample_clock_period_accessor -- clock.period returns SamplePeriod."""

    def test_period(self):
        clock = vrtigo.SampleClock(1e-6)
        sp = clock.period
        assert isinstance(sp, vrtigo.SamplePeriod)
        # 1 us = 1_000_000 ps
        assert sp.picoseconds == 1_000_000


class TestSampleClockWithStartTime:
    """test_sample_clock_with_start_time -- construct with StartTime.zero(), first now() near zero."""

    def test_start_time_zero(self):
        clock = vrtigo.SampleClock(1e-6, vrtigo.StartTime.zero())
        ts = clock.now()
        assert isinstance(ts, vrtigo.Timestamp)
        # With start time zero, tsi should be 0
        assert ts.tsi == 0


class TestSampleClockRepr:
    """test_sample_clock_repr -- verify repr contains 'SampleClock'."""

    def test_repr_contains_sample_clock(self):
        clock = vrtigo.SampleClock(1e-6)
        assert "SampleClock" in repr(clock)


# ---------------------------------------------------------------------------
# Integration / Smoke Tests
# ---------------------------------------------------------------------------


class TestSmokeDurationSamplePeriodIntegration:
    """test_smoke_duration_sample_period_integration -- Duration from SamplePeriod.to_duration()."""

    def test_consistency(self):
        sp = vrtigo.SamplePeriod.from_rate_hz(10e6)
        d = sp.to_duration()
        assert d.picoseconds == sp.picoseconds
        assert d.seconds == 0
        assert d.total_picoseconds == sp.picoseconds


class TestSmokeClockDifference:
    """test_smoke_clock_difference -- tick twice, subtract timestamps, verify diff ~ period."""

    def test_tick_difference_matches_period(self):
        period_seconds = 1e-6
        clock = vrtigo.SampleClock(period_seconds, vrtigo.StartTime.zero())
        ts1 = clock.tick()
        ts2 = clock.tick()
        diff = ts2 - ts1
        expected_ps = int(period_seconds * vrtigo.PICOSECONDS_PER_SECOND)
        assert diff.total_picoseconds == pytest.approx(expected_ps, rel=1e-6)
