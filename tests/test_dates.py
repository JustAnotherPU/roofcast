import pandas as pd
import pytest

from pvensemble.engine import detailed_forecast, local_days

TZ = 'Europe/Amsterdam'


def utc(t):
    t = pd.Timestamp(t)
    return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')


def hourly(start, end):
    idx = pd.date_range(utc(start), utc(end), freq='h', inclusive='left')
    return pd.DataFrame({'p10': 0.5, 'p25': 0.8, 'p50': 1.0, 'p90': 1.2}, index=idx)


@pytest.mark.parametrize('now, day, hours', [
    ('2026-10-25T08:00Z', '2026-10-25', 25),   # clocks go back: 25 h day
    ('2027-03-28T08:00Z', '2027-03-28', 23),   # clocks go forward: 23 h day
    ('2026-10-24T21:30Z', '2026-10-24', 24),   # 23:30 local the evening before
])
def test_local_days_across_dst(now, day, hours):
    d0, d1, d2, d3 = local_days(pd.Timestamp(now), TZ)
    assert str(d0.date()) == day
    for d in (d0, d1, d2, d3):
        assert (d.hour, d.minute) == (0, 0)                 # always local midnight, never 23:00 or 01:00
    assert (d1 - d0) == pd.Timedelta(hours=hours)


@pytest.mark.parametrize('day, slots', [('2026-10-25', 50), ('2027-03-28', 46), ('2026-11-12', 48)])
def test_detailed_forecast_one_day(day, slots):
    d0, d1 = local_days(pd.Timestamp(day + 'T12:00Z'), TZ)[:2]
    Q = hourly(pd.Timestamp(day, tz=TZ) - pd.Timedelta(days=1), pd.Timestamp(day, tz=TZ) + pd.Timedelta(days=2))
    rows = detailed_forecast(Q, d0, d1, TZ)
    assert len(rows) == slots
    starts = [pd.Timestamp(r['period_start']) for r in rows]
    assert starts[0] == d0 and len(set(starts)) == slots
    assert all(b - a == pd.Timedelta(minutes=30) for a, b in zip(starts, starts[1:]))


def test_detailed_forecast_format():
    d0, _, _, d3 = local_days(pd.Timestamp('2026-10-24T10:00Z'), TZ)
    rows = detailed_forecast(hourly('2026-10-23', '2026-10-29'), d0, d3, TZ)
    assert len(rows) == 2 * (24 + 25 + 24)
    r = rows[0]
    assert set(r) == {'period_start', 'pv_estimate', 'pv_estimate10', 'pv_estimate25', 'pv_estimate90'}
    assert r['period_start'] == '2026-10-24T00:00:00+02:00'
    assert rows[-1]['period_start'] == '2026-10-26T23:30:00+01:00'
    assert (r['pv_estimate10'], r['pv_estimate'], r['pv_estimate90']) == (0.5, 1.0, 1.2)
