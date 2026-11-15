"""Issue #6, step 4: coverage of the published day-ahead band, measured on table out. Describes the band only."""
import json

import pandas as pd
import pytest

from conftest import FakeHA, clear_sky_weather
from test_run import NOW, eng  # noqa: F401  (fixture)


def _archive(eng, ys, p10=0.8, p50=1.0, p90=1.2, issue_hour=20, hourly=None):
    """One published day-ahead band per day (9 daylight hours), and the actual of each day as a flat hourly value."""
    tz = eng.tz
    c = eng.db()
    rows, act = [], {}
    for i, y in enumerate(ys, start=1):
        day = NOW.tz_convert(tz).normalize() - pd.DateOffset(days=i)
        issue = int((day - pd.Timedelta(hours=24 - issue_hour)).timestamp())
        for h in range(8, 17):
            t = (day + pd.Timedelta(hours=h)).tz_convert('UTC')
            rows.append((issue, int(t.timestamp()), p10, (p10 + p50) / 2, p50, p90, 3))
            act[t] = hourly(i, h) if hourly else y
    c.executemany('insert into out values(?,?,?,?,?,?,?)', rows)
    c.commit()
    a = pd.Series(act).sort_index()
    return c, pd.DataFrame({'total': a, 'ok': True})


def test_counts_days_and_hours_outside_the_band(eng):
    ys = [0.5, 0.5, 1.5] + [1.0] * 7                       # 2 days below P10, 1 above P90, out of 10
    c, a = _archive(eng, ys)
    r = eng.band_coverage(c, a)
    assert r['days'] == 10 and r['hours'] == 90 and r['window_days'] == 60
    assert r['days_below_p10_pct'] == pytest.approx(20.0) and r['days_above_p90_pct'] == pytest.approx(10.0)
    assert r['hours_below_p10_pct'] == pytest.approx(20.0) and r['hours_above_p90_pct'] == pytest.approx(10.0)


def test_hours_not_ok_are_left_out(eng):
    c, a = _archive(eng, [0.5] * 10)
    a.loc[a.index[:9], 'ok'] = False                         # the most recent day is curtailed: not scored
    assert eng.band_coverage(c, a)['days'] == 9


def test_too_few_days_gives_counts_only(eng):
    c, a = _archive(eng, [1.0] * 5)
    r = eng.band_coverage(c, a)
    assert r == {'window_days': 60, 'days': 5, 'hours': 45}


def test_issue_after_21h_is_not_day_ahead(eng):
    c, a = _archive(eng, [1.0] * 10, issue_hour=23)          # issued at 23:00 the evening before: too late
    assert eng.band_coverage(c, a)['days'] == 0


def test_learn_stores_and_run_publishes_coverage(eng, monkeypatch):
    cfg = eng.cfg
    cfg.history_days = 30
    w = clear_sky_weather(cfg, (NOW - pd.Timedelta(days=31)).floor('D'), 32 * 24)
    F = eng.pv(w)
    eng._ha = FakeHA({x.entities[0]: F[x.name] * 900 for x in cfg.arrays})
    monkeypatch.setattr(eng, 'fetch_prev_runs', lambda m, lead, s, e: w)
    assert eng.learn() == 0
    cal = json.load(open(eng.cal_path))
    assert cal['coverage']['days'] == 0                     # nothing published yet: counts only
    eng._ha = FakeHA()
    eng.run()
    assert eng.ha.states['sensor.pv_forecast_today'][1]['band_coverage'] == cal['coverage']


def test_hours_out_while_the_day_is_in(eng):
    """Hours alternate 0.5 / 1.5 around a P50 of 1.0: every hour is outside the band, every day total inside."""
    c, a = _archive(eng, [1.0] * 10, hourly=lambda i, h: 0.5 if h % 2 else 1.5)
    r = eng.band_coverage(c, a)
    assert r['days_below_p10_pct'] == 0 and r['days_above_p90_pct'] == 0
    assert r['hours_below_p10_pct'] + r['hours_above_p90_pct'] == pytest.approx(100.0)


def test_band_clipped_flat_is_not_scored(eng):
    c, a = _archive(eng, [1.02] * 10, p10=1.0, p50=1.0, p90=1.0)   # P10 = P90 = the AC limit
    assert eng.band_coverage(c, a) == {'window_days': 60, 'days': 0, 'hours': 0}
