import sqlite3

import pandas as pd
import pytest
import requests

from roofcast.engine import Engine, detailed_forecast, local_days

from conftest import FakeHA
from test_run import NOW, eng  # noqa: F401  (eng is a fixture)


def split_for(cfg, tz, p50=1.0):
    """per_array_split on 6 days of flat hourly output, south twice the size of west."""
    cfg.timezone = tz
    e = Engine(cfg, ha=FakeHA())
    idx = pd.date_range('2026-10-22', periods=6 * 24, freq='h', tz='UTC')
    Q = pd.DataFrame({k: p50 for k in ('p10', 'p25', 'p50', 'p90')}, index=idx)
    per_array = {'south': {'m': pd.Series(2.0, index=idx)}, 'west': {'m': pd.Series(1.0, index=idx)}}
    d0, d1, d2, d3 = local_days(NOW, tz)
    return e.per_array_split(per_array, Q, d0, d2), (d0, d2)


def test_per_array_half_hour_zones_match_whole_hour_zone(cfg):
    # Amsterdam has a 25 h day on 25 Oct, so compare per hour: flat output split 2:1 gives 2/3 and 1/3 kW in every hour
    ref, _ = split_for(cfg, 'Europe/Amsterdam')
    assert set(ref['south']) == {round(2 / 3, 3)}
    for tz, start in (('Australia/Adelaide', '2026-10-24T00:30:00+10:30'), ('Asia/Kathmandu', '2026-10-24T00:45:00+05:45')):
        per, _ = split_for(cfg, tz)
        assert per is not None
        assert per['south'] == [round(2 / 3, 3)] * 48 and per['west'] == [round(1 / 3, 3)] * 48
        assert per['start'] == start and per['step_min'] == 60


def test_per_array_whole_hour_zone_unchanged(cfg):
    per, (d0, d2) = split_for(cfg, 'Europe/Amsterdam')
    assert per['start'] == '2026-10-24T00:00:00+02:00'
    assert len(per['south']) == 24 + 25                       # the clocks go back on 25 Oct
    # what the old code produced: whole UTC hours from local midnight, same values
    hh = pd.date_range(d0.tz_convert('UTC'), d2.tz_convert('UTC'), freq='h', inclusive='left')
    assert len(hh) == len(per['south'])
    assert per['south'] == [round(2 / 3, 3)] * len(hh) and per['west'] == [round(1 / 3, 3)] * len(hh)


def test_per_array_gap_in_Q_keeps_length_and_start(cfg):
    cfg.timezone = 'Australia/Adelaide'
    e = Engine(cfg, ha=FakeHA())
    idx = pd.date_range('2026-10-22', periods=6 * 24, freq='h', tz='UTC')
    Q = pd.DataFrame({k: 1.0 for k in ('p10', 'p25', 'p50', 'p90')}, index=idx)
    per_array = {'south': {'m': pd.Series(2.0, index=idx)}, 'west': {'m': pd.Series(1.0, index=idx)}}
    d0, d1, d2, d3 = local_days(NOW, cfg.timezone)
    gap = pd.Timestamp('2026-10-24T03:00Z')
    per = e.per_array_split(per_array, Q.drop(gap), d0, d2)
    i = int((gap - pd.Timestamp(per['start'])) / pd.Timedelta(hours=1))
    assert len(per['south']) == 48 and per['south'][i] == 0 and per['south'][i + 1] == round(2 / 3, 3)
    assert per['start'] == detailed_forecast(Q, d0, d2, cfg.timezone)[0]['period_start']


class DownHA(FakeHA):
    def get_state(self, entity):
        raise requests.ConnectionError('HA unreachable')


def test_run_archives_and_publishes_when_compare_entity_unreadable(eng):  # noqa: F811
    eng.cfg.solcast_entities = ['sensor.solcast_pv_forecast_forecast_today']
    eng._ha = DownHA()
    assert eng.run() == 0
    c = sqlite3.connect(eng.db_path)
    assert c.execute('select count(*) from nwp').fetchone()[0] > 0
    assert c.execute('select count(*) from out').fetchone()[0] > 0
    assert 'sensor.pv_forecast_today' in eng.ha.states


class HalfHA(FakeHA):
    """The first compare entity reads fine, the second one fails."""
    def get_state(self, entity):
        if entity.endswith('tomorrow'):
            raise requests.ConnectionError('HA unreachable')
        t0 = NOW.floor('h')
        rows = [{'period_start': (t0 + pd.Timedelta(minutes=30 * i)).isoformat(), 'pv_estimate': 1.0} for i in range(48)]
        return {'state': '10', 'attributes': {'detailedForecast': rows}}


def test_half_written_snapshot_is_not_kept(eng):  # noqa: F811
    eng.cfg.solcast_entities = ['sensor.solcast_pv_forecast_forecast_today', 'sensor.solcast_pv_forecast_forecast_tomorrow']
    eng._ha = HalfHA()
    assert eng.run() == 0
    c = sqlite3.connect(eng.db_path)
    assert c.execute('select count(*) from ext').fetchone()[0] == 0
    assert c.execute('select count(*) from out').fetchone()[0] > 0


def test_db_uses_wal(eng):  # noqa: F811
    c = eng.db()
    assert c.execute('pragma journal_mode').fetchone()[0] == 'wal'


def test_quality_rows(eng):  # noqa: F811
    c = eng.db()
    today = NOW.tz_convert(eng.tz).normalize()
    act, out = [], []
    for i in (1, 2):
        day = today - pd.DateOffset(days=i)
        issue = int((day - pd.Timedelta(hours=4)).timestamp())          # 20:00 local the evening before
        for h in range(8, 17):
            t = int((day + pd.Timedelta(hours=h)).timestamp())
            act.append((t, 1.0, None, None))
            out.append((issue, t, 1.0, 1.0, 1.1, 1.2, 3))
    c.executemany('insert into actual values(?,?,?,?)', act)
    c.executemany('insert into out values(?,?,?,?,?,?,?)', out)
    c.commit()
    eng.quality(c)
    rows = sorted(c.execute('select day, provider, lead, fc_kwh, real_kwh from quality').fetchall())
    assert [(r[0], r[1], r[2]) for r in rows] == [('2026-10-22', 'ensemble', 'day_ahead'), ('2026-10-23', 'ensemble', 'day_ahead')]
    assert all(r[3] == pytest.approx(9.9) and r[4] == pytest.approx(9.0) for r in rows)
