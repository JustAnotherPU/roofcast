import json
import sqlite3

import pandas as pd
import pytest

from roofcast import engine as E
from roofcast.engine import Engine

from conftest import FakeHA, clear_sky_weather

NOW = pd.Timestamp('2026-10-24T10:05Z')


class Resp:
    def __init__(self, data):
        self.data = data

    def json(self):
        return self.data


@pytest.fixture
def eng(cfg, monkeypatch):
    w = clear_sky_weather(cfg, '2026-10-23', 5 * 24 + 1)
    hourly = {'time': [(t + pd.Timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M') for t in w.index],   # Open-Meteo labels the hour END
              **{v: w[v].tolist() for v in E.VARS}}
    monkeypatch.setattr(E, 'om_get', lambda url, params, **k: Resp({'hourly': hourly}))
    cfg.models = cfg.models[:3]
    e = Engine(cfg, ha=FakeHA())
    monkeypatch.setattr(e, 'now', lambda: NOW)
    return e


def test_cold_start_run_publishes(eng):
    assert eng.run() == 0                                    # learn fails (no history), run still publishes uncalibrated
    st, at = eng.ha.states['sensor.pv_forecast_today']
    assert st > 0 and at['models_ok'] == 3 and at['quantiles_fallback'] is True
    assert at['p10'] < st < at['p90']
    det = at['detailedForecast']
    assert len(det) == 2 * (24 + 25 + 24)
    assert det[0]['period_start'] == '2026-10-24T00:00:00+02:00'
    assert sum(r['pv_estimate'] for r in det[:48]) / 2 == pytest.approx(st, abs=0.05)
    per = at['per_array']
    assert set(per) == {'start', 'step_min', 'south', 'west'} and len(per['south']) == 24 + 25
    assert 'sensor.pv_forecast_tomorrow' in eng.ha.states


def test_archive_is_raw_and_k_applies_to_published(eng):
    cal = {'generated': 'x', 'n_hours': 0, 'models': {}, 'level': {'k': 1.1},
           'quantiles': {'p10': 0.6, 'p25': 0.8, 'p50': 1.0, 'p90': 1.2, 'n_days': 30}}
    json.dump(cal, open(eng.cal_path, 'w'))
    eng.run()
    c = sqlite3.connect(eng.db_path)
    t = int(pd.Timestamp('2026-10-24T11:00Z').timestamp())
    raw = [r[0] for r in c.execute('select pv from nwp where target = ?', (t,))]
    p50 = c.execute('select p50 from out where target = ?', (t,)).fetchone()[0]
    assert len(raw) == 3
    assert p50 == pytest.approx(min(sum(raw) / 3 * 1.1, eng.cfg.ac_limit_kw))   # members archived BEFORE k


def test_knmi_corrects_published_curve_only(eng):
    eng.cfg.knmi_enabled = True
    eng.run()
    c = sqlite3.connect(eng.db_path)
    t = int(pd.Timestamp('2026-10-24T12:00Z').timestamp())
    p50 = c.execute('select p50 from out where target = ?', (t,)).fetchone()[0]
    c.execute('insert into ext values(?,?,?,?,?,?)', (int(NOW.timestamp()), 'knmi', t, p50 * 0.5, p50, None))
    c.commit()
    before = {r['period_start']: r['pv_estimate'] for r in eng.ha.states['sensor.pv_forecast_today'][1]['detailedForecast']}
    eng.run()
    at = eng.ha.states['sensor.pv_forecast_today'][1]
    after = {r['period_start']: r['pv_estimate'] for r in at['detailedForecast']}
    key = '2026-10-24T14:00:00+02:00'
    assert at['knmi']['hours_corrected'] == 1
    assert after[key] == pytest.approx(before[key] * 0.5, rel=1e-3)
    assert c.execute('select p50 from out where target = ? order by issue desc', (t,)).fetchone()[0] == pytest.approx(p50)


def test_learn_end_to_end_on_synthetic_history(eng, monkeypatch):
    cfg = eng.cfg
    cfg.history_days = 30
    start = (NOW - pd.Timedelta(days=31)).floor('D')
    w = clear_sky_weather(cfg, start, 32 * 24)
    F = eng.pv(w)
    ratio = {'south': 0.9, 'west': 0.8}
    stats = {x.entities[0]: F[x.name] * ratio[x.name] * 1000 for x in cfg.arrays}     # HA statistics in W
    eng._ha = FakeHA(stats)
    monkeypatch.setattr(eng, 'fetch_prev_runs', lambda m, lead, s, e: w)
    assert eng.learn() == 0
    cal = json.load(open(eng.cal_path))
    assert len(cal['models']) == 2 * len(cfg.models)
    ent = cal['models'][f'{cfg.models[0]}|1']
    assert ent['south']['g'] == pytest.approx(0.9) and ent['west']['g'] == pytest.approx(0.8)
    assert cal['quantiles']['p50'] == pytest.approx(1.0) and not cal['quantiles']['fallback']
    assert cal['level']['k'] == 1.0 and cal['level']['days'] == 0      # no archived day-ahead yet
    c = sqlite3.connect(eng.db_path)
    assert c.execute('select count(*) from actual').fetchone()[0] > 24 * 25


def test_level_measured_on_archived_day_ahead(eng):
    """Archive says 10 % too much every day for 14 days: k comes out at ~0.909 from table nwp, out of sample."""
    tz = eng.tz
    c = eng.db()
    hours, rows = [], []
    for i in range(1, 15):
        day = (NOW.tz_convert(tz).normalize() - pd.DateOffset(days=i))
        issue = int((day - pd.Timedelta(hours=4)).timestamp())          # 20:00 local the evening before
        for h in range(8, 17):
            t = (day + pd.Timedelta(hours=h)).tz_convert('UTC')
            hours.append(t)
            for m in eng.cfg.models:
                rows.append((issue, m, int(t.timestamp()), 0, 0, 0, 0, 0, 0, 0, 1.1))
    c.executemany('insert into nwp values(?,?,?,?,?,?,?,?,?,?,?)', rows)
    c.commit()
    a = pd.DataFrame({'total': 1.0, 'ok': True}, index=pd.DatetimeIndex(sorted(hours)))
    t = eng.level_day_ahead(c, a)
    assert len(t) == 14
    k, raw = E.level_factor(t.y, t.f, [(NOW.tz_convert(tz).date() - d).days for d in t.index])
    assert raw == pytest.approx(1 / 1.1) and k == pytest.approx(1 / 1.1)


def test_failed_learn_keeps_previous_calibration(eng, monkeypatch):
    cal = {'generated': 'old', 'n_hours': 1, 'models': {'icon_seamless|1': {'south': {'g': 0.9, 'shape': {}}}},
           'quantiles': {'p10': 0.6, 'p25': 0.8, 'p50': 1.0, 'p90': 1.2}}
    json.dump(cal, open(eng.cal_path, 'w'))
    idx = pd.date_range(NOW - pd.Timedelta(days=20), NOW, freq='h')
    eng._ha = FakeHA({x.entities[0]: pd.Series(1000.0, index=idx) for x in eng.cfg.arrays})

    def down(*a, **k):
        raise TimeoutError('previous runs API down')
    monkeypatch.setattr(eng, 'fetch_prev_runs', down)
    assert eng.learn() == 1
    assert json.load(open(eng.cal_path)) == cal


def test_learn_with_nothing_fitted_writes_nothing(eng, monkeypatch):
    idx = pd.date_range(NOW - pd.Timedelta(days=20), NOW, freq='h')
    eng._ha = FakeHA({x.entities[0]: pd.Series(1000.0, index=idx) for x in eng.cfg.arrays})

    def down(*a, **k):
        raise TimeoutError('previous runs API down')
    monkeypatch.setattr(eng, 'fetch_prev_runs', down)
    import os
    assert eng.learn() == 1
    assert not os.path.exists(eng.cal_path)


def test_run_skips_models_without_calibration(eng):
    m0 = eng.cfg.models[0]
    cal = {'generated': 'x', 'n_hours': 0, 'models': {f'{m0}|1': {'south': {'g': 0.9, 'shape': {}}, 'west': {'g': 0.9, 'shape': {}}}},
           'quantiles': {'p10': 0.6, 'p25': 0.8, 'p50': 1.0, 'p90': 1.2}}
    json.dump(cal, open(eng.cal_path, 'w'))
    assert eng.run() == 0
    at = eng.ha.states['sensor.pv_forecast_today'][1]
    assert at['models_ok'] == 1 and at['models'][m0] == 'ok'
    assert all(at['models'][m] == 'no calibration' for m in eng.cfg.models[1:])
    c = sqlite3.connect(eng.db_path)
    assert {r[0] for r in c.execute('select distinct provider from nwp')} == {m0}


def test_actuals_without_any_statistics(eng):
    eng._ha = FakeHA({'sensor.something_else': pd.Series([1.0], index=[NOW.floor('h')])})
    with pytest.raises(ValueError, match='no long-term statistics'):
        eng.actuals()


def test_missing_soc_statistics_is_logged(eng, capsys):
    eng.cfg.soc_entity, eng.cfg.grid_entity = 'sensor.battery_soc', 'sensor.grid_power'
    idx = pd.date_range(NOW - pd.Timedelta(days=2), NOW, freq='h')
    eng._ha = FakeHA({**{x.entities[0]: pd.Series(1000.0, index=idx) for x in eng.cfg.arrays},
                      'sensor.grid_power': pd.Series(0.0, index=idx)})
    a = eng.actuals()
    assert a.ok.any()
    assert 'no statistics for sensor.battery_soc' in capsys.readouterr().out


def test_learn_prunes_old_long_lead_rows(eng, monkeypatch):
    c = eng.db()
    old = int((NOW - pd.Timedelta(days=40)).timestamp())
    recent = int((NOW - pd.Timedelta(days=20)).timestamp())
    rows = [(i, 'm', i + 3600 * h, h, 0, 0, 0, 0, 0, 0, 1.0) for i in (old, recent) for h in (0, 1, 2, 30)]
    c.executemany('insert into nwp values(?,?,?,?,?,?,?,?,?,?,?)', rows)
    c.commit()
    cfg = eng.cfg
    cfg.history_days = 30
    w = clear_sky_weather(cfg, (NOW - pd.Timedelta(days=31)).floor('D'), 32 * 24)
    F = eng.pv(w)
    eng._ha = FakeHA({x.entities[0]: F[x.name] * 900 for x in cfg.arrays})
    monkeypatch.setattr(eng, 'fetch_prev_runs', lambda m, lead, s, e: w)
    assert eng.learn() == 0
    left = sorted((r[0] == old, r[1]) for r in c.execute("select issue, lead_h from nwp where provider = 'm'"))
    assert left == [(False, 0), (False, 1), (False, 2), (False, 30), (True, 0), (True, 1)]


def test_published_attributes_size(eng):
    eng.run()
    size = len(json.dumps(eng.ha.states['sensor.pv_forecast_today'][1]))
    print('attributes bytes', size)
    assert size > 16384      # documented in the README: exclude from the recorder
