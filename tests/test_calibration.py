import numpy as np
import pandas as pd
import pytest

from roofcast.config import Array, Face
from roofcast.engine import Engine, binkey, quantiles_from_ratios

from conftest import FakeHA, clear_sky_weather


def actual_frame(eng, F, ratio):
    a = pd.DataFrame({x.name: F[x.name] * ratio[x.name] for x in eng.cfg.arrays})
    a['total'] = sum(a[x.name] for x in eng.cfg.arrays)
    a['ok'] = True
    a['okb'] = True
    return a


def test_level_and_flat_shape_are_learned(cfg):
    eng = Engine(cfg, ha=FakeHA())
    F = eng.pv(clear_sky_weather(cfg, '2026-06-01', 20 * 24))
    ent = eng.fit_model(F, actual_frame(eng, F, {'south': 0.9, 'west': 0.8}))
    assert ent['south']['g'] == pytest.approx(0.9)
    assert ent['west']['g'] == pytest.approx(0.8)
    assert np.allclose(list(ent['south']['shape'].values()), 1.0)


def test_shape_learns_a_shaded_bin(cfg):
    eng = Engine(cfg, ha=FakeHA())
    F = eng.pv(clear_sky_weather(cfg, '2026-06-01', 20 * 24))
    kb = binkey(F)
    shaded = kb == kb[F.south.idxmax()]                     # the bin of the daily peak loses half its output
    a = actual_frame(eng, F, {'south': 1.0, 'west': 1.0})
    a.loc[shaded, 'south'] *= 0.5
    a['total'] = a.south + a.west
    ent = eng.fit_model(F, a)
    s = ent['south']
    peak_bin = kb[F.south.idxmax()]
    assert s['shape'][peak_bin] < 0.8                       # pulled toward 0.5, shrunk toward the level
    cal = eng.apply_cal(F, kb, ent, cfg.arrays[0])
    assert cal[shaded].sum() < F.south[shaded].sum() * 0.85


def test_bad_unit_is_refused(cfg):
    eng = Engine(cfg, ha=FakeHA())
    F = eng.pv(clear_sky_weather(cfg, '2026-06-01', 10 * 24))
    with pytest.raises(ValueError, match='power_unit'):
        eng.fit_model(F, actual_frame(eng, F, {'south': 1000.0, 'west': 1000.0}))   # W read as kW


def test_new_array_uses_start_efficiency_then_level_only(cfg):
    since = pd.Timestamp('2026-06-15T00:00Z')
    cfg.arrays.append(Array('east', [Face(90, 30, 3.0)], ['sensor.inverter2_pv1_power'], since=since))
    eng = Engine(cfg, ha=FakeHA())
    F = eng.pv(clear_sky_weather(cfg, '2026-06-01', 20 * 24))
    assert F.east[F.index < since].sum() == 0 and F.east[F.index >= since].sum() > 0   # 0 before `since`
    a = actual_frame(eng, F, {'south': 0.9, 'west': 0.9, 'east': 0.7})

    early = a.index < since + pd.Timedelta(hours=8)          # only a few daylight hours after `since`
    ent = eng.fit_model(F[early], a[early])
    assert ent['east']['g'] == cfg.start_efficiency and ent['east']['shape'] == {}

    ent = eng.fit_model(F, a)                                # ~70 daylight hours: level only, no shape yet
    assert ent['east']['g'] == pytest.approx(0.7) and ent['east']['shape'] == {}
    assert ent['south']['g'] == pytest.approx(0.9)          # older arrays are not disturbed by the newcomer

    a.east = F.east * 1.4                                    # implausible early ratio: capped at the main array's level
    assert eng.fit_model(F, a)['east']['g'] == pytest.approx(0.9)


def test_cold_start_without_calibration(cfg):
    eng = Engine(cfg, ha=FakeHA())
    F = eng.pv(clear_sky_weather(cfg, '2026-06-01', 48))
    p = eng.apply_cal(F, binkey(F), None, cfg.arrays[0])
    assert np.allclose(p, F.south * cfg.start_efficiency)
    q = quantiles_from_ratios(pd.Series([1.0, 0.9]))
    assert q['fallback'] and q['p50'] == 1.0 and q['p10'] < 1 < q['p90']
    q = quantiles_from_ratios(pd.Series(np.linspace(0.5, 1.5, 40)))
    assert not q['fallback'] and q['p10'] < q['p25'] < q['p50'] < q['p90']


def test_fresh_install_drops_zero_hours_before_the_panels(cfg, monkeypatch):
    """No live archive yet and HA statistics that start with 20 days at 0 W (sensor created before the panels):
    those hours must not reach learning, or P10/P25 come out as 0."""
    import json

    eng = Engine(cfg, ha=FakeHA())
    now = pd.Timestamp('2026-06-30T12:00Z')
    monkeypatch.setattr(pd.Timestamp, 'now', classmethod(lambda cls, tz=None: now.tz_convert(tz) if tz else now))
    w = clear_sky_weather(cfg, '2026-05-01', 61 * 24)
    F = eng.pv(w)
    before = F.index < pd.Timestamp('2026-05-21', tz='UTC')
    stats = {x.entities[0]: (F[x.name] * 0.9 * 1000).where(~before, 0.0)
             for x in cfg.arrays}
    eng._ha = FakeHA(stats)
    monkeypatch.setattr(eng, 'fetch_prev_runs', lambda m, lead, s, e: w)
    cfg.history_days = 60
    assert eng.learn() == 0
    q = json.load(open(eng.cal_path))['quantiles']
    assert q['p10'] > 0.8 and q['p25'] > 0.8, q
