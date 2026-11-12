import numpy as np
import pandas as pd

from pvensemble.engine import Engine, hmask

from conftest import FakeHA, clear_sky_weather

FULL_CUT = {'step': 5, 'bins': {str(z): [90.0, 0.0] for z in range(0, 360, 5)}}


def test_hmask_bins():
    m = {'step': 5, 'bins': {'180': [20.0, 0.3]}}
    saz = pd.Series([181.0, 181.0, 90.0])
    el = pd.Series([10.0, 30.0, 10.0])
    assert list(hmask(m, saz, el)) == [0.3, 1.0, 1.0]       # below h in its bin / above h / bin without mask
    assert hmask(None, saz, el) == 1.0


def test_mask_cuts_direct_only(cfg):
    eng = Engine(cfg, ha=FakeHA())
    w = clear_sky_weather(cfg, '2026-06-21', 24)
    free = eng.pv(w, horizon={})
    cut = eng.pv(w, horizon={'south': FULL_CUT})
    assert cut.south.sum() < free.south.sum()
    assert np.allclose(cut.west, free.west)                 # other arrays untouched
    diffuse_only = w.assign(direct_normal_irradiance=0.0, shortwave_radiation=w.diffuse_radiation)
    assert np.allclose(eng.pv(diffuse_only, horizon={}).south, eng.pv(diffuse_only, horizon={'south': FULL_CUT}).south)
    assert cut.south.sum() > 0                              # diffuse still produces behind the obstacle


def test_mask_ignored_for_multi_face_arrays(cfg):
    from pvensemble.config import Face
    cfg.arrays[1].faces.append(Face(90, 30, 2.0))
    eng = Engine(cfg, ha=FakeHA())
    w = clear_sky_weather(cfg, '2026-06-21', 24)
    assert np.allclose(eng.pv(w, horizon={}).west, eng.pv(w, horizon={'west': FULL_CUT}).west)


def test_horizon_learns_a_synthetic_obstacle(cfg, monkeypatch, capsys):
    """Clear-sky history where the west array loses 80 % of its beam below 20 deg, azimuth 250-290: a mask appears there."""
    import json

    from pvensemble import horizon

    eng = Engine(cfg, ha=FakeHA())
    now = pd.Timestamp('2026-06-30T12:00Z')
    monkeypatch.setattr(pd.Timestamp, 'now', classmethod(lambda cls, tz=None: now.tz_convert(tz) if tz else now))
    w = clear_sky_weather(cfg, '2026-05-25', 40 * 24)
    parts = horizon.parts(eng, w)
    blocked = (parts.saz >= 250) & (parts.saz < 290) & (parts.el < 20)
    west = (parts.west_d + parts.west_b.where(~blocked, parts.west_b * 0.2)) * 0.95
    stats = {'sensor.inverter_pv1_power': (parts.south_b + parts.south_d) * 1000, 'sensor.inverter_pv2_power': west * 1000}
    eng._ha = FakeHA(stats)
    monkeypatch.setattr(eng, 'fetch_prev_runs', lambda m, lead, s, e: w)
    cfg.history_days = 30
    assert horizon.main(eng, write=True) == 0
    hz = json.load(open(eng.hz_path))
    assert hz['clear_days'] >= 15 and hz['reference'] == 'south'
    assert hz['south']['bins'] == {}
    bins = hz['west']['bins']
    assert bins and all(240 <= int(z) < 295 for z in bins)   # mask only where the obstacle is
    assert all(v[1] < 0.5 for v in bins.values())
