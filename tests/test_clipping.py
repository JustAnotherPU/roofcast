"""Issue #5: an oversized array clipped at the inverter limit must not be learned as lost yield, and the level factor
must apply before the AC clip, not after it."""
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from roofcast.config import ConfigError, parse

from conftest import FakeHA, clear_sky_weather
from test_config import mk
from test_run import NOW, eng  # noqa: F401  (fixture)

LIMIT = 5.0          # kW, on the example's 10 kWp: clear days clip from mid-morning to mid-afternoon


def _history(eng, monkeypatch, noise=0.0):
    """30 clear days; the roof makes 0.9 x pvlib on each array (times hourly forecast error), measured as min(DC, AC),
    split pro rata."""
    cfg = eng.cfg
    cfg.history_days = 30
    w = clear_sky_weather(cfg, (NOW - pd.Timedelta(days=31)).floor('D'), 32 * 24)
    F = eng.pv(w)
    err = pd.Series(np.random.default_rng(1).normal(1.0, noise, len(F)), index=F.index).clip(lower=0.3)
    F = F.mul(err, axis=0)
    dc = sum(F[x.name] * 0.9 for x in cfg.arrays)
    scale = (LIMIT / dc).clip(upper=1.0).fillna(1.0)
    eng._ha = FakeHA({x.entities[0]: F[x.name] * 0.9 * scale * 1000 for x in cfg.arrays})
    monkeypatch.setattr(eng, 'fetch_prev_runs', lambda m, lead, s, e: w)
    assert (dc > LIMIT).sum() > 90       # the setup really clips


def _g(eng):
    cal = json.load(open(eng.cal_path))
    ent = cal['models'][f'{eng.cfg.models[0]}|1']
    return ent['south']['g'], ent['west']['g']


def test_clipped_hours_learned_as_lost_yield_without_the_option(eng, monkeypatch):
    _history(eng, monkeypatch)
    assert eng.learn() == 0
    s, w = _g(eng)
    assert s < 0.88 and w < 0.88          # the flat tops pull the level down: this is what the option is for


def test_clip_limit_leaves_clipped_hours_out(eng, monkeypatch):
    _history(eng, monkeypatch)
    eng.cfg.clip_limit_kw = LIMIT
    assert eng.actuals().ok.all()                       # censored in learn(), on the forecast, not on the measured hour
    assert eng.learn() == 0
    s, w = _g(eng)
    assert s == pytest.approx(0.9, abs=0.005) and w == pytest.approx(0.9, abs=0.005)


def test_forecast_error_does_not_bias_the_level(eng, monkeypatch):
    """With hourly forecast error, dropping hours on the measured value keeps the hours where the forecast was too
    high near the limit and pulls g low. Censoring on the forecast does not."""
    _history(eng, monkeypatch, noise=0.15)
    eng.cfg.clip_limit_kw = LIMIT
    assert eng.learn() == 0
    s, w = _g(eng)
    assert s == pytest.approx(0.9, abs=0.02) and w == pytest.approx(0.9, abs=0.02)


def test_default_keeps_every_hour(eng, monkeypatch):
    _history(eng, monkeypatch)
    assert eng.cfg.clip_limit_kw is None
    a = eng.actuals()
    assert a.ok.all()


def test_k_applies_before_the_clip(eng):
    """k = 0.9 with every member far above the limit: P50 is the limit, not 0.9 x the limit."""
    eng.cfg.ac_limit_kw = 2.0
    cal = {'generated': 'x', 'n_hours': 0, 'models': {}, 'level': {'k': 0.9},
           'quantiles': {'p10': 0.6, 'p25': 0.8, 'p50': 1.0, 'p90': 1.2, 'n_days': 30}}
    json.dump(cal, open(eng.cal_path, 'w'))
    assert eng.run() == 0
    c = sqlite3.connect(eng.db_path)
    t = int(pd.Timestamp('2026-10-24T11:00Z').timestamp())
    raw = [r[0] for r in c.execute('select pv from nwp where target = ?', (t,))]
    mean = sum(raw) / len(raw)
    assert mean * 0.9 > 2.0 / 0.6                      # every band is above the limit at noon
    p10, p50, p90 = c.execute('select p10, p50, p90 from out where target = ?', (t,)).fetchone()
    assert p50 == pytest.approx(2.0) and p10 == pytest.approx(2.0) and p90 == pytest.approx(2.0)


def test_bands_come_from_the_unclipped_expectation(eng):
    """Where the expectation straddles the limit, P10 is a fraction of the expectation, not of the limit."""
    eng.cfg.ac_limit_kw = 3.0
    cal = {'generated': 'x', 'n_hours': 0, 'models': {}, 'level': {'k': 1.0},
           'quantiles': {'p10': 0.6, 'p25': 0.8, 'p50': 1.0, 'p90': 1.2, 'n_days': 30}}
    json.dump(cal, open(eng.cal_path, 'w'))
    eng.run()
    c = sqlite3.connect(eng.db_path)
    rows = c.execute('select o.p10, o.p50, (select avg(pv) from nwp n where n.target = o.target) from out o').fetchall()
    for p10, p50, mean in rows:
        assert p50 == pytest.approx(min(mean, 3.0))
        assert p10 == pytest.approx(min(mean * 0.6, 3.0))


def test_clip_limit_config():
    assert parse(mk(curtailment={'clip_limit_kw': 7.5})).clip_limit_kw == 7.5
    assert parse(mk()).clip_limit_kw is None
    with pytest.raises(ConfigError, match='clip_limit_kw'):
        parse(mk(curtailment={'clip_limit_kw': 0}))
