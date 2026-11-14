"""Failure handling that shows up outside Europe: models without data at the site, short model lists, bad NWP values."""
import glob
import json
import os

import pandas as pd
import pytest
import requests

from roofcast import engine as E
from roofcast import horizon
from roofcast.config import DEFAULT_MODELS
from roofcast.engine import Engine

from conftest import FakeHA, clear_sky_weather

URL = 'https://api.open-meteo.com/v1/forecast'
PARAMS = {'latitude': -33.87, 'longitude': 151.21, 'models': 'knmi_harmonie_arome_europe'}
NOW = pd.Timestamp('2026-10-24T10:05Z')


class FakeResp:
    def __init__(self, status, body=None, reason=''):
        self.status_code, self.body, self.reason = status, body, reason
        self.ok = status < 400

    def json(self):
        if self.body is None:
            raise ValueError('no JSON')
        return self.body


def _hourly(w):
    """Open-Meteo's hourly JSON for a weather frame (it labels each hour by its END)."""
    return {'time': [(t + pd.Timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M') for t in w.index], **{v: w[v].tolist() for v in E.VARS}}


def _write(path, text):
    with open(path, 'w') as fh:
        fh.write(text)


# ---------------------------------------------------------------- om_get
@pytest.fixture
def api(monkeypatch):
    """Replays a list of answers (a FakeResp, or an exception to raise) and records calls and pauses."""
    def play(answers):
        calls, sleeps = [], []

        def get(url, params=None, timeout=None):
            calls.append(params)
            a = answers[min(len(calls), len(answers)) - 1]
            if isinstance(a, Exception):
                raise a
            return a
        monkeypatch.setattr(E.requests, 'get', get)
        monkeypatch.setattr(E.time, 'sleep', sleeps.append)
        return calls, sleeps
    return play


def test_400_fails_at_once_with_open_meteo_reason(api):
    calls, sleeps = api([FakeResp(400, {'error': True, 'reason': 'No data is available for this location'}, 'Bad Request')])
    with pytest.raises(requests.HTTPError) as e:
        E.om_get(URL, PARAMS)
    assert len(calls) == 1 and sleeps == []                  # a model with no data here gives the same 400 on a retry
    assert str(e.value) == 'HTTP 400: No data is available for this location'


@pytest.mark.parametrize('status', [500, 503, 520, 408])
def test_5xx_and_408_still_retried(api, status):
    calls, sleeps = api([FakeResp(status, reason='Server Error'), FakeResp(200, {'hourly': {}})])
    assert E.om_get(URL, PARAMS).status_code == 200
    assert len(calls) == 2 and sleeps == [20]


def test_429_retried_twice_then_raised(api):
    calls, sleeps = api([FakeResp(429, {'error': True, 'reason': 'Minutely API request limit exceeded'})])
    with pytest.raises(requests.HTTPError, match='^HTTP 429: Minutely API request limit exceeded$'):
        E.om_get(URL, PARAMS)
    assert len(calls) == 3 and sleeps == [20, 40]


@pytest.mark.parametrize('body, reason, text', [(None, 'Not Found', 'HTTP 404: Not Found'),
                                                (['not', 'a', 'dict'], '', 'HTTP 404: no reason given')])
def test_error_without_json_reason(api, body, reason, text):
    api([FakeResp(404, body, reason)])
    with pytest.raises(requests.HTTPError) as e:
        E.om_get(URL, PARAMS)
    assert str(e.value) == text


@pytest.mark.parametrize('cls', [requests.ConnectionError, requests.ConnectTimeout, requests.ReadTimeout, requests.exceptions.ProxyError])
def test_connection_error_retried_and_url_query_dropped(api, cls):
    msg = ("HTTPSConnectionPool(host='api.open-meteo.com', port=443): Max retries exceeded with url: "
           "/v1/forecast?latitude=-33.87&longitude=151.21&models=gfs_seamless (Caused by NameResolutionError('failed'))")
    calls, sleeps = api([cls(msg)])
    with pytest.raises(cls) as e:
        E.om_get(URL, PARAMS)
    assert len(calls) == 3 and sleeps == [20, 40]
    assert type(e.value) is cls
    assert '151.21' not in str(e.value) and 'latitude' not in str(e.value)
    assert '/v1/forecast (Caused by' in str(e.value)        # host, path and cause stay readable


# ---------------------------------------------------------------- roofcast horizon
def _clear_history(cfg, monkeypatch):
    """40 clear days of synthetic history (as in test_horizon), so `roofcast horizon` has clear days to learn from."""
    eng = Engine(cfg, ha=FakeHA())
    now = pd.Timestamp('2026-06-30T12:00Z')
    monkeypatch.setattr(pd.Timestamp, 'now', classmethod(lambda cls, tz=None: now.tz_convert(tz) if tz else now))
    w = clear_sky_weather(cfg, '2026-05-25', 40 * 24)
    p = horizon.parts(eng, w)
    eng._ha = FakeHA({'sensor.inverter_pv1_power': (p.south_b + p.south_d) * 950, 'sensor.inverter_pv2_power': (p.west_b + p.west_d) * 950})
    cfg.history_days = 30
    return eng, w


def test_horizon_skips_a_model_without_data(cfg, monkeypatch, capsys):
    eng, w = _clear_history(cfg, monkeypatch)
    dead = cfg.models[1]

    def fetch(m, lead, s, e):
        if m == dead:
            raise requests.HTTPError('HTTP 400: No data is available for this location')
        return w
    monkeypatch.setattr(eng, 'fetch_prev_runs', fetch)
    assert horizon.main(eng, write=True) == 0
    with open(eng.hz_path) as fh:
        assert json.load(fh)['clear_days'] >= 15
    assert f'horizon: {dead} skipped: HTTP 400' in capsys.readouterr().out


def test_horizon_with_no_model_writes_nothing(cfg, monkeypatch):
    eng, w = _clear_history(cfg, monkeypatch)

    def down(*a, **k):
        raise requests.ConnectionError('no route to host')
    monkeypatch.setattr(eng, 'fetch_prev_runs', down)
    assert horizon.main(eng, write=True) == 1
    assert not os.path.exists(eng.hz_path)


# ---------------------------------------------------------------- calibration.json
@pytest.fixture
def eng(cfg, monkeypatch):
    """Three models, clear-sky forecasts from a stubbed om_get, and the same weather as previous runs for learn."""
    hourly = _hourly(clear_sky_weather(cfg, '2026-10-23', 5 * 24 + 1))
    monkeypatch.setattr(E, 'om_get', lambda url, params, **k: FakeResp(200, {'hourly': hourly}))
    cfg.models = cfg.models[:3]
    cfg.history_days = 30
    e = Engine(cfg, ha=FakeHA())
    monkeypatch.setattr(e, 'now', lambda: NOW)
    e.hist = clear_sky_weather(cfg, (NOW - pd.Timedelta(days=31)).floor('D'), 32 * 24)
    monkeypatch.setattr(e, 'fetch_prev_runs', lambda m, lead, s, end: e.hist)
    return e


def _with_history(eng):
    """Measured production = 90 % of the forecast, so learn() can fit."""
    F = eng.pv(eng.hist)
    eng._ha = FakeHA({x.entities[0]: F[x.name] * 900 for x in eng.cfg.arrays})


def test_run_with_cut_short_calibration_learns_again(eng):
    _with_history(eng)
    _write(eng.cal_path, '{"generated": "2026-10-23T01:30:00+00:00", "models": {"icon_sea')
    assert eng.run() == 0
    with open(eng.cal_path) as fh:                           # rewritten by the learn the run started
        assert len(json.load(fh)['models']) == 2 * len(eng.cfg.models)
    assert eng.ha.states['sensor.pv_forecast_today'][1]['models_ok'] == 3


@pytest.mark.parametrize('text', ['[]', '{"models": {}, "quantiles": {}}', '{"models": [], "quantiles": {"p10": 1, "p25": 1, "p50": 1, "p90": 1}}'])
def test_run_with_unusable_calibration_and_failed_learn_publishes(eng, text):
    _write(eng.cal_path, text)                               # learn fails here (FakeHA has no history)
    assert eng.run() == 0
    at = eng.ha.states['sensor.pv_forecast_today'][1]
    assert at['quantiles_fallback'] is True and at['models_ok'] == 3


def test_unreadable_calibration_file_still_raises(eng, monkeypatch):
    """A permission or I/O error says nothing about the content: no relearn over a file that may be fine."""
    _write(eng.cal_path, '{}')

    def denied(path, *a, **k):
        raise PermissionError(13, 'Permission denied', path)
    monkeypatch.setattr(E, 'open', denied, raising=False)
    with pytest.raises(PermissionError):
        eng.run()
    assert eng.ha.states == {}


def test_learn_replaces_a_cut_short_calibration(eng):
    _with_history(eng)
    _write(eng.cal_path, '{"generated": "2026-10-23T01:30:00+00:00", "mo')
    assert eng.learn() == 0
    with open(eng.cal_path) as fh:
        assert len(json.load(fh)['models']) == 2 * len(eng.cfg.models)
    assert glob.glob(eng.cal_path + '*.tmp') == []


def test_crash_while_writing_keeps_the_old_calibration(eng, monkeypatch):
    _with_history(eng)
    old = json.dumps({'generated': 'old', 'n_hours': 1, 'models': {}, 'quantiles': {'p10': 0.6, 'p25': 0.8, 'p50': 1.0, 'p90': 1.2}})
    _write(eng.cal_path, old)
    written = []

    def cut(obj, fh, **k):
        written.append(fh.name)
        fh.write('{"generated": ')
        raise OSError('No space left on device')
    monkeypatch.setattr(E.json, 'dump', cut)
    with pytest.raises(OSError, match='No space left on device'):
        eng.learn()
    assert written and written[0] != eng.cal_path            # the crash happened while writing, to another file
    with open(eng.cal_path) as fh:
        assert fh.read() == old
    assert glob.glob(eng.cal_path + '*.tmp') == []


# ---------------------------------------------------------------- level factor
def _archive_day_ahead(eng, models, days=14, pv=1.1):
    """Each model said `pv` kWh for 08-16 local on each of the last `days` days, issued 20:00 local the evening before."""
    c = eng.db()
    hours, rows = [], []
    for i in range(1, days + 1):
        day = NOW.tz_convert(eng.tz).normalize() - pd.DateOffset(days=i)
        issue = int((day - pd.Timedelta(hours=4)).timestamp())
        for h in range(8, 17):
            t = (day + pd.Timedelta(hours=h)).tz_convert('UTC')
            hours.append(t)
            rows += [(issue, m, int(t.timestamp()), 0, 0, 0, 0, 0, 0, 0, pv) for m in models]
    c.executemany('insert or replace into nwp values(?,?,?,?,?,?,?,?,?,?,?)', rows)
    c.commit()
    return c, pd.DataFrame({'total': 1.0, 'ok': True}, index=pd.DatetimeIndex(sorted(hours)))


def test_level_counts_archived_models_not_configured_ones(eng):
    """Three models configured, one never answers at the site: the two that do still give a level factor."""
    c, a = _archive_day_ahead(eng, eng.cfg.models[:2])
    t = eng.level_day_ahead(c, a)
    assert len(t) == 14
    assert E.level_factor(t.y, t.f, [(NOW.tz_convert(eng.tz).date() - d).days for d in t.index])[1] == pytest.approx(1 / 1.1)


def test_level_still_skips_days_with_too_few_models(eng):
    """A model that is archived but missing on some days: those days still need every archived model (up to 4)."""
    _archive_day_ahead(eng, eng.cfg.models[:3], days=7)
    c, a = _archive_day_ahead(eng, eng.cfg.models[:2], days=14)      # days 8-14: only two of the three
    assert len(eng.level_day_ahead(c, a)) == 7


def test_level_with_eight_models_still_needs_four(eng):
    """The default list, every model answering: a day with only three archived models is still skipped."""
    eng.cfg.models = list(DEFAULT_MODELS)
    _archive_day_ahead(eng, eng.cfg.models, days=7)
    c, a = _archive_day_ahead(eng, eng.cfg.models[:3], days=14)
    assert len(eng.level_day_ahead(c, a)) == 7
