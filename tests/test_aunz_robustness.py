"""Failure handling that shows up outside Europe: models without data at the site, short model lists, bad NWP values."""
import json
import os

import pandas as pd
import pytest
import requests

from roofcast import engine as E
from roofcast import horizon
from roofcast.engine import Engine

from conftest import FakeHA, clear_sky_weather

URL = 'https://api.open-meteo.com/v1/forecast'
PARAMS = {'latitude': -33.87, 'longitude': 151.21, 'models': 'knmi_harmonie_arome_europe'}


class FakeResp:
    def __init__(self, status, body=None, reason=''):
        self.status_code, self.body, self.reason = status, body, reason
        self.ok = status < 400

    def json(self):
        if self.body is None:
            raise ValueError('no JSON')
        return self.body


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
