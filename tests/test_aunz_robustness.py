"""Failure handling that shows up outside Europe: models without data at the site, short model lists, bad NWP values."""
import pytest
import requests

from roofcast import engine as E

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
