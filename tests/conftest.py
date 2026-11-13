import os
import socket

import pandas as pd
import pvlib
import pytest

from roofcast.config import load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Every test runs offline: any socket connection fails loudly."""
    def guard(*a, **k):
        raise RuntimeError('network access attempted in a test')
    monkeypatch.setattr(socket.socket, 'connect', guard)
    monkeypatch.setattr(socket, 'create_connection', guard)


@pytest.fixture
def cfg(tmp_path):
    c = load(os.path.join(ROOT, 'roofcast.example.yaml'))
    c.data_dir = str(tmp_path / 'data')
    return c


def clear_sky_weather(cfg, start, hours):
    """Synthetic hourly weather (indexed by the start of the hour, like om_frame output) from pvlib clear sky."""
    idx = pd.date_range(start, periods=hours, freq='h', tz='UTC')
    loc = pvlib.location.Location(cfg.latitude, cfg.longitude, 'UTC', cfg.altitude)
    cs = loc.get_clearsky(idx + pd.Timedelta(minutes=30), model='ineichen')
    cs.index = idx
    return pd.DataFrame({'shortwave_radiation': cs.ghi, 'direct_normal_irradiance': cs.dni, 'diffuse_radiation': cs.dhi,
                         'temperature_2m': 15.0, 'cloud_cover': 0.0, 'wind_speed_10m': 10.0}, index=idx)


class FakeHA:
    def __init__(self, stats=None):
        self.states = {}
        self.stats = stats

    def set_state(self, entity, state, attrs):
        self.states[entity] = (state, attrs)

    def get_state(self, entity):
        return None

    def statistics(self, ids, start):
        if self.stats is None:
            raise RuntimeError('no history')
        return self.stats
