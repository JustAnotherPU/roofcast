import copy

import pandas as pd
import pytest
import yaml

from roofcast.config import ConfigError, parse

from conftest import ROOT

BASE = yaml.safe_load(open(f'{ROOT}/config.example.yaml'))


def mk(**over):
    d = copy.deepcopy(BASE)
    d.update(over)
    return d


def test_example_config_is_valid():
    c = parse(mk())
    assert [a.name for a in c.arrays] == ['south', 'west']
    assert c.total_kwp == pytest.approx(10.0)
    assert c.pv_cap_kw == pytest.approx(10.0)          # default = nameplate total
    assert c.sensor_today == 'sensor.pv_forecast_today'
    assert c.knmi_enabled is False and c.soc_entity is None


@pytest.mark.parametrize('over, msg', [
    ({'arrays': []}, 'at least one'),
    ({'power_unit': 'mW'}, 'power_unit'),
    ({'location': {'latitude': 52.37, 'longitude': 4.90, 'timezone': 'Mars/Base'}}, 'timezone'),
    ({'location': {'latitude': 120, 'longitude': 4.90, 'timezone': 'Europe/Amsterdam'}}, 'latitude'),
    ({'inverter_ac_limit_kw': None}, 'inverter_ac_limit_kw'),
])
def test_bad_values_are_rejected(over, msg):
    with pytest.raises(ConfigError, match=msg):
        parse(mk(**over))


def test_array_checks():
    with pytest.raises(ConfigError, match='unique'):
        parse(mk(arrays=[BASE['arrays'][0], BASE['arrays'][0]]))
    bad = dict(BASE['arrays'][0], actual_entity='light.kitchen')
    with pytest.raises(ConfigError, match='actual_entity'):
        parse(mk(arrays=[bad]))
    with pytest.raises(ConfigError, match='azimuth'):
        parse(mk(arrays=[dict(BASE['arrays'][0], azimuth=400)]))


def test_faces_and_since():
    arr = {'name': 'ew', 'faces': [{'azimuth': 90, 'tilt': 15, 'kwp': 2}, {'azimuth': 270, 'tilt': 15, 'kwp': 2.5}],
           'actual_entity': ['sensor.a', 'sensor.b'], 'since': '2026-06-01'}
    c = parse(mk(arrays=[arr]))
    a = c.arrays[0]
    assert a.kwp == pytest.approx(4.5) and a.entities == ['sensor.a', 'sensor.b']
    assert a.since == pd.Timestamp('2026-05-31T22:00Z')     # local midnight (CEST) stored in UTC


def test_token_comes_from_env(monkeypatch):
    c = parse(mk())
    monkeypatch.delenv('HA_TOKEN', raising=False)
    with pytest.raises(ConfigError, match='HA_TOKEN'):
        c.ha_token()
    monkeypatch.setenv('HA_TOKEN', 'x')
    assert c.ha_token() == 'x'


def test_curtailment_needs_both_entities():
    with pytest.raises(ConfigError, match='both'):
        parse(mk(curtailment={'battery_soc_entity': 'sensor.battery_soc'}))
    with pytest.raises(ConfigError, match='both'):
        parse(mk(curtailment={'grid_power_entity': 'sensor.grid_power'}))
    c = parse(mk(curtailment={'battery_soc_entity': 'sensor.battery_soc', 'grid_power_entity': 'sensor.grid_power'}))
    assert c.soc_entity == 'sensor.battery_soc' and c.grid_entity == 'sensor.grid_power'
