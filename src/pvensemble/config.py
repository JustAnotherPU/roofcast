"""Load and validate config.yaml. Secrets never live in the file: they come from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

DEFAULT_MODELS = [
    'icon_seamless', 'ecmwf_ifs025', 'ecmwf_aifs025_single', 'meteofrance_seamless',
    'knmi_harmonie_arome_europe', 'ukmo_seamless', 'dmi_seamless', 'gfs_seamless',
]
POWER_UNITS = {'W': 0.001, 'kW': 1.0}


class ConfigError(ValueError):
    pass


@dataclass
class Face:
    azimuth: float
    tilt: float
    kwp: float


@dataclass
class Array:
    name: str
    faces: list[Face]
    entities: list[str]
    since: pd.Timestamp | None = None

    @property
    def kwp(self) -> float:
        return sum(f.kwp for f in self.faces)


@dataclass
class Config:
    latitude: float
    longitude: float
    altitude: float
    timezone: str
    arrays: list[Array]
    ac_limit_kw: float
    pv_cap_kw: float
    power_unit: str = 'W'
    history_days: int = 150
    models: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    data_dir: str = './data'
    ha_url: str = 'http://homeassistant.local:8123'
    ha_token_env: str = 'HA_TOKEN'
    sensor_today: str = 'sensor.pv_forecast_today'
    sensor_tomorrow: str = 'sensor.pv_forecast_tomorrow'
    soc_entity: str | None = None
    grid_entity: str | None = None
    soc_full_pct: float = 97.0
    export_threshold_w: float = -300.0
    solcast_entities: list[str] = field(default_factory=list)
    open_meteo_entities: list[str] = field(default_factory=list)
    start_efficiency: float = 0.85
    new_array_min_hours: int = 6
    new_array_level_only_hours: int = 150
    horizon_mask: bool = True
    knmi_enabled: bool = False
    knmi_key_env: str = 'KNMI_API_KEY'
    knmi_max_age_s: int = 3600

    @property
    def total_kwp(self) -> float:
        return sum(a.kwp for a in self.arrays)

    @property
    def power_factor(self) -> float:
        return POWER_UNITS[self.power_unit]

    def path(self, name: str) -> str:
        os.makedirs(self.data_dir, exist_ok=True)
        return os.path.join(self.data_dir, name)

    def ha_token(self) -> str:
        tok = os.environ.get(self.ha_token_env, '').strip()
        if not tok:
            raise ConfigError(f'environment variable {self.ha_token_env} is empty (Home Assistant long-lived token)')
        return tok

    def knmi_key(self) -> str:
        key = os.environ.get(self.knmi_key_env, '').strip()
        if not key:
            raise ConfigError(f'environment variable {self.knmi_key_env} is empty (KNMI Data Platform API key)')
        return key


def _num(d, key, where, lo=None, hi=None, default=None):
    v = d.get(key, default)
    if v is None:
        raise ConfigError(f'{where}: "{key}" is required')
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise ConfigError(f'{where}: "{key}" must be a number, got {v!r}') from None
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise ConfigError(f'{where}: "{key}" = {v} is outside [{lo}, {hi}]')
    return v


def _face(d, where):
    return Face(azimuth=_num(d, 'azimuth', where, 0, 360), tilt=_num(d, 'tilt', where, 0, 90), kwp=_num(d, 'kwp', where, 0.01, 1000))


def _array(d, i):
    where = f'arrays[{i}]'
    if not isinstance(d, dict) or not d.get('name'):
        raise ConfigError(f'{where}: needs a "name"')
    where = f'array "{d["name"]}"'
    faces = [_face(f, f'{where} face {j}') for j, f in enumerate(d['faces'])] if d.get('faces') else [_face(d, where)]
    ents = d.get('actual_entity')
    ents = [ents] if isinstance(ents, str) else list(ents or [])
    if not ents or not all(isinstance(e, str) and e.startswith('sensor.') for e in ents):
        raise ConfigError(f'{where}: "actual_entity" must be one sensor.* id or a list of them')
    since = None
    if d.get('since') is not None:
        try:
            since = pd.Timestamp(str(d['since']))
        except ValueError:
            raise ConfigError(f'{where}: "since" is not a date: {d["since"]!r}') from None
    return Array(name=str(d['name']), faces=faces, entities=ents, since=since)


def parse(raw: dict) -> Config:
    if not isinstance(raw, dict):
        raise ConfigError('config must be a mapping')
    loc = raw.get('location') or {}
    tz = str(loc.get('timezone', ''))
    try:
        ZoneInfo(tz)
    except Exception:
        raise ConfigError(f'location.timezone {tz!r} is not an IANA zone (e.g. Europe/Amsterdam)') from None
    arrays = [_array(a, i) for i, a in enumerate(raw.get('arrays') or [])]
    if not arrays:
        raise ConfigError('at least one entry under "arrays" is required')
    names = [a.name for a in arrays]
    if len(set(names)) != len(names):
        raise ConfigError(f'array names must be unique: {names}')
    for a in arrays:
        if a.since is not None:  # "since" is a local date or datetime; store it in UTC
            a.since = (a.since.tz_localize(tz) if a.since.tzinfo is None else a.since).tz_convert('UTC')
    total = sum(a.kwp for a in arrays)
    unit = str(raw.get('power_unit', 'W'))
    if unit not in POWER_UNITS:
        raise ConfigError(f'power_unit must be one of {list(POWER_UNITS)}, got {unit!r}')
    ha = raw.get('home_assistant') or {}
    pub = raw.get('publish') or {}
    cur = raw.get('curtailment') or {}
    cmp_ = raw.get('compare') or {}
    new = raw.get('new_array') or {}
    knmi = raw.get('knmi') or {}
    if bool(cur.get('battery_soc_entity')) != bool(cur.get('grid_power_entity')):
        raise ConfigError('curtailment needs both battery_soc_entity and grid_power_entity (or neither)')
    models = raw.get('models') or list(DEFAULT_MODELS)
    if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
        raise ConfigError('models must be a list of Open-Meteo model names')
    return Config(
        latitude=_num(loc, 'latitude', 'location', -90, 90),
        longitude=_num(loc, 'longitude', 'location', -180, 180),
        altitude=_num(loc, 'altitude', 'location', -500, 9000, default=0),
        timezone=tz,
        arrays=arrays,
        ac_limit_kw=_num(raw, 'inverter_ac_limit_kw', 'config', 0.1),
        # default cap: a measured hour above the nameplate total is a sensor glitch
        pv_cap_kw=_num(raw, 'pv_cap_kw', 'config', 0.1, default=total),
        power_unit=unit,
        history_days=int(_num(raw, 'history_days', 'config', 14, 1000, default=150)),
        models=models,
        data_dir=str(raw.get('data_dir', './data')),
        ha_url=str(ha.get('url', 'http://homeassistant.local:8123')).rstrip('/'),
        ha_token_env=str(ha.get('token_env', 'HA_TOKEN')),
        sensor_today=str(pub.get('today', 'sensor.pv_forecast_today')),
        sensor_tomorrow=str(pub.get('tomorrow', 'sensor.pv_forecast_tomorrow')),
        soc_entity=cur.get('battery_soc_entity'),
        grid_entity=cur.get('grid_power_entity'),
        soc_full_pct=_num(cur, 'soc_full_pct', 'curtailment', 0, 100, default=97),
        export_threshold_w=_num(cur, 'export_threshold_w', 'curtailment', default=-300),
        solcast_entities=list(cmp_.get('solcast_entities') or []),
        open_meteo_entities=list(cmp_.get('open_meteo_entities') or []),
        start_efficiency=_num(new, 'start_efficiency', 'new_array', 0.1, 1.5, default=0.85),
        new_array_min_hours=int(_num(new, 'min_hours', 'new_array', 0, default=6)),
        new_array_level_only_hours=int(_num(new, 'level_only_hours', 'new_array', 0, default=150)),
        horizon_mask=bool(raw.get('horizon_mask', True)),
        knmi_enabled=bool(knmi.get('enabled', False)),
        knmi_key_env=str(knmi.get('api_key_env', 'KNMI_API_KEY')),
        knmi_max_age_s=int(_num(knmi, 'max_age_s', 'knmi', 600, default=3600)),
    )


def load(path: str) -> Config:
    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigError(f'config file not found: {path}') from None
    return parse(raw)
