"""Multi-model PV forecast, calibrated on the system's own production.

Pipeline:
  NWP models (Open-Meteo)  ->  pvlib per array (Perez transposition, IAM, Faiman cell temperature, horizon mask)
  ->  per-array calibration learned every night (level + azimuth x elevation shape, per model and per lead day)
  ->  ensemble mean  ->  rolling level factor k  ->  P10/P25/P50/P90  ->  Home Assistant sensors (Solcast format).
Every forecast that is produced is archived in SQLite, so accuracy can be measured later on what was really published.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import time

import numpy as np
import pandas as pd
import pvlib
import requests

from .config import Array, Config

VARS = ['shortwave_radiation', 'direct_normal_irradiance', 'diffuse_radiation', 'temperature_2m', 'cloud_cover', 'wind_speed_10m']
FORECAST_URL = 'https://api.open-meteo.com/v1/forecast'
PREVIOUS_RUNS_URL = 'https://previous-runs-api.open-meteo.com/v1/forecast'
AZ_BIN, EL_BIN = 15, 6                       # shape bins: 15 deg of solar azimuth x 6 deg of elevation
# Rolling level factor: k = sum(actual) / sum(day-ahead ensemble), last 28 days, half-life 14 days, bounded.
LEVEL_DAYS, LEVEL_HALF, LEVEL_MIN, LEVEL_BOUNDS = 28, 14.0, 10, (0.85, 1.10)
# Used until enough clear days exist to learn the spread (cold start). P50 stays the ensemble mean.
DEFAULT_QUANTILES = {'p10': 0.55, 'p25': 0.8, 'p50': 1.0, 'p90': 1.25}
MIN_QUANTILE_DAYS = 5
NEW_ARRAY_BOUNDS = (0.1, 1.5)
NWP_KEEP_DAYS = 35                           # > LEVEL_DAYS + 3 days of margin used by level_day_ahead


def log(*a):
    print(dt.datetime.now().strftime('%F %T'), *a, flush=True)


# ---------------------------------------------------------------- date helpers (DST safe)
def local_days(now: pd.Timestamp, tz: str, n: int = 4) -> list[pd.Timestamp]:
    """Local midnights of today, tomorrow, ... as tz-aware timestamps.
    DateOffset(days=1) moves the calendar day; Timedelta(days=1) would add 24 h and land at 23:00 or 01:00 on DST days."""
    today = now.tz_convert(tz).normalize()
    return [today + pd.DateOffset(days=i) for i in range(n)]


def level_factor(y, f, age, bounds=LEVEL_BOUNDS, half=LEVEL_HALF, mn=LEVEL_MIN):
    """(bounded k, raw k) with k = sum(w * actual) / sum(w * forecast), w = 0.5 ** (age / half-life); 1.0 below `mn` days.
    `f` MUST be the raw forecast (before k). Measured on the published forecast, k only sees its own residual and drifts
    back to 1 (see tests/test_level.py)."""
    y, f, age = np.asarray(y, float), np.asarray(f, float), np.asarray(age, float)
    if len(y) < mn or f.sum() <= 0:
        return 1.0, 1.0
    w = 0.5 ** (age / half)
    raw = float((w * y).sum() / (w * f).sum())
    return min(max(raw, bounds[0]), bounds[1]), raw


def quantiles_from_ratios(r: pd.Series) -> dict:
    """Daily actual / forecast ratios -> fixed multipliers. The empirical levels (5/17/50/95 %) differ from the labels;
    they are kept from the original tuning, the reason was not recorded. Re-check band coverage before changing them."""
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < MIN_QUANTILE_DAYS:
        return {**DEFAULT_QUANTILES, 'n_days': int(len(r)), 'fallback': True}
    return {'p10': float(r.quantile(0.05)), 'p25': float(r.quantile(0.17)), 'p50': float(r.quantile(0.5)),
            'p90': float(r.quantile(0.95)), 'n_days': int(len(r)), 'fallback': False}


def detailed_forecast(Q: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, tz: str) -> list[dict]:
    """Hourly kW quantiles -> Solcast-style 30-min list (keys kept as Solcast names so EMHASS and cards read it unchanged)."""
    rows = []
    sel = Q[(Q.index >= start.tz_convert('UTC')) & (Q.index < end.tz_convert('UTC'))]
    for t, r in sel.iterrows():
        for half in (0, 30):
            rows.append({'period_start': (t + pd.Timedelta(minutes=half)).tz_convert(tz).isoformat(),
                         'pv_estimate': round(float(r.p50), 4), 'pv_estimate10': round(float(r.p10), 4),
                         'pv_estimate25': round(float(r.p25), 4), 'pv_estimate90': round(float(r.p90), 4)})
    return rows


def om_frame(hourly: dict, suffix: str = '') -> pd.DataFrame:
    t = pd.to_datetime(hourly['time'], utc=True)
    df = pd.DataFrame({v: hourly.get(v + suffix) if hourly.get(v + suffix) is not None else hourly.get(v) for v in VARS}, index=t).astype(float)
    # Open-Meteo radiation is the mean of the PRECEDING hour: re-index it by the start of that hour.
    df.index = df.index - pd.Timedelta(hours=1)
    return df


def om_get(url: str, params: dict, tries: int = 3, timeout: int = 60):
    """Open-Meteo sometimes answers 429/5xx in bursts: retry with 20 s then 40 s pauses."""
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504) and i < tries - 1:
                raise requests.HTTPError(r.status_code)
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            if i == tries - 1:
                raise
            time.sleep(20 * (i + 1))


def binkey(o: pd.DataFrame) -> pd.Series:
    return ((o.saz // AZ_BIN) * AZ_BIN).astype(int).astype(str) + '_' + ((o.el.clip(lower=0) // EL_BIN) * EL_BIN).astype(int).astype(str)


def hmask(mask: dict | None, saz: pd.Series, el: pd.Series):
    """Share of the direct beam kept (0-1) when the sun is below the learned horizon of its azimuth bin; 1 elsewhere."""
    if not mask or not mask.get('bins'):
        return 1.0
    key = ((saz // mask['step']) * mask['step']).astype(int).astype(str)
    h = key.map({k: v[0] for k, v in mask['bins'].items()}).astype(float).fillna(-90)
    f = key.map({k: v[1] for k, v in mask['bins'].items()}).astype(float).fillna(1.0)
    return f.where(el < h, 1.0)


def load_horizon(path: str) -> dict:
    try:
        hz = json.load(open(path)) if os.path.exists(path) else {}
    except Exception as e:  # never block run/learn on a bad mask file
        log('horizon.json unreadable, no mask:', e)
        hz = {}
    return {k: v for k, v in (hz if isinstance(hz, dict) else {}).items()
            if isinstance(v, dict) and isinstance(v.get('bins'), dict) and v.get('step')}


class Engine:
    def __init__(self, cfg: Config, ha=None):
        self.cfg = cfg
        self.tz = cfg.timezone
        self.loc = pvlib.location.Location(cfg.latitude, cfg.longitude, 'UTC', cfg.altitude)
        self.db_path = cfg.path('pvensemble.db')
        self.cal_path = cfg.path('calibration.json')
        self.hz_path = cfg.path('horizon.json')
        self.cache = cfg.path('cache')
        os.makedirs(self.cache, exist_ok=True)
        self.horizon = load_horizon(self.hz_path) if cfg.horizon_mask else {}
        self._ha = ha
        k = cfg.total_kwp
        # Absolute kWh thresholds of the original code, scaled to the system size.
        self.min_day_kwh = 0.2 * k           # days with a smaller forecast say nothing about the level
        self.dead_hour_kwh = 0.03 * k        # forecast above this while the inverter reports ~0 = outage, not weather

    @property
    def ha(self):
        if self._ha is None:
            from .ha import HomeAssistant
            self._ha = HomeAssistant(self.cfg.ha_url, self.cfg.ha_token())
        return self._ha

    def now(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz='UTC')

    # ------------------------------------------------------------ storage
    def db(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=60)
        c.executescript('''
        create table if not exists nwp(issue integer, provider text, target integer, lead_h real, ghi real, dni real, dhi real,
                                       t2m real, cc real, wind real, pv real, primary key(issue, provider, target));
        create table if not exists out(issue integer, target integer, p10 real, p25 real, p50 real, p90 real, members integer,
                                       primary key(issue, target));
        create table if not exists ext(issue integer, provider text, target integer, kw real, kw10 real, kw90 real,
                                       primary key(issue, provider, target));
        create table if not exists actual(hour integer primary key, total real, soc real, grid real);
        create table if not exists quality(day text, provider text, lead text, fc_kwh real, real_kwh real,
                                           primary key(day, provider, lead));''')
        return c

    # ------------------------------------------------------------ PV physics
    def pv(self, df: pd.DataFrame, horizon: dict | None = None) -> pd.DataFrame:
        """kWh per hour for each array, before calibration. Hours before an array's `since` are 0 (array not installed)."""
        horizon = self.horizon if horizon is None else horizon
        idx = df.index
        mid = idx + pd.Timedelta(minutes=30)
        sp = self.loc.get_solarposition(mid)
        sp.index = idx
        ext = pvlib.irradiance.get_extra_radiation(mid)
        ext.index = idx
        am = pvlib.atmosphere.get_relative_airmass(sp.apparent_zenith)
        out = {}
        for a in self.cfg.arrays:
            # a mask learned on a multi-face array cannot be split per face: only single-face arrays get one
            mask = horizon.get(a.name) if len(a.faces) == 1 else None
            tot = pd.Series(0.0, index=idx)
            for f in a.faces:
                poa = pvlib.irradiance.get_total_irradiance(
                    f.tilt, f.azimuth, sp.apparent_zenith, sp.azimuth,
                    df.direct_normal_irradiance.clip(lower=0), df.shortwave_radiation.clip(lower=0), df.diffuse_radiation.clip(lower=0),
                    dni_extra=ext, model='perez', airmass=am)
                iam = pvlib.iam.physical(pvlib.irradiance.aoi(f.tilt, f.azimuth, sp.apparent_zenith, sp.azimuth)).fillna(0)
                # The horizon mask only cuts the DIRECT beam; diffuse light still reaches a shaded panel.
                eff = poa.poa_direct * iam * hmask(mask, sp.azimuth, 90 - sp.apparent_zenith) + poa.poa_diffuse
                tc = pvlib.temperature.faiman(poa.poa_global, df.temperature_2m, df.wind_speed_10m.fillna(7) / 3.6)
                tot = tot + (f.kwp * eff / 1000 * (1 - 0.0029 * (tc - 25))).clip(lower=0).fillna(0)
            out[a.name] = tot.where(self.active(a, idx), 0.0)
        o = pd.DataFrame(out)
        o['el'] = 90 - sp.apparent_zenith
        o['saz'] = sp.azimuth
        o['has_nwp'] = df.shortwave_radiation.notna()
        return o

    @staticmethod
    def active(a: Array, idx) -> pd.Series:
        return pd.Series(True if a.since is None else idx >= a.since, index=idx)

    def apply_cal(self, F: pd.DataFrame, kb: pd.Series, ent: dict | None, a: Array) -> pd.Series:
        e = (ent or {}).get(a.name)
        if not e:  # no calibration yet (cold start): plain pvlib times the starting efficiency
            return F[a.name] * self.cfg.start_efficiency
        return F[a.name] * kb.map(e['shape']).fillna(1.0) * e['g']

    # ------------------------------------------------------------ measured production
    def actuals(self) -> pd.DataFrame:
        cfg = self.cfg
        start = (self.now() - pd.Timedelta(days=cfg.history_days)).isoformat()
        cols = sorted({e for a in cfg.arrays for e in a.entities})
        extra = [x for x in (cfg.soc_entity, cfg.grid_entity) if x]
        s = self.ha.statistics(cols + extra, start)
        if not any(x in s for x in cols):
            raise ValueError('no long-term statistics for any actual_entity (state_class: measurement missing?)')
        for x in extra:
            if x not in s:
                log('no statistics for', x, '(curtailment filter), every hour is treated as not curtailed')
        a = pd.DataFrame({x: s[x] * cfg.power_factor for x in cols if x in s})
        for x in cols:
            if x not in a:
                log('no statistics for', x, '(state_class: measurement missing?), its hours are excluded')
                a[x] = np.nan
        for arr in cfg.arrays:
            v = sum(a[x] for x in arr.entities)
            a[arr.name] = v.where(self.active(arr, a.index), 0.0)   # 0 before `since`, not NaN: older hours stay usable
        a['soc'] = s.get(cfg.soc_entity) if cfg.soc_entity else np.nan
        a['grid'] = s.get(cfg.grid_entity) if cfg.grid_entity else np.nan
        a = a[[x.name for x in cfg.arrays] + ['soc', 'grid']].copy()
        a['total'] = sum(a[x.name] for x in cfg.arrays)
        ok = pd.Series(True, index=a.index)
        if cfg.soc_entity and cfg.grid_entity:
            # battery full and nothing exported: the inverter may be throttling PV, so these hours do not show the potential
            ok &= ~((a.soc >= cfg.soc_full_pct) & (a.grid > cfg.export_threshold_w))
        loc_day = a.index.tz_convert(self.tz).date
        for arr in cfg.arrays:
            if arr.since is not None:   # the day of commissioning is half old wiring, half new: drop it entirely
                ok &= loc_day != arr.since.tz_convert(self.tz).date()
        a['okb'] = ok                     # per-array base: only that array needs to be measured
        a['ok'] = ok & a.total.notna()    # ok = every array measured
        try:
            c = self.db()
            inv = self.invalid_hours(c, {int(t.timestamp()): v for t, v in a.total.items()})
            c.close()
            if inv:
                m = np.array([int(t.timestamp()) in inv for t in a.index])
                a.loc[m, ['ok', 'okb']] = False
        except Exception as e:
            log('invalid-hours filter failed:', e)
        return a

    def invalid_hours(self, c, total: dict) -> set:
        """Hours with no usable measurement: inverter at ~0 while the ensemble expected real production, or an impossible value.
        `total` maps hour start (unix, UTC) to kWh. Forecast = mean of the models, last issue before the hour.
        Known limit: a snow day (production ~0) is also classed invalid, so it is left out of the quality stats."""
        e = dict(c.execute('''select n.target, avg(n.pv) from nwp n join
            (select target, max(issue) i from nwp where issue <= target group by target) m on m.target = n.target and m.i = n.issue
            group by n.target''').fetchall())
        return {h for h, v in total.items() if v is not None and v == v
                and ((v < 0.02 and (e.get(h) or 0) > self.dead_hour_kwh) or v > self.cfg.pv_cap_kw)}

    # ------------------------------------------------------------ nightly learning
    def fetch_prev_runs(self, model: str, lead: int, start: str, end: str) -> pd.DataFrame:
        f = os.path.join(self.cache, f'{model}_d{lead}_{start}_{end}.json')
        if not os.path.exists(f):
            r = om_get(PREVIOUS_RUNS_URL, dict(latitude=self.cfg.latitude, longitude=self.cfg.longitude,
                                               hourly=','.join(f'{v}_previous_day{lead}' for v in VARS), models=model,
                                               start_date=start, end_date=end, timezone='UTC'), timeout=180)
            with open(f, 'w') as fh:
                json.dump(r.json(), fh)
        with open(f) as fh:
            return om_frame(json.load(fh)['hourly'], f'_previous_day{lead}')

    def fit_model(self, F: pd.DataFrame, a: pd.DataFrame) -> dict:
        """Calibration of one model x lead: {array: {'g': level, 'shape': {bin: factor}}}."""
        cfg = self.cfg
        ok = a.ok & (F.el > 0) & F.has_nwp      # hours with no forecast would add y > 0 with x = 0 and inflate the level
        kb = binkey(F)
        ent = {}
        after = {x.name: self.active(x, a.index) for x in cfg.arrays}
        usable = {x.name: a.okb & a[x.name].notna() & after[x.name] & (F.el > 0) & F.has_nwp & (F[x.name] > 0) for x in cfg.arrays}
        new = {x.name for x in cfg.arrays if x.since is not None and usable[x.name].sum() < cfg.new_array_level_only_hours}
        for x in [x for x in cfg.arrays if x.name not in new]:
            okk = ok & after[x.name]
            y, xs = a[x.name][okk], F[x.name][okk]
            if xs.sum() < 5 * x.kwp:
                raise ValueError(f'empty forecast for {x.name}')        # guard: no calibration from too little data
            g = float(y.sum() / xs.sum())
            if not 0.3 < g < 1.5:
                raise ValueError(f'level {g:.2f} out of bounds for {x.name} (wrong power_unit or entity?)')
            d = pd.DataFrame({'b': kb[okk], 'y': y, 'x': xs}).groupby('b').sum()
            shrink = 0.8 * x.kwp                                         # pseudo-kWh pulling sparse bins back to the level
            shape = ((d.y + shrink * g) / (d.x + shrink)).clip(0, 1.5) / g
            ent[x.name] = {'g': g, 'shape': {k: round(float(v), 4) for k, v in shape.items()}}
        # A freshly added array: level only (no shape) until it has enough hours, starting efficiency before that.
        # Capped at the largest established array's level (kept from the original after an audit of this case).
        main = max((x for x in cfg.arrays if x.name in ent), key=lambda x: x.kwp, default=None)
        hi = min(ent[main.name]['g'], NEW_ARRAY_BOUNDS[1]) if main else NEW_ARRAY_BOUNDS[1]
        for x in [x for x in cfg.arrays if x.name in new]:
            m = usable[x.name]
            n = int(m.sum())
            xs = F[x.name][m].sum()
            if n < cfg.new_array_min_hours or xs <= 0:
                g = cfg.start_efficiency
            else:
                g = float(a[x.name][m].sum() / xs)
                if not NEW_ARRAY_BOUNDS[0] <= g <= hi:   # clamp, never reset: a dead new inverter must show up as low
                    log('learn', x.name, f'level {g:.2f} clamped')
                    g = min(max(g, NEW_ARRAY_BOUNDS[0]), hi)
            ent[x.name] = {'g': g, 'shape': {}, 'hours': n}
        return ent

    def learn(self) -> int:
        cfg = self.cfg
        raw = self.actuals()
        today = self.now().tz_convert(self.tz).normalize()
        end = (today - pd.DateOffset(days=1)).date().isoformat()
        start = (today - pd.DateOffset(days=cfg.history_days)).date().isoformat()
        # only hours that have an archived forecast: learning in the daytime would otherwise add y > 0 with x = 0
        stop = min(today.tz_convert('UTC'), pd.Timestamp(end, tz='UTC') + pd.Timedelta(days=1))
        a = raw[(raw.index < stop) & (raw.index >= pd.Timestamp(start, tz='UTC'))]
        for f in os.listdir(self.cache):     # yesterday's cache is replaced
            if end not in f:
                os.remove(os.path.join(self.cache, f))
        prev = self.load_cal() if os.path.exists(self.cal_path) else {}
        cal = {'generated': self.now().isoformat(), 'n_hours': int(a.ok.sum()), 'models': {}}
        days = np.array(a.index.tz_convert(self.tz).date)
        ens = {}
        for m in cfg.models:
            for lead in (0, 1):
                try:
                    F = self.pv(self.fetch_prev_runs(m, lead, start, end).reindex(a.index).interpolate(limit=2))
                    ent = self.fit_model(F, a)
                except Exception as e:
                    log('learn', m, lead, 'failed:', e)
                    if f'{m}|{lead}' in (prev.get('models') or {}):     # keep last night's entry rather than none
                        cal['models'][f'{m}|{lead}'] = prev['models'][f'{m}|{lead}']
                    continue
                cal['models'][f'{m}|{lead}'] = ent
                if lead == 1:
                    kb = binkey(F)
                    ens[m] = sum(self.apply_cal(F, kb, ent, x) for x in cfg.arrays).clip(upper=cfg.ac_limit_kw)
        if not ens:
            # nothing fitted tonight (API down, wrong unit, no history): never write an empty or stale-only
            # calibration, and fail so systemd retries
            log('learn: no model could be fitted, calibration.json not written')
            return 1
        # quantiles: daily actual / day-ahead ensemble ratio
        if ens:
            E = pd.DataFrame(ens).mean(axis=1)
            ok = a.ok.reindex(E.index).fillna(False).astype(bool)
            dd = pd.DataFrame({'f': E[ok].groupby(days[ok.values]).sum(), 'y': a.total[ok].groupby(days[ok.values]).sum()})
            dd = dd[dd.f > self.min_day_kwh]
            r = dd.y / dd.f
        else:
            r = pd.Series(dtype=float)
        cal['quantiles'] = quantiles_from_ratios(r)
        c = self.db()
        # Level k is measured OUT OF SAMPLE on the archived raw day-ahead members (table nwp), not on the ensemble
        # recomputed with tonight's calibration: that one fits the past by construction and saw ~0.97 while the
        # published day-ahead was still biased by +8 to +17 % during an autumn drift.
        try:
            t = self.level_day_ahead(c, a)
        except Exception as e:
            log('level factor failed, k = 1:', e)
            t = pd.DataFrame(columns=['y', 'f'])
        k, raw_k = level_factor(t.y, t.f, [(today.date() - d).days for d in t.index])
        cal['level'] = {'k': round(k, 4), 'raw': round(raw_k, 4), 'days': int(len(t)), 'half_life_days': LEVEL_HALF,
                        'source': 'archived day-ahead members (out of sample)'}
        with open(self.cal_path, 'w') as fh:
            json.dump(cal, fh)
        log('calibration:', len(cal['models']), 'model x lead,', cal['quantiles'], 'k =', cal['level']['k'])
        c.executemany('insert or replace into actual(hour, total, soc, grid) values(?,?,?,?)',
                      [(int(t.timestamp()), _f(r.total), _f(r.soc), _f(r.grid)) for t, r in raw.iterrows()])
        # Keep the archive bounded. k reads issues from the last LEVEL_DAYS + 3 days; invalid_hours only needs the
        # short-lead rows (last issue before each target); quality reads table out, not nwp.
        cut = int((self.now() - pd.Timedelta(days=NWP_KEEP_DAYS)).timestamp())
        c.execute('delete from nwp where issue < ? and lead_h > 1', (cut,))
        c.commit()
        self.quality(c)
        c.close()
        return 0

    def level_day_ahead(self, c, a: pd.DataFrame, days: int = LEVEL_DAYS) -> pd.DataFrame:
        """Per local day: y = actual, f = mean of the raw archived members (before k) of the last issue before 21:00 the
        day before (at most 6 h older, at least min(4, n_models) models), on ok hours only."""
        ok = a[a.ok]
        loc = ok.index.tz_convert(self.tz)
        today = self.now().tz_convert(self.tz).date()
        need = min(4, len(self.cfg.models))
        iss = pd.read_sql('select distinct issue from nwp where issue >= ?', c,
                          params=(int((self.now() - pd.Timedelta(days=days + 3)).timestamp()),)).issue
        rows = []
        for d, yh in ok.total.groupby(loc.normalize()):
            cut = (d - pd.Timedelta(hours=3)).timestamp()
            I = iss[iss <= cut]
            if d.date() >= today or I.empty or cut - I.max() > 6 * 3600:
                continue
            n = pd.read_sql(f"select provider, pv from nwp where issue = ? and target in ({','.join('?' * len(yh))})", c,
                            params=(int(I.max()), *[int(t.timestamp()) for t in yh.index]))
            if n.provider.nunique() < need:
                continue
            rows.append((d.date(), float(yh.sum()), float(n.groupby('provider').pv.sum().mean())))
        t = pd.DataFrame(rows, columns=['day', 'y', 'f']).set_index('day')
        return t[t.f > self.min_day_kwh].tail(days)

    # ------------------------------------------------------------ quality tracking
    def quality(self, c) -> None:
        """Day-ahead (last issue before 21:00 the day before) forecast vs actual, per provider."""
        act = pd.read_sql('select hour, total from actual', c)
        inv = self.invalid_hours(c, dict(zip(act.hour, act.total)))
        act = act[~act.hour.isin(inv)].dropna()                     # the same hours are removed from actual AND forecasts
        act['day'] = pd.to_datetime(act.hour, unit='s', utc=True).dt.tz_convert(self.tz).dt.date.astype(str)
        real = act.groupby('day').total.sum()
        today = self.now().tz_convert(self.tz).date().isoformat()
        srcs = [('ensemble', 'select issue, target, p50 kw from out')]
        srcs += [(p, f"select issue, target, kw from ext where provider = '{p}'") for p in ('solcast', 'open_meteo')]
        for prov, sql in srcs:
            f = pd.read_sql(sql, c)
            f = f[~(f.target // 3600 * 3600).isin(inv)]
            if f.empty:
                continue
            f['day'] = pd.to_datetime(f.target, unit='s', utc=True).dt.tz_convert(self.tz).dt.date.astype(str)
            f['iss'] = pd.to_datetime(f.issue, unit='s', utc=True).dt.tz_convert(self.tz)
            for day in f.day.unique():
                if day not in real.index or day >= today:
                    continue
                cut = pd.Timestamp(day, tz=self.tz) - pd.Timedelta(hours=3)
                g = f[(f.day == day) & (f.iss <= cut)]
                if g.empty:
                    continue
                g = g[g.issue == g.issue.max()]
                ts = sorted(g.target.unique())
                # real step of each source (Open-Meteo integration 15 min, Solcast 30 min, this engine 1 h)
                step = float(pd.Series(ts).diff().median()) / 3600 if len(ts) > 2 else 1.0
                c.execute('insert or replace into quality values(?,?,?,?,?)', (day, prov, 'day_ahead', float(g.kw.sum() * step), float(real[day])))
        c.commit()

    @staticmethod
    def quality_summary(c) -> dict:
        q = pd.read_sql("select * from quality where lead = 'day_ahead'", c)
        out = {}
        for prov, g in q.groupby('provider'):
            g = g.sort_values('day')
            for n in (7, 30):
                h = g.tail(n)
                e = h.fc_kwh - h.real_kwh
                out[f'{prov}_wmape_{n}d'] = round(float(e.abs().sum() / max(h.real_kwh.sum(), 1) * 100), 1)
                out[f'{prov}_bias_{n}d'] = round(float(e.sum() / max(h.real_kwh.sum(), 1) * 100), 1)
            out[f'{prov}_days'] = int(len(g))
        return out

    # ------------------------------------------------------------ forecast run (every 15 min)
    def load_cal(self) -> dict:
        with open(self.cal_path) as fh:
            return json.load(fh)

    def run(self) -> int:
        cfg = self.cfg
        if not os.path.exists(self.cal_path):
            try:
                self.learn()
            except Exception as e:
                log('first learn failed, publishing uncalibrated:', e)
        cal = self.load_cal() if os.path.exists(self.cal_path) else {'models': {}, 'quantiles': quantiles_from_ratios(pd.Series(dtype=float)),
                                                                     'generated': None, 'n_hours': 0}
        now = self.now()
        issue = int(now.timestamp())
        d0, d1, d2, d3 = local_days(now, self.tz)
        c = self.db()
        members, status, per_array = {}, {}, {x.name: {} for x in cfg.arrays}
        calibrated = bool(cal['models'])
        for m in cfg.models:
            if calibrated and not any(f'{m}|{lead}' in cal['models'] for lead in (0, 1)):
                # mixing an uncalibrated member into a calibrated ensemble biases it; cold start (no models) is the exception
                status[m] = 'no calibration'
                continue
            try:
                r = om_get(FORECAST_URL, dict(latitude=cfg.latitude, longitude=cfg.longitude, hourly=','.join(VARS), models=m,
                                              past_days=1, forecast_days=4, timezone='UTC'))
                df = om_frame(r.json()['hourly']).dropna(subset=['shortwave_radiation'])
                if df.empty:
                    raise ValueError('empty response')
                F = self.pv(df)
                kb = binkey(F)
                lead0 = F.index.tz_convert(self.tz).normalize() <= d0
                tot = pd.Series(0.0, index=F.index)
                for x in cfg.arrays:
                    p = pd.Series(0.0, index=F.index)
                    for lead, sel in ((0, lead0), (1, ~lead0)):
                        ent = cal['models'].get(f'{m}|{lead}') or cal['models'].get(f'{m}|{1 - lead}')
                        p[sel] = self.apply_cal(F, kb, ent, x)[sel]
                    per_array[x.name][m] = p
                    tot = tot + p
                members[m] = tot.clip(upper=cfg.ac_limit_kw)
                status[m] = 'ok'
                # archived RAW: before k and before the AC clip, so level k can be measured without feedback
                c.executemany('insert or replace into nwp values(?,?,?,?,?,?,?,?,?,?,?)',
                              [(issue, m, int(t.timestamp()), (t - now).total_seconds() / 3600, _f(x.shortwave_radiation),
                                _f(x.direct_normal_irradiance), _f(x.diffuse_radiation), _f(x.temperature_2m), _f(x.cloud_cover),
                                _f(x.wind_speed_10m), float(tot[t])) for t, x in df.iterrows() if t >= now - pd.Timedelta(hours=2)])
            except Exception as e:
                status[m] = f'failed: {str(e)[:60]}'
                log(m, 'failed:', e)
        if not members:
            log('no model available, nothing published')
            c.close()
            return 1
        K = float((cal.get('level') or {}).get('k', 1.0))
        if not LEVEL_BOUNDS[0] <= K <= LEVEL_BOUNDS[1]:
            K = 1.0                                                   # corrupt calibration file: no correction
        M = pd.DataFrame(members)
        p50m = (M.mean(axis=1, skipna=True) * K).clip(upper=cfg.ac_limit_kw)
        n = M.notna().sum(axis=1)
        q = cal['quantiles']
        base = p50m / max(q['p50'], 0.5)
        Q = pd.DataFrame({k: (base * q[k]).clip(upper=cfg.ac_limit_kw) for k in ('p10', 'p25', 'p50', 'p90')})
        fut = Q[Q.index >= now.floor('h') - pd.Timedelta(hours=1)]
        c.executemany('insert or replace into out values(?,?,?,?,?,?,?)',
                      [(issue, int(t.timestamp()), r.p10, r.p25, r.p50, r.p90, int(n[t])) for t, r in fut.iterrows()])
        self.snapshot_others(c, issue)
        c.commit()
        knmi = {'enabled': cfg.knmi_enabled, 'hours_corrected': 0}
        if cfg.knmi_enabled:      # applied to the PUBLISHED curve only; table out stays raw, so no feedback loop
            try:
                knmi.update(self.apply_knmi(c, Q, issue, now))
            except Exception as e:
                log('KNMI correction failed, curve left as is:', e)
        detailed = detailed_forecast(Q, d0, d3, self.tz)
        per = self.per_array_split(per_array, Q, d0, d2)
        qs = self.quality_summary(c)
        c.close()
        loc = Q.tz_convert(self.tz)
        day = loc.index.normalize()

        def dsum(d, k):
            return round(float(loc[k][day == d].sum()), 2)

        common = {'unit_of_measurement': 'kWh', 'device_class': 'energy', 'icon': 'mdi:solar-power-variant',
                  'generated_at': now.isoformat(), 'models_ok': sum(1 for v in status.values() if v == 'ok'), 'models': status,
                  'calibration': cal.get('generated'), 'calibration_hours': cal.get('n_hours', 0),
                  'level_factor': cal.get('level') or {'k': 1.0}, 'quantiles_fallback': bool(q.get('fallback')), 'knmi': knmi}
        self.ha.set_state(cfg.sensor_today, dsum(d0, 'p50'), {
            **common, 'friendly_name': 'PV forecast today (P50)', 'p10': dsum(d0, 'p10'), 'p25': dsum(d0, 'p25'),
            'p90': dsum(d0, 'p90'), 'detailedForecast': detailed, **({'per_array': per} if per else {}), **qs})
        self.ha.set_state(cfg.sensor_tomorrow, dsum(d1, 'p50'), {
            **common, 'friendly_name': 'PV forecast tomorrow (P50)', 'p10': dsum(d1, 'p10'), 'p25': dsum(d1, 'p25'),
            'p90': dsum(d1, 'p90'), 'day_after_tomorrow_p50': dsum(d2, 'p50')})
        log('published: today', dsum(d0, 'p50'), 'kWh, tomorrow', dsum(d1, 'p50'), 'kWh,', common['models_ok'], 'models')
        return 0

    def per_array_split(self, per_array: dict, Q: pd.DataFrame, d0, d2) -> dict | None:
        """Hourly P50 per array = published P50 x the array's share of the ensemble, so the parts add up to the total
        (AC clipping and KNMI included)."""
        try:
            ms = pd.DataFrame({s: pd.DataFrame(v).mean(axis=1) for s, v in per_array.items() if v})
            sh = ms.div(ms.sum(axis=1), axis=0).fillna(0)
            hh = pd.date_range(d0.tz_convert('UTC'), d2.tz_convert('UTC'), freq='h', inclusive='left')
            p50 = Q.p50.reindex(hh).fillna(0)
            shh = sh.reindex(hh).fillna(0)
            return {'start': d0.isoformat(), 'step_min': 60, **{s: [round(float(v), 3) for v in p50 * shh[s]] for s in sh}}
        except Exception as e:     # never block publishing the total
            log('per_array split failed:', e)
            return None

    def snapshot_others(self, c, issue: int) -> None:
        """Archive what the other forecasts in HA say right now, so they can be scored on the same day-ahead rule."""
        for e in self.cfg.solcast_entities:
            st = self.ha.get_state(e)
            for x in (st or {}).get('attributes', {}).get('detailedForecast', []) or []:
                t = pd.Timestamp(x['period_start']).tz_convert('UTC')
                c.execute('insert or replace into ext values(?,?,?,?,?,?)',
                          (issue, 'solcast', int(t.timestamp()), x.get('pv_estimate'), x.get('pv_estimate10'), x.get('pv_estimate90')))
        for e in self.cfg.open_meteo_entities:
            st = self.ha.get_state(e)
            for k, v in ((st or {}).get('attributes', {}).get('watts') or {}).items():
                t = pd.Timestamp(k).tz_convert('UTC')
                c.execute('insert or replace into ext values(?,?,?,?,?,?)', (issue, 'open_meteo', int(t.timestamp()), float(v) / 1000, None, None))

    def apply_knmi(self, c, Q: pd.DataFrame, issue: int, now: pd.Timestamp) -> dict:
        """Multiply H+1..H+3 of the published curve by knmi / reference, from the latest KNMI nowcast row set (ext provider
        'knmi', kw10 = the raw P50 it was computed against)."""
        res = {'hours_corrected': 0}
        k = pd.read_sql("select issue, target, kw, kw10 from ext where provider = 'knmi' and issue = (select max(issue) from ext where provider = 'knmi')", c)
        if k.empty or issue - int(k.issue.iloc[0]) > self.cfg.knmi_max_age_s:
            return res
        res['available'] = pd.Timestamp(int(k.issue.iloc[0]), unit='s', tz='UTC').isoformat()
        for r in k.itertuples():
            t = pd.Timestamp(r.target, unit='s', tz='UTC')
            if (r.kw10 and r.kw10 > 0 and t in Q.index and t > now.floor('h')       # current hour left alone
                    and abs(r.kw / r.kw10 - 1) > 1e-6
                    and abs(Q.p50[t] / r.kw10 - 1) < 0.3):                          # P50 moved a lot since: avoid double counting
                Q.loc[t] = (Q.loc[t] * min(max(r.kw / r.kw10, 0.2), 2.0)).clip(upper=self.cfg.ac_limit_kw)
                res['hours_corrected'] += 1
        return res


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if v != v else v
