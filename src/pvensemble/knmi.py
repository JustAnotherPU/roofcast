"""Optional: KNMI satellite nowcast (Data Platform dataset `surface_solar_irradiance` 1.0, 0-4 h ahead, GRIB2).

Coverage: the dataset grid is centred on the Netherlands and is only useful for roughly the Benelux area.
Needs the `knmi` extra (`pip install '.[knmi]'` from the repository, for eccodes) and a free KNMI Data Platform API key in $KNMI_API_KEY.

Writes two row sets to table ext (kw10 = the raw P50 of table out it was computed against):
  knmi_raw : P50 x (KNMI GHI / mean model GHI), for full hours after the file became available
  knmi     : the rule kept after backtesting: use KNMI only when the last measured hour was off the forecast by more
             than 20 % (|log ratio| > 0.2) AND KNMI corrects in the same direction; otherwise keep P50.
Engine.run() then applies knmi / kw10 to H+1..H+3 of the published curve (never to table out).
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import requests

from .engine import Engine, log

API = 'https://api.dataplatform.knmi.nl/open-data/v1/datasets/surface_solar_irradiance/versions/1.0/files'
GRID_LAT0, GRID_LON0, GRID_STEP = 35.0, -11.0, 0.05     # regular lat/lon grid, origin 35 N / 11 W
THR, CLIP = 0.2, (0.2, 2.0)


def grid_index(lat: float, lon: float) -> tuple[int, int]:
    return round((lat - GRID_LAT0) / GRID_STEP), round((lon - GRID_LON0) / GRID_STEP)


def ghi_15min(path: str, lat: float, lon: float) -> pd.Series:
    """Accumulated ssrd (J/m2) since issue -> mean GHI (W/m2) per 15-min slot, 3x3 pixels around the site."""
    import eccodes  # optional extra

    J, I = grid_index(lat, lon)
    acc = []
    with open(path, 'rb') as f:
        while (g := eccodes.codes_grib_new_from_file(f)) is not None:
            ni = eccodes.codes_get(g, 'Ni')
            v = eccodes.codes_get_values(g).reshape(-1, ni)
            if not (1 <= J < v.shape[0] - 1 and 1 <= I < ni - 1):
                eccodes.codes_release(g)
                raise ValueError('location is outside the KNMI nowcast grid (Benelux area only)')
            end = int(eccodes.codes_get(g, 'stepRange').split('-')[1].rstrip('mh')) * (60 if eccodes.codes_get(g, 'stepUnits') == 1 else 1)
            acc.append((end, float(v[J - 1:J + 2, I - 1:I + 2].mean())))
            eccodes.codes_release(g)
    acc = pd.DataFrame(sorted(acc), columns=['end', 'cum'])
    return pd.Series(acc.cum.diff().fillna(acc.cum).values / (acc.end.diff().fillna(acc.end).values * 60), index=acc.end)


def rule(r_prev: float, k: float) -> bool:
    """True when KNMI should replace P50 for this hour (see module doc)."""
    return bool(r_prev == r_prev and r_prev > 0 and k > 0 and abs(np.log(r_prev)) > THR and np.sign(np.log(r_prev)) == np.sign(np.log(k)))


def prod_last_hour(eng: Engine, start: pd.Timestamp) -> float:
    """Mean PV production (kW) over the hour, NaN on a gap or when output may be curtailed (same rule as learning)."""
    cfg = eng.cfg
    pv = sorted({e for a in cfg.arrays for e in a.entities})
    extra = [x for x in (cfg.soc_entity, cfg.grid_entity) if x]
    m = eng.ha.hour_means(start, pv + extra)
    if m is None:
        return np.nan
    if cfg.soc_entity and cfg.grid_entity and m[cfg.soc_entity] >= cfg.soc_full_pct and m[cfg.grid_entity] > cfg.export_threshold_w:
        return np.nan
    return sum(m[x] for x in pv) * cfg.power_factor


def main(eng: Engine) -> int:
    cfg = eng.cfg
    now = pd.Timestamp.now(tz='UTC')
    if not 5 <= now.hour <= 17:      # night (UTC hours, fine for the covered area)
        return 0
    S = requests.Session()
    S.headers['Authorization'] = cfg.knmi_key()
    f = S.get(f'{API}?maxKeys=1&sorting=desc&orderBy=created', timeout=30).json()['files'][0]
    issue = pd.Timestamp(f['filename'].split('__')[2][:15], tz='UTC')
    avail = pd.Timestamp(f['created'])
    c = eng.db()
    if c.execute("select 1 from ext where provider = 'knmi_raw' and issue = ? limit 1", (int(avail.timestamp()),)).fetchone():
        return 0                       # already processed
    path = cfg.path('knmi_latest.grb2')
    url = S.get(f"{API}/{f['filename']}/url", timeout=30).json()['temporaryDownloadUrl']
    with open(path, 'wb') as fh:
        fh.write(requests.get(url, timeout=120).content)
    g = ghi_15min(path, cfg.latitude, cfg.longitude)
    g.index = issue + pd.to_timedelta(g.index - 15, unit='m')            # start of each slot
    gh = g.groupby(g.index.floor('h')).agg(['mean', 'size'])
    gh = gh[(gh['size'] == 4) & (gh.index >= avail)]['mean']
    if gh.empty:
        log('KNMI: no full hour in this file')
        return 0
    av = int(avail.timestamp())
    lo, hi = int(gh.index.min().timestamp()) - 3600, int(gh.index.max().timestamp())
    out = pd.read_sql('select issue, target, p50 from out where issue <= ? and target between ? and ?', c, params=(av, lo, hi))
    out = out[out.issue == out.groupby('target').issue.transform('max')].set_index('target').p50
    nwp = pd.read_sql('select issue, provider, target, ghi from nwp where issue <= ? and target between ? and ?', c, params=(av, lo + 3600, hi))
    nwp = nwp[nwp.issue == nwp.groupby(['provider', 'target']).issue.transform('max')].groupby('target').ghi.mean()
    hp = (av // 3600) * 3600 - 3600                                       # last full hour
    try:
        y = prod_last_hour(eng, pd.Timestamp(hp, unit='s', tz='UTC'))
    except Exception as e:
        log('KNMI: reading production from HA failed:', e)
        y = np.nan
    row = c.execute('select p50 from out where target = ? and issue <= ? order by issue desc limit 1', (hp, hp)).fetchone()
    p_prev = row[0] if row else np.nan
    r_prev = y / p_prev if p_prev == p_prev and p_prev > 0.02 * cfg.total_kwp and y == y else np.nan
    rows = []
    for t, gk in gh.items():
        ts = int(t.timestamp())
        p50, gn = out.get(ts), nwp.get(ts)
        if p50 is None or gn is None or gn <= 30:
            continue
        k = min(max(gk / gn, CLIP[0]), CLIP[1])
        raw = p50 * k
        rows += [(av, 'knmi_raw', ts, raw, p50, None), (av, 'knmi', ts, raw if rule(r_prev, k) else p50, p50, None)]
    c.executemany('insert or replace into ext values(?,?,?,?,?,?)', rows)
    c.commit()
    c.close()
    log(f'KNMI issue {issue:%H:%M} (available {avail:%H:%M}): {len(rows) // 2} h archived, last hour actual/forecast = '
        + (f'{r_prev:.2f}' if r_prev == r_prev else 'n/a (rule inactive)'))
    return 0


def score(db_path: str) -> str:
    """Shadow-mode report: hourly WMAPE of P50 vs the KNMI rule vs raw KNMI, per lead hour, on the archive. Read only."""
    c = sqlite3.connect(f'file:{os.path.abspath(db_path)}?mode=ro', uri=True)
    e = pd.read_sql("select issue, provider, target, kw from ext where provider in ('knmi', 'knmi_raw')", c)
    if e.empty:
        return 'nothing to score yet'
    e = e.pivot_table(index=['issue', 'target'], columns='provider', values='kw').reset_index()
    act = pd.read_sql('select hour, total from actual', c).set_index('hour').total
    out = pd.read_sql('select issue, target, p50 from out where target >= ?', c, params=(int(e.target.min()),))
    c.close()
    e['p50'] = [o.loc[o.issue.idxmax(), 'p50'] if not (o := out[(out.target == t) & (out.issue <= i)]).empty else np.nan
                for i, t in zip(e.issue, e.target)]
    e['y'] = e.target.map(act)
    e['lead'] = (e.target - (e.issue // 3600) * 3600) // 3600
    e = e.dropna(subset=['y', 'p50', 'knmi'])
    e = e[(e.p50 + e.y) > 0.05]
    if e.empty:
        return 'nothing to score yet (table actual is filled by the nightly learn)'

    def w(g, col):
        return 100 * (g[col] - g.y).abs().sum() / g.y.sum()

    lines = [f'H+{lead}: n={len(g):4d}  forecast {w(g, "p50"):5.1f}%  KNMI rule {w(g, "knmi"):5.1f}%  KNMI raw {w(g, "knmi_raw"):5.1f}%'
             for lead, g in e.groupby('lead')]
    lines.append(f'ALL: forecast {w(e, "p50"):.1f}%  KNMI rule {w(e, "knmi"):.1f}%  KNMI raw {w(e, "knmi_raw"):.1f}%')
    return '\n'.join(lines)
