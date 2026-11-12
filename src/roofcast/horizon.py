"""Learn a horizon mask per array from measured clear days, and write horizon.json (read by Engine.pv()).

Clear day: the reference array (largest single-face array) produced >= 70 % of its clear-sky output and the models
agreed it was clear (GHI >= 90 % of clear sky). For each clear hour: share of the direct beam that got through
f = (actual / k - diffuse) / beam, with k the array's unshaded level (85th percentile, sun above 25 deg).
Per 5 deg azimuth bin (pooled +-5 deg): the height h <= 35 deg and the f below it (median) that best explain the points.
A bin is kept with >= 6 points over >= 3 days, f <= 0.8 and an error reduction of >= 30 %. Isolated bins are dropped,
empty bins mean no mask. Not scheduled: run it by hand (monthly in winter is enough), then `roofcast learn`.
"""
from __future__ import annotations

import json
import os
import shutil
import time

import numpy as np
import pandas as pd

from .engine import Engine, log

STEP, POOL, H_MAX = 5, 5, 35.0   # above 35 deg an "obstacle" is almost always a model artefact, not a tree or a roof
MIN_CLEAR_DAYS = 15


def parts(eng: Engine, df: pd.DataFrame) -> pd.DataFrame:
    """Beam and diffuse part per array: once without mask, once with the beam cut everywhere (same temperature factor)."""
    F1 = eng.pv(df, horizon={})
    F0 = eng.pv(df, horizon={a.name: {'step': STEP, 'bins': {str(z): [90.0, 0.0] for z in range(0, 360, STEP)}} for a in eng.cfg.arrays})
    out = pd.DataFrame({'el': F1.el, 'saz': F1.saz})
    for a in eng.cfg.arrays:
        out[a.name + '_b'] = F1[a.name] - F0[a.name]
        out[a.name + '_d'] = F0[a.name]
    return out


def derive(eng: Engine, a: pd.DataFrame, R: pd.DataFrame) -> dict:
    """a: measured production (Engine.actuals()), R: mean model weather on the same hourly index."""
    cs = eng.loc.get_clearsky(a.index + pd.Timedelta(minutes=30), model='ineichen')
    cs.index = a.index
    C = parts(eng, pd.DataFrame({'shortwave_radiation': cs.ghi, 'direct_normal_irradiance': cs.dni, 'diffuse_radiation': cs.dhi,
                                 'temperature_2m': R.temperature_2m, 'wind_speed_10m': R.wind_speed_10m}, index=a.index))
    C['day'] = a.index.tz_convert(eng.tz).date
    C['kt'] = R.shortwave_radiation / cs.ghi.where(cs.ghi > 50)
    single = [x for x in eng.cfg.arrays if len(x.faces) == 1]
    if not single:
        raise ValueError('horizon learning needs at least one single-face array')
    ref = max(single, key=lambda x: x.kwp).name
    s = C[C.el > 5].assign(y=a[ref], x=C[ref + '_b'] + C[ref + '_d']).groupby('day').agg(y=('y', 'sum'), x=('x', 'sum'), kt=('kt', 'mean'))
    clear = C.day.map((s.y / s.x >= 0.70) & (s.kt >= 0.9)).fillna(False).astype(bool)
    sel = clear & a.ok & (C.el > 1)
    hz = {'generated': pd.Timestamp.now(tz='UTC').isoformat(), 'clear_days': int(C.day[clear].nunique()), 'reference': ref}
    for x in single:
        arr = x.name
        b, d = C[arr + '_b'], C[arr + '_d']
        m = sel & eng.active(x, a.index)
        kk = (a[arr] / (b + d))[m & (C.el > 25) & (b / (b + d) > 0.5)]
        if kk.empty:
            continue
        k = float(kk.quantile(0.85))
        P = pd.DataFrame({'el': C.el, 'az': (C.saz // STEP * STEP).astype(int), 'f': ((a[arr] / k - d) / b).clip(0, 1.3), 'day': C.day})[m & (b / (b + d) > 0.3)]
        bins = fit_bins(P)
        hz[arr] = {'step': STEP, 'k': round(k, 3), 'bins': bins}
    return hz


def fit_bins(P: pd.DataFrame) -> dict:
    """P: columns el, az (bin start), f (share of beam kept), day. Returns {az: [h, f, n points below h, lowest elevation seen]}."""
    bins = {}
    for az in range(0, 360, STEP):
        g = P[(P.az >= az - POOL) & (P.az <= az + POOL)]
        if len(g) < 6:
            continue
        base = float(((g.f - 1) ** 2).sum())
        best = (base, None, None)
        for h in [v for v in g.el + 0.5 if v <= H_MAX]:
            lo = g[g.el < h]
            if len(lo) < 6 or lo.day.nunique() < 3:
                continue
            fb = float(np.clip(lo.f.median(), 0, 1))
            loss = float(((lo.f - fb) ** 2).sum() + ((g[g.el >= h].f - 1) ** 2).sum())
            if loss < best[0]:
                best = (loss, h, fb)
        if best[1] is not None and best[2] <= 0.8 and best[0] < 0.7 * base:
            bins[str(az)] = [round(best[1], 1), round(best[2], 2), int((g.el < best[1]).sum()), round(float(g.el.min()), 1)]
    return {z: v for z, v in bins.items() if str(int(z) - STEP) in bins or str(int(z) + STEP) in bins}


def main(eng: Engine, write: bool = False) -> int:
    a = eng.actuals()
    today = pd.Timestamp.now(tz=eng.tz).normalize()
    a = a[a.index < today.tz_convert('UTC')]
    end = (today - pd.DateOffset(days=1)).date().isoformat()
    start = (today - pd.DateOffset(days=eng.cfg.history_days)).date().isoformat()
    a = a[a.index >= pd.Timestamp(start, tz='UTC')]
    R = pd.concat([eng.fetch_prev_runs(m, 0, start, end).reindex(a.index).interpolate(limit=2) for m in eng.cfg.models]).groupby(level=0).mean()
    hz = derive(eng, a, R)
    hz.update({k: v for k, v in eng.horizon.items() if k not in hz})   # arrays not recomputed keep their current mask
    print(json.dumps(hz, indent=1))
    if not write:
        return 0
    if hz['clear_days'] < MIN_CLEAR_DAYS:
        print(f"refused: {hz['clear_days']} clear days < {MIN_CLEAR_DAYS}, horizon.json unchanged")
        return 1
    if os.path.exists(eng.hz_path):
        shutil.copy(eng.hz_path, f'{eng.hz_path}.bak-{int(time.time())}')
    with open(eng.hz_path, 'w') as fh:
        json.dump(hz, fh, indent=1)
    log('horizon.json written, run `roofcast learn` next')
    return 0
