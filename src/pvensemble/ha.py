"""Minimal Home Assistant client: REST for states and history, websocket for long-term statistics."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import requests


class HomeAssistant:
    def __init__(self, url: str, token: str):
        self.url = url.rstrip('/')
        self.token = token
        self.s = requests.Session()
        self.s.headers['Authorization'] = 'Bearer ' + token

    def set_state(self, entity: str, state, attrs: dict) -> None:
        r = self.s.post(f'{self.url}/api/states/{entity}', json={'state': state, 'attributes': attrs}, timeout=30)
        r.raise_for_status()

    def get_state(self, entity: str) -> dict | None:
        r = self.s.get(f'{self.url}/api/states/{entity}', timeout=30)
        return r.json() if r.ok else None

    def statistics(self, ids: list[str], start: str) -> dict[str, pd.Series]:
        """Hourly mean of each statistic id since `start` (ISO), indexed by the start of the hour (UTC).
        Uses the recorder's long-term statistics, which keep months of history even with a short recorder purge window."""
        import websocket  # websocket-client, imported lazily so the package imports without it

        ws = websocket.create_connection(self.url.replace('http', 'ws', 1) + '/api/websocket', timeout=60)
        try:
            ws.recv()
            ws.send(json.dumps({'type': 'auth', 'access_token': self.token}))
            if json.loads(ws.recv()).get('type') != 'auth_ok':
                raise RuntimeError('Home Assistant websocket authentication failed')
            ws.send(json.dumps({'id': 1, 'type': 'recorder/statistics_during_period', 'start_time': start,
                                'period': 'hour', 'statistic_ids': ids, 'types': ['mean']}))
            r = json.loads(ws.recv())
        finally:
            ws.close()
        if not r.get('success', True):
            raise RuntimeError(f'statistics request failed: {r.get("error")}')
        return {k: pd.Series({pd.Timestamp(x['start'], unit='ms', tz='UTC'): x.get('mean') for x in v}, dtype=float)
                for k, v in r['result'].items()}

    def hour_means(self, start: pd.Timestamp, ents: list[str]) -> dict[str, float] | None:
        """Time-weighted mean of each entity over [start, start + 1 h) from the state history; None if there is a gap."""
        s, e = start, start + pd.Timedelta(hours=1)
        r = self.s.get(f'{self.url}/api/history/period/{(s - pd.Timedelta(hours=1)):%Y-%m-%dT%H:%M:%SZ}',
                       params={'end_time': e.isoformat(), 'filter_entity_id': ','.join(ents),
                               'minimal_response': '', 'no_attributes': ''}, timeout=30)
        r.raise_for_status()
        m = {}
        for ser in r.json():
            vals = []
            for p in ser:
                try:
                    vals.append(float(p['state']))
                except (TypeError, ValueError):
                    vals.append(np.nan)
            x = pd.Series(vals, index=pd.to_datetime([p['last_changed'] for p in ser], utc=True, format='ISO8601')).ffill()
            x = x.reindex(x.index.union([s, e])).ffill()[s:e]
            w = np.diff(x.index.values).astype('timedelta64[ms]').astype(float) / 1000
            if w.sum() < 3000:  # less than 50 min covered: do not trust the mean
                return None
            m[ser[0]['entity_id']] = float((x.values[:-1] * w).sum() / w.sum())   # NaN on an unavailable stretch: no rule
        return m if len(m) == len(set(ents)) else None
