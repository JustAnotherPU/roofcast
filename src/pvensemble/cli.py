"""Command line: pvensemble {run,learn,horizon,test,knmi}."""
from __future__ import annotations

import argparse
import os
import sys

from .config import ConfigError, load


def selftest() -> int:
    """Level factor: bounds, fallback, and no feedback (measured on the raw forecast, never on the published one)."""
    import numpy as np

    from .engine import LEVEL_BOUNDS, level_factor

    n = 30
    age = np.ones(n)
    x = np.full(n, 20.0)
    y = x / 1.10                                                   # raw forecast constantly 10 % too high
    assert level_factor(x, x, age) == (1.0, 1.0)
    assert level_factor(y[:5], x[:5], age[:5]) == (1.0, 1.0)      # too few days
    assert level_factor(x / 2, x, age)[0] == LEVEL_BOUNDS[0] and level_factor(x * 2, x, age)[0] == LEVEL_BOUNDS[1]
    k_raw = level_factor(y, x, age)[0]
    assert abs(k_raw - 1 / 1.10) < 1e-9
    k, pub = 1.0, []                                               # simulated deployment: published_t = raw_t * k_(t-1)
    for i in range(n):
        pub.append(x[i] * k)
        k = level_factor(y[:i + 1], pub, age[:i + 1])[0]
    assert k > k_raw + 0.03, (k_raw, k)                            # on the PUBLISHED curve k only sees its residual
    print('selftest ok: k on the raw forecast', round(k_raw, 4), '; measured on the published one it drifts to', round(k, 4))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog='pvensemble', description=__doc__)
    p.add_argument('-c', '--config', default=os.environ.get('PVENSEMBLE_CONFIG', 'config.yaml'))
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('run', help='fetch forecasts, publish to Home Assistant (every 15 min)')
    sub.add_parser('learn', help='recalibrate on the measured history (nightly)')
    h = sub.add_parser('horizon', help='learn the horizon mask from clear days')
    h.add_argument('--write', action='store_true', help='write horizon.json (needs >= 15 clear days)')
    sub.add_parser('test', help='offline self-test of the level factor')
    sub.add_parser('check', help='validate the config file and exit')
    k = sub.add_parser('knmi', help='optional KNMI satellite nowcast (Benelux only)')
    k.add_argument('--score', action='store_true', help='print the shadow-mode score instead of fetching')
    a = p.parse_args(argv)
    if a.cmd == 'test':
        return selftest()
    try:
        cfg = load(a.config)
    except ConfigError as e:
        print('config error:', e, file=sys.stderr)
        return 2
    if a.cmd == 'check':
        print(f'ok: {len(cfg.arrays)} arrays, {cfg.total_kwp:.2f} kWp, {len(cfg.models)} models')
        return 0
    from .engine import Engine

    eng = Engine(cfg)
    try:
        if a.cmd == 'run':
            return eng.run()
        if a.cmd == 'learn':
            return eng.learn()
        if a.cmd == 'horizon':
            from . import horizon
            return horizon.main(eng, write=a.write)
        if a.cmd == 'knmi':
            from . import knmi
            if a.score:
                print(knmi.score(eng.db_path))
                return 0
            if not cfg.knmi_enabled:
                print('knmi.enabled is false in the config, nothing to do')
                return 0
            return knmi.main(eng)
    except ConfigError as e:
        print('config error:', e, file=sys.stderr)
        return 2
    return 0
