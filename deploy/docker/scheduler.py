"""Container entry point: the schedule of the systemd timers, without systemd.

run every 15 min (:07 :22 :37 :52), learn at 01:30 UTC with up to 3 retries 20 min apart,
knmi at :12 :27 :42 :57 if knmi.enabled. On start, learn first when calibration.json is
missing or older than a day, so the first runs are not archived uncalibrated.
With arguments, it runs that one roofcast command and exits.
"""
import datetime as dt
import os
import signal
import subprocess
import sys
import time

from roofcast.config import ConfigError, load

CFG = os.environ.get('ROOFCAST_CONFIG', '/config/config.yaml')
RUN_MIN, KNMI_MIN = {7, 22, 37, 52}, {12, 27, 42, 57}
LEARN_AT, RETRY_S, RETRIES = (1, 30), 20 * 60, 3


def roofcast(*args):
    print(dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ'), 'roofcast', *args, flush=True)
    try:
        return subprocess.run(['roofcast', '-c', CFG, *args], timeout=900).returncode
    except subprocess.TimeoutExpired:
        print('timed out after 900 s', flush=True)
        return 1


def main():
    if len(sys.argv) > 1:
        os.execvp('roofcast', ['roofcast', '-c', CFG, *sys.argv[1:]])
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))      # PID 1 ignores SIGTERM without a handler
    if roofcast('check'):
        sys.exit(2)
    try:
        cfg = load(CFG)
    except ConfigError as e:
        sys.exit(f'config error: {e}')
    cal = cfg.path('calibration.json')
    retry_at, retries_left = None, 0

    def learn(first):
        nonlocal retry_at, retries_left
        retries_left = RETRIES if first else retries_left - 1
        ok = roofcast('learn') == 0
        retry_at = None if ok or retries_left <= 0 else time.time() + RETRY_S

    if not os.path.exists(cal) or time.time() - os.path.getmtime(cal) > 86400:
        learn(first=True)
    while True:
        # shortcut: one job at a time, so a slot that falls inside a long learn is skipped (next one is 15 min later)
        time.sleep(60 - time.time() % 60 + 1)
        now = dt.datetime.now(dt.timezone.utc)
        if (now.hour, now.minute) == LEARN_AT:
            learn(first=True)
        elif retry_at and time.time() >= retry_at:
            learn(first=False)
        if now.minute in RUN_MIN:
            roofcast('run')
        if cfg.knmi_enabled and now.minute in KNMI_MIN:
            roofcast('knmi')


if __name__ == '__main__':
    main()
