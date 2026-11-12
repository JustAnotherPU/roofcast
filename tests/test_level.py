import numpy as np

from roofcast.cli import selftest
from roofcast.engine import LEVEL_BOUNDS, level_factor


def test_bounds_and_fallback():
    age = np.ones(30)
    x = np.full(30, 20.0)
    assert level_factor(x, x, age) == (1.0, 1.0)
    assert level_factor(x[:5], x[:5] * 2, age[:5]) == (1.0, 1.0)        # fewer than 10 days
    assert level_factor(x / 2, x, age) == (LEVEL_BOUNDS[0], 0.5)
    assert level_factor(x * 2, x, age) == (LEVEL_BOUNDS[1], 2.0)
    assert level_factor(x, np.zeros(30), age) == (1.0, 1.0)


def test_recent_days_weigh_more():
    y = np.array([20.0] * 15 + [18.0] * 15)
    f = np.full(30, 20.0)
    age = np.arange(30)[::-1]                                          # last entries are the most recent
    k = level_factor(y, f, age)[0]
    assert 0.9 < k < 0.95                                              # closer to 0.9 (recent) than to 0.95 (plain mean)


def test_no_feedback_selftest():
    assert selftest() == 0
