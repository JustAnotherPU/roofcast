import numpy as np

from roofcast import knmi


def test_rule_needs_a_big_miss_and_agreement():
    assert knmi.rule(0.7, 0.6)            # panels 30 % under forecast, KNMI also lower: use KNMI
    assert not knmi.rule(0.7, 1.3)        # KNMI disagrees with what the panels just did
    assert not knmi.rule(0.9, 0.6)        # last hour within 20 %: keep the forecast
    assert knmi.rule(1.4, 1.5) and not knmi.rule(np.nan, 1.5)


def test_grid_index():
    assert knmi.grid_index(35.0, -11.0) == (0, 0)
    assert knmi.grid_index(52.37, 4.90) == (347, 318)
