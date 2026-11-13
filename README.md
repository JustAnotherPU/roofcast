# RoofCast

A solar PV forecast for Home Assistant that learns from your own inverter data. It pulls several weather models from Open-Meteo, turns each one into a PV estimate per array with pvlib, corrects each estimate with what your panels actually produced, and publishes the ensemble to Home Assistant in the same format as the Solcast integration, so EMHASS and existing cards can read it without changes.

It runs as a small Python program next to Home Assistant (a container, a VM, a Raspberry Pi), started by systemd timers. It is not a custom integration.

## Why

I wrote it for a roof with three faces pointing in different directions and some shade from trees. Solcast and the stock Open-Meteo integration model each face as an unshaded plane and know nothing about how a given inverter performs, so on this roof their forecasts were often far off.

On the author's roof, over the last 30 days, the day-ahead error (WMAPE of daily energy) was 21 % for this project, against 32 % for the stock Open-Meteo integration and 44 % for Solcast. That is one roof, one autumn, and your numbers will be different. The engine records the same measurement for your system, see "Accuracy tracking" below.

## How it works

- Eight NWP models through Open-Meteo by default: ICON, ECMWF IFS and AIFS, Meteo-France, KNMI HARMONIE-AROME, UKMO, DMI and GFS. Use the ones that cover your area.
- pvlib per array: Perez transposition, physical IAM, Faiman cell temperature, a simple temperature coefficient.
- Nightly calibration per array, per model and per lead day (today or day-ahead), trained on the Open-Meteo Previous Runs API, so each model is judged on the forecast it really gave a day earlier, not on its analysis. The calibration has a level and a shape over solar azimuth x elevation bins. Sparse bins are pulled back toward the level.
- Optional horizon mask per array, learned from clear days. It only removes part of the direct beam. Diffuse light still reaches a shaded panel.
- Rolling level factor k: actual / forecast over the last 28 days, half-life 14 days, bounded to 0.85 to 1.10. It is measured on the archived raw day-ahead members, before k. If you measure it on the published forecast instead, k only sees its own residual and drifts back to 1 (there is a test for that).
- P10, P25, P50 and P90 as fixed multipliers learned from past daily actual / forecast ratios.
- Hours where the measurement is wrong are left out of learning: inverter at zero while the forecast expected real output, values above the physical cap, and, if you configure it, hours where a full battery may have throttled the PV.
- Everything is archived in SQLite: each model's forecast, the published output, the actuals, and optionally Solcast and Open-Meteo integration forecasts for comparison.
- Day boundaries use local calendar days, so the 23 h and 25 h days at DST changes come out right.
- Optional KNMI satellite nowcast for the next 1 to 3 hours (Benelux only, see below).

## Install

Python 3.11 or newer. There is no PyPI package: install from a clone of this repository.

```sh
sudo useradd --system --home /var/lib/roofcast roofcast
git clone <this repository> roofcast && cd roofcast
sudo python3 -m venv /opt/roofcast/venv
sudo /opt/roofcast/venv/bin/pip install .                                # or '.[knmi]' for the nowcast
sudo mkdir -p /etc/roofcast
sudo cp config.example.yaml /etc/roofcast/config.yaml                    # then edit it
sudo cp deploy/systemd/roofcast.env.example /etc/roofcast/roofcast.env
sudo chmod 600 /etc/roofcast/roofcast.env                              # put the HA token in it
sudo cp deploy/systemd/*.service deploy/systemd/*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now roofcast-run.timer roofcast-learn.timer
```

With the systemd units, set `data_dir: /var/lib/roofcast` in the config.

Check the config, then do a first calibration and a first run by hand:

```sh
/opt/roofcast/venv/bin/roofcast -c /etc/roofcast/config.yaml check
sudo systemctl start roofcast-learn.service
sudo systemctl start roofcast-run.service
journalctl -u roofcast-learn -u roofcast-run -n 50
```

Commands:

| command | what it does | when |
|---|---|---|
| `roofcast run` | fetch forecasts, publish to HA | every 15 min (timer) |
| `roofcast learn` | recalibrate on the history | nightly (timer) |
| `roofcast horizon [--write]` | learn the horizon mask from clear days | by hand, monthly in winter, then `learn` |
| `roofcast test` | offline self-test of the level factor | any time |
| `roofcast check` | validate the config | after editing it |
| `roofcast knmi [--score]` | KNMI nowcast, or its shadow score | every 15 min (optional timer) |

Run the manual commands (`horizon`, `knmi --score`) as the service user, with the env file loaded, so the files in the data dir keep the right owner and the token is available. The env file is readable by root only, so let systemd load it:

```sh
sudo systemd-run --pipe --wait --quiet --uid=roofcast --gid=roofcast \
  -p EnvironmentFile=/etc/roofcast/roofcast.env -p WorkingDirectory=/var/lib/roofcast \
  /opt/roofcast/venv/bin/roofcast -c /etc/roofcast/config.yaml horizon
```

## Configuration

Everything is in `config.example.yaml`, with comments. The parts that matter most:

- `location`: your latitude, longitude, altitude and IANA time zone.
- `arrays`: one entry per independently measured input (usually one MPPT). Give azimuth (180 = south), tilt and nameplate kWp, and the HA power sensor(s) for that input in `actual_entity`. If one input has panels facing two ways, list them under `faces`. Orientation matters more than kWp: the level is learned, the geometry is not.
- `since`: for an array added later. Before that date it counts as zero and its data is ignored. Until it has `new_array.level_only_hours` of data it gets a level only, starting from `new_array.start_efficiency`.
- `power_unit`: `W` or `kW`, the unit of your sensors. With a wrong unit, `learn` logs the reason and exits with an error instead of writing a calibration.
- `inverter_ac_limit_kw`: the forecast is clipped to it.
- `curtailment`: battery SoC and grid power sensors, to drop hours where a full battery may have throttled the PV. Set both or neither.
- `compare`: Solcast or Open-Meteo integration sensors to archive and score next to this forecast.

The HA token is read from the `HA_TOKEN` environment variable, never from the file. The power sensors must have `state_class: measurement` so that HA keeps long-term statistics for them: learning reads hourly means from those statistics.

## Home Assistant and EMHASS

Two sensors are published (names configurable):

- `sensor.pv_forecast_today`: state = P50 kWh today. Attributes: `p10`, `p25`, `p90`, `detailedForecast` (30-min slots for today and the next two days, Solcast keys `period_start`, `pv_estimate`, `pv_estimate10`, `pv_estimate90`, plus `pv_estimate25`, values in kW), `per_array` (hourly P50 per array, adds up to the total), `models_ok`, `models`, `generated_at`, `level_factor`, `knmi`, and accuracy stats such as `ensemble_wmape_30d`.
- `sensor.pv_forecast_tomorrow`: the same for tomorrow, plus `day_after_tomorrow_p50`.

These are state objects written through the REST API. They are not tied to an integration, so they disappear after an HA restart until the next run (at most 15 minutes).

The attributes of `sensor.pv_forecast_today` are about 21 KB with two arrays, above the 16 KB the recorder accepts, so HA logs a warning on every update. Exclude both sensors from the recorder:

```yaml
recorder:
  exclude:
    entities:
      - sensor.pv_forecast_today
      - sensor.pv_forecast_tomorrow
```

`ha/pv_forecast.yaml` is an example package with a switch to choose between this forecast and another one, a freshness sensor (published less than 2 h ago, at least 3 models), and an example REST command for an EMHASS MPC run. It sends the curve as `pv_power_forecast` with a timestamp per slot, so it works with a 15, 30 or 60-min EMHASS time step, plus `pv_power_forecast_p10` (EMHASS 0.18.4 or newer) and `soc_init`. Set the time step, horizon and battery SOC sensor at the top of the payload. EMHASS does not read the sensor by itself: it only sees the forecast through a call like this one. The REST command was tested on EMHASS 0.18.8: POSTed to a live instance at a 15-min step, and run offline at 15, 30 and 60 min.

## Accuracy tracking

Every night `learn` compares each day's actual production with the last forecast issued before 21:00 the day before, for this engine and for any sensors listed under `compare`. The results are in the `quality` table and in the attributes `<provider>_wmape_7d`, `_wmape_30d`, `_bias_7d`, `_bias_30d`.

## Database size

Everything goes into `data_dir/roofcast.db`. With 8 models and a run every 15 minutes, the per-model archive (table `nwp`) grows by about 10 MB a day. `learn` deletes rows older than 35 days except the short-lead ones (up to 1 h ahead), which the outage filter needs for the whole history. Expect roughly 350 MB for the rolling window plus under 1 MB a day for what is kept, including table `out`. These figures are estimates. SQLite does not shrink the file by itself; run `sqlite3 roofcast.db VACUUM` now and then if disk space matters.

## KNMI nowcast (optional)

The Dutch weather service publishes a satellite-based solar irradiance nowcast for the next few hours. Its grid is only useful for roughly the Benelux area. To use it:

1. From the repository: `/opt/roofcast/venv/bin/pip install '.[knmi]'` (needs eccodes).
2. Get a free API key from the KNMI Data Platform and put `KNMI_API_KEY=...` in the env file.
3. Set `knmi.enabled: true` and enable `roofcast-knmi.timer`.

The nowcast is only used when the last measured hour was more than 20 % off the forecast and KNMI corrects in the same direction. That rule came out best in the author's backtest. It only changes the next 1 to 3 hours of the published curve. `roofcast knmi --score` compares forecast, rule and raw KNMI on the archive.

## Limitations

- It needs history. With less than about two weeks of measured production the calibration is rough, and before that it publishes plain pvlib times a starting efficiency, with default uncertainty bands.
- Runs published before the first successful `learn` are archived uncalibrated, and the level factor k reads that archive, so they can pull k toward its bounds for up to 28 days.
- Some thresholds (the minimum daily energy for a usable day, the outage detector) are scaled from the system size and were tuned on one roof.
- A snow day looks like an outage to the filter and is left out of learning and scoring.
- The KNMI nowcast covers the Benelux area only.
- Open-Meteo is free for non-commercial use. Check their terms if your use is commercial.
- No warranty. Do not let it drive anything that can do damage without a sanity check on your side.

## Tests

```sh
pip install -e '.[test]'
pytest
```

The tests run offline (network access fails them).

## License

MIT, see `LICENSE`.
