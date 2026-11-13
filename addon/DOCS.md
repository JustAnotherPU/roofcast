# RoofCast add-on

Runs RoofCast inside Home Assistant OS on the same schedule as the systemd timers: a forecast every 15 minutes and a calibration every night at 01:30 UTC. It talks to Home Assistant through the Supervisor, so no access token is needed.

## First start

1. Start the add-on once. It creates `config.yaml` in its folder (`/addon_configs/<id>_roofcast/`, visible in the File editor or Studio Code Server add-ons) and stops.
2. Edit that file: location, arrays and the HA power sensor of each array. Leave `home_assistant.url` at `http://supervisor/core` and `data_dir` at `/data`.
3. Start the add-on again. The first run calibrates on your HA history before it publishes anything. Watch the log.

The configuration options are described in the [README](https://github.com/JustAnotherPU/roofcast#configuration). To try RoofCast without touching what EMHASS uses, publish under other sensor names (`publish.today`, `publish.tomorrow`) and list your current forecast under `compare`.

The KNMI nowcast is not included in the add-on.
