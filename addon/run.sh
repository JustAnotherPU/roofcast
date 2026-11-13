#!/bin/sh
# /config is this add-on's folder, /addon_configs/<id>_roofcast on the host. /data persists across updates.
set -e
CFG=/config/config.yaml
if [ ! -f "$CFG" ]; then
    sed -e 's|url: http://homeassistant.local:8123|url: http://supervisor/core|' \
        -e 's|^data_dir: ./data |data_dir: /data   |' /opt/roofcast/config.example.yaml > "$CFG"
    echo "Created $CFG from the example. Edit it (location, arrays, sensors) and restart the add-on."
    exit 1
fi
export HA_TOKEN="$SUPERVISOR_TOKEN" ROOFCAST_CONFIG="$CFG"
exec python /opt/roofcast/deploy/docker/scheduler.py
