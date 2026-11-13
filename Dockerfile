FROM python:3.12-slim

# EXTRAS=knmi adds the optional KNMI nowcast (Benelux only)
ARG EXTRAS=""

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".${EXTRAS:+[$EXTRAS]}" && rm -rf /src
COPY deploy/docker/scheduler.py /usr/local/bin/roofcast-scheduler

RUN useradd --system --uid 1000 --home-dir /data roofcast && mkdir -p /data /config && chown roofcast /data
USER roofcast
WORKDIR /data
ENV ROOFCAST_CONFIG=/config/config.yaml PYTHONUNBUFFERED=1
VOLUME /data

# No arguments: run the schedule. With arguments: one roofcast command, e.g. `check` or `horizon`.
ENTRYPOINT ["python", "/usr/local/bin/roofcast-scheduler"]
