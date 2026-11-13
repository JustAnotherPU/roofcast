# The Supervisor passes BUILD_FROM; this add-on uses a plain Debian Python image instead,
# so numpy, pandas and pvlib install from ready-made wheels.
FROM python:3.12-slim

ARG ROOFCAST_REF
ARG ROOFCAST_SRC=https://github.com/JustAnotherPU/roofcast/archive/${ROOFCAST_REF}.tar.gz
ADD ${ROOFCAST_SRC} /tmp/roofcast.tar.gz
RUN mkdir /opt/roofcast && tar xzf /tmp/roofcast.tar.gz -C /opt/roofcast --strip-components=1 \
    && pip install --no-cache-dir /opt/roofcast && rm /tmp/roofcast.tar.gz

COPY run.sh /run.sh
RUN chmod +x /run.sh
ENV PYTHONUNBUFFERED=1
CMD ["/run.sh"]
