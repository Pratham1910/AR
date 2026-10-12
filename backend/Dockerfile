# TVASTA backend. Python 3.11 because the pins in requirements.txt
# (numpy 1.26.4, open3d 0.18.0, opencv 4.10.0.84, psycopg2-binary 2.9.9) only
# publish wheels up to 3.11/3.12 — on a newer interpreter pip falls back to
# source builds that fail.
#
# The base image is pinned by tag AND digest so a rebuild next year starts from
# the same bytes. To update it:
#   docker pull python:3.11-slim-bookworm
#   docker image inspect python:3.11-slim-bookworm --format '{{index .RepoDigests 0}}'
#   docker run --rm python:3.11-slim-bookworm python --version
# then put the new patch version and digest below (or pass --build-arg
# PYTHON_IMAGE=...), rebuild, and run the checks in docs/docker.md.
ARG PYTHON_IMAGE=python:3.11.17-slim-bookworm@sha256:0a310eeecf4e1f5a0743f9a6520c90c88d089c903ca5fd283f501e3a805f5f89

# Optional local wheelhouse. Empty by default, so pip downloads everything.
# A directory of pre-downloaded wheels can take its place:
#   docker build --build-context wheels=docker/wheelhouse backend
# (docker-compose.yml does this already)
# When it contains wheels, pip installs from it ALONE (--no-index): given the
# choice, pip downloads a wheel again rather than use the local copy. So it
# must hold every wheel the build needs, or be left empty. Useful
# on a slow or unreliable link, where one stalled 400 MB download otherwise
# fails the whole build (docs/docker.md, "Slow or metered connections").
FROM scratch AS wheels

FROM ${PYTHON_IMAGE}

# PYTHONUNBUFFERED: log lines reach `docker logs` as they are written, so a
# crash during startup is not lost in a buffer.
# YOLO_CONFIG_DIR / MPLCONFIGDIR: ultralytics and matplotlib write a settings
# file / font cache on import and need a directory the runtime user owns.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    HOME=/home/tvasta \
    YOLO_CONFIG_DIR=/home/tvasta/.config \
    MPLCONFIGDIR=/home/tvasta/.cache/matplotlib

# Shared libraries the wheels load at import time and do not bundle:
#   libgl1, libegl1, libgomp1 - open3d (its pybind module links libGL/libEGL/libgomp)
#   libglib2.0-0              - opencv
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Runtime user. The ids are build args because docker-compose bind-mounts
# ./data (uploaded GLBs, reference images, calibrations) and the files written
# there should belong to the developer, not root. 1000:1000 is the first user
# on most Linux installs.
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd --gid "${APP_GID}" tvasta \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --home-dir /home/tvasta tvasta \
    # /data and /models are where the app's "../data/..." defaults and
    # MODEL_PATH resolve from WORKDIR /app. Created here so the image also
    # runs without the compose bind mounts.
    && mkdir -p /app /data /models /opt/tvasta/weights /home/tvasta/.config /home/tvasta/.cache/matplotlib \
    && chown -R tvasta:tvasta /app /data /models /opt/tvasta /home/tvasta

WORKDIR /app

# The pip cache is a BuildKit cache mount: downloaded wheels survive between
# builds on this machine but are not part of the image. --cache-dir is explicit
# because HOME points at the runtime user's directory, which root's pip refuses
# to cache into.
#
# torch is not in requirements.txt (ultralytics pulls it in), so left alone pip
# takes the newest release and, on Linux, the CUDA build with ~2.5 GB of nvidia
# wheels. Pin it and default to the CPU build: ultralytics picks CUDA when
# torch reports it and falls back to CPU otherwise, so inference behaves the
# same either way. For a GPU image build with
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
# (see docker-compose.gpu.yml).
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.4.1
ARG TORCHVISION_VERSION=0.19.1
RUN --mount=type=cache,target=/var/cache/pip \
    --mount=type=bind,from=wheels,target=/wheels \
    if ls /wheels/*.whl >/dev/null 2>&1; then src="--no-index --find-links /wheels"; \
    else src="--index-url ${TORCH_INDEX_URL}"; fi \
    && pip install --cache-dir /var/cache/pip ${src} \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

# Dependencies before the application code: this layer is rebuilt only when
# the pins change, not on every code edit. constraints.txt freezes the
# transitive dependencies requirements.txt leaves open.
COPY requirements.txt constraints.txt ./
RUN --mount=type=cache,target=/var/cache/pip \
    --mount=type=bind,from=wheels,target=/wheels \
    if ls /wheels/*.whl >/dev/null 2>&1; then src="--no-index --find-links /wheels"; else src=""; fi \
    && pip install --cache-dir /var/cache/pip ${src} -r requirements.txt -c constraints.txt \
    && pip check

USER tvasta

# Segmentation weights (SEGMENTATION_MODEL_NAME, default yolov8n-seg.pt).
# ultralytics downloads them into the working directory on first use, which an
# offline deployment cannot do, so they are put in the image at build time: a
# copy already in the build context is used as is (the glob matches nothing on
# a fresh clone, where the file is gitignored), otherwise it is downloaded
# here. Loading it also proves torch + ultralytics + numpy work together.
COPY --chown=tvasta:tvasta yolov8n-seg.p[t] ./
RUN python -c "from ultralytics import YOLO; m = YOLO('yolov8n-seg.pt'); print('segmentation weights ok:', len(m.names), 'classes')"
# A second copy outside /app: docker-compose bind-mounts the source tree over
# /app for development, which hides everything the image put there.
# SEGMENTATION_MODEL_NAME points here in that case.
RUN cp yolov8n-seg.pt /opt/tvasta/weights/yolov8n-seg.pt

COPY --chown=tvasta:tvasta . .

EXPOSE 8000

# /health is the app's own liveness route (app/main.py); it does not touch the
# database, so it reports "the server is up", not "the schema is migrated".
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

# Exec form: uvicorn is PID 1 and receives SIGTERM from `docker stop` itself.
# --timeout-graceful-shutdown: the AR page holds a WebSocket open, and without
# a limit uvicorn waits for it forever and is SIGKILLed when the stop grace
# period runs out. One worker only: app/api/vision.py keeps AR sessions in
# process memory.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "10"]
