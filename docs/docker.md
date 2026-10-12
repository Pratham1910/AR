# Docker environment

The backend runs in a Python 3.11 image, so it no longer depends on whichever
Python the host has. (The pins in `backend/requirements.txt` — numpy 1.26.4,
open3d 0.18.0, opencv 4.10.0.84 — have no wheels for newer interpreters.)

## What runs where

```
 browser ──5173──► frontend   Vite dev server (node:20)
    │
    ├──────8000──► backend    FastAPI / uvicorn, one worker, user "tvasta" (not root)
    │                 │  starts only after `migrate` has exited 0
    │              migrate    one-shot: `alembic upgrade head`, same image
    │                 │
    │              postgres   postgres:16, volume pgdata        (host port 5432)
    └──────9000──► minio      evidence frames, volume miniodata (host ports 9000/9001)
```

| Service | Image | Reaches the others as |
|---|---|---|
| `backend`, `migrate` | `tvasta-backend:local`, built from `backend/Dockerfile` | `postgres:5432`, `minio:9000` |
| `postgres` | `postgres:16` | — |
| `minio` | `pgsty/minio` (via `docker-compose.override.yml`) | — |
| `frontend` | built from `frontend/Dockerfile` | the browser calls `http://127.0.0.1:8000` |

The webcam is the **browser's**: frames are captured in the page and sent to
the API. The backend container needs no camera device.

## Start

```bash
cp .env.example .env            # optional; every value has a development default
docker compose up -d            # first run builds the images (~1 GB of wheels); later runs reuse them
docker compose ps               # backend should become "healthy"
curl http://localhost:8000/health        # {"status":"ok"}
docker compose exec backend python -m app.workers.seed_demo   # optional demo data
```

Open http://localhost:5173. To keep running the backend by hand instead, start
only the infrastructure: `docker compose up -d postgres minio`.

Stop with `docker compose down`. **Never add `-v`**: that deletes the
`pgdata` and `miniodata` volumes, i.e. the database and all evidence.

## Configuration

`docker compose` reads the repo-root `.env` (template: `.env.example`).

| Variable | Default | Purpose |
|---|---|---|
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | `tvasta` / `tvasta` / `tvasta` | Database credentials, used by `postgres` and to build the backend's `DATABASE_URL` |
| `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` | `tvasta` / `tvasta12345` | MinIO credentials, used by `minio` and the backend |
| `MINIO_PUBLIC_ENDPOINT` | `localhost:9000` | Address the **browser** uses for evidence image links |
| `JWT_SECRET_KEY` | `dev-only-change-me` | Token signing key |
| `MODEL_PATH` | unset (mock detector) | Trained detector weights, e.g. `/models/yolo/best.pt` |
| `POSE_SERVICE_URL` | `http://host.docker.internal:8765` | `pose_service/` running on the host |
| `TVASTA_UID` / `TVASTA_GID` | `1000` / `1000` | Owner of files written to `./data` |
| `TVASTA_BACKEND_IMAGE` | `tvasta-backend:local` | Image to run (offline installs) |

The defaults are for development on one machine. For anything other people
can reach, set real values for the two passwords and `JWT_SECRET_KEY` in
`.env` (which is gitignored and excluded from the image). PostgreSQL and MinIO
apply their credentials only when their volume is first initialised; changing
them later in `.env` does not change what is stored.

Inside the compose network the backend always uses `postgres:5432` and
`minio:9000`; the `localhost` values in `.env.example` are for a backend
started by hand and are overridden in `docker-compose.yml`.

## Migrations

The schema is changed by exactly one thing: the `migrate` service, which runs
`alembic upgrade head` and exits. `backend` waits for it to finish
successfully, and `migrate` waits for PostgreSQL's health check
(`pg_isready`), so the API never runs against a missing or half-applied
schema. `alembic/env.py` also holds a PostgreSQL advisory lock while
migrating, so starting it twice at once is safe.

```bash
docker compose up -d                       # applies pending migrations, then starts the backend
docker compose run --rm migrate            # apply them without (re)starting anything else
docker compose run --rm migrate alembic current   # expected: "0005 (head)"
docker compose run --rm migrate alembic heads     # expected: "0005 (head)"
docker compose run --rm migrate alembic check     # expected: "No new upgrade operations detected."
docker compose logs migrate                # why a migration failed
```

A second `alembic upgrade head` does nothing. If a migration fails,
`migrate` exits non-zero, the whole upgrade is rolled back (PostgreSQL DDL is
transactional) and `backend` is not started.

Take a backup before upgrading a database that matters:

```bash
docker compose exec postgres pg_dump -U tvasta -Fc tvasta > tvasta-$(date +%F).dump
```

To run the migration tests against a real server (they create and drop their
own `tvasta_migtest_*` databases and touch nothing else), use a throwaway
PostgreSQL rather than the one holding your data:

```bash
docker run -d --rm --name tvasta-migtest -e POSTGRES_USER=tvasta -e POSTGRES_PASSWORD=tvasta \
  --tmpfs /var/lib/postgresql/data -p 127.0.0.1:55432:5432 postgres:16
cd backend
MIGRATION_TEST_DATABASE_URL=postgresql+psycopg2://tvasta:tvasta@127.0.0.1:55432/postgres \
  python -m pytest ../tests/backend/test_migrations.py -q
docker stop tvasta-migtest
```

## Data and writable paths

| In the container | On the host | Contents |
|---|---|---|
| `/data` | `./data` (bind mount) | `models/*.glb` (served at `/static/models`), uploads, `reference_images/`, `part_calibration/`, `procedures/`, `clips/`, `debug_frames/` |
| `/models` | `./models` (bind mount) | Trained detector weights for `MODEL_PATH` |
| `/app` | `./backend` (bind mount) | Source code, reloaded on change. The image also holds a copy, used when it is run without compose |
| `/opt/tvasta/weights` | — (in the image) | `yolov8n-seg.pt` |
| volume `pgdata` | Docker volume `ar_pgdata` | PostgreSQL |
| volume `miniodata` | Docker volume `ar_miniodata` | Evidence frames |

The backend runs as user `tvasta` with uid/gid 1000. If your user has another
id (`id -u`), set `TVASTA_UID`/`TVASTA_GID` in `.env` and rebuild, otherwise
uploads into `./data` fail with "Permission denied".

## Models, GPU and offline use

- **Segmentation weights (`yolov8n-seg.pt`)** are baked into the image at build
  time (copied from `backend/` if present, otherwise downloaded during the
  build), so the AR features do not download anything at run time. Setting
  `SEGMENTATION_MODEL_NAME` to another stock model name makes ultralytics
  download it on first use into the container; for offline use put the file
  under `./models` and set `SEGMENTATION_MODEL_NAME=/models/<file>.pt`.
- **Detector weights (`MODEL_PATH`)** are not in the image. Put them under
  `./models` and set `MODEL_PATH=/models/yolo/<file>.pt`. Unset means the mock
  detector, as on the host.
- **CPU by default.** The image carries CPU-only torch. Ultralytics uses CUDA
  when torch reports it and the CPU otherwise, so results are the same, only
  slower. For GPU inference add `-f docker-compose.gpu.yml` (see that file):
  it needs the NVIDIA driver plus the NVIDIA Container Toolkit on the host, and
  builds a several-GB-larger image with CUDA torch.
- **Model-based (CAD) tracking** needs `pose_service/` (MegaPose, CUDA, its
  own Python 3.10 environment) running **on the host** — it is not part of
  compose. See [Pose service](#pose-service) below.
- **FBX upload** needs Blender, which is not in the image (GLB upload works).
- Ultralytics checks for connectivity when it is imported; with no network
  that costs a few seconds on the first AR request and nothing else.

The backend image is about 4.6 GB (open3d and its dependencies ~2.1 GB, CPU
torch ~1 GB).

**Building needs internet** (base image, Debian packages, PyPI, the
segmentation weights). **Running does not.** For a machine without internet,
build on a connected one and move the images:

```bash
# connected machine
docker compose build backend frontend
docker save -o tvasta-images.tar tvasta-backend:local ar-frontend postgres:16 pgsty/minio:latest

# offline machine: copy the tar and the repository (including data/ and models/)
docker load -i tvasta-images.tar
docker compose up -d --no-build
```

### Slow or metered connections, offline builds

The image needs about 1 GB of wheels (open3d alone is ~400 MB) and the pip in
the base image cannot resume an interrupted download: one stall past its
timeout fails the build. Put the wheels in `docker/wheelhouse/` (contents
gitignored) with any resumable downloader, or copy that directory from a
machine that has it. When it contains wheels the build installs from it
**alone** and never contacts PyPI, so it must hold every wheel the build
needs; leave it empty to download as usual. `docker compose build` passes the
directory automatically; with plain `docker build`:

```bash
docker build --build-context wheels=docker/wheelhouse -t tvasta-backend:local backend
```

## Pose service

"Model-based pose service not reachable at http://host.docker.internal:8765"
means one of two things:

1. **It is not running.** Nothing is listening on the host's port 8765
   (`ss -ltn | grep 8765`). Install it once following `pose_service/README.md`
   (needs an NVIDIA GPU with CUDA and several GB of downloads), then start it:

   ```bash
   bash pose_service/start.sh
   curl http://localhost:8765/health      # expect "cuda": true
   ```

2. **The host firewall drops the container's connection.** The backend
   container reaches the host through the Docker bridge, and a default-deny
   firewall such as `ufw` silently discards that traffic — the symptom is a
   *timeout* rather than "connection refused". Allow the Docker networks to
   reach the port (once):

   ```bash
   sudo ufw allow from 172.16.0.0/12 to any port 8765 proto tcp
   ```

Check from inside the container:

```bash
docker compose exec backend python -c \
  "import httpx; print(httpx.get('http://host.docker.internal:8765/health', timeout=5).text)"
```

## Rebuilding and updating

- **Code change: nothing to do.** `docker-compose.yml` mounts `./backend` over
  `/app` and runs uvicorn with `--reload`, and the frontend container runs the
  Vite dev server on a mount of `./frontend`. Saving a file is enough. Start
  the stack with `docker compose up -d` — **without `--build`**; `--build`
  re-checks both images every time and needs the network.
- **New migration:** `docker compose run --rm migrate` (it sees the same
  source mount), or restart the stack.
- **New npm package:** `docker compose exec frontend npm install`, since
  `node_modules` lives in a container volume.
- **`requirements.txt` change:** rebuild, then refresh `constraints.txt` (the
  frozen versions of everything `requirements.txt` leaves open) and commit it:

  ```bash
  cp backend/constraints.txt /tmp/constraints.old
  grep '^#' /tmp/constraints.old > backend/constraints.txt      # keep the header, drop the pins
  docker compose build backend
  docker run --rm tvasta-backend:local pip freeze \
    | grep -viE '^(torch|torchvision)==' >> backend/constraints.txt
  ```

  Then set `opencv-python` back to the same version as
  `opencv-contrib-python-headless` (the header of the file explains why) and
  rebuild once more.
- **Base image:** it is pinned by tag and digest at the top of
  `backend/Dockerfile`; the comment there gives the three commands to move to
  a newer 3.11 patch release.

## Checks

```bash
docker run --rm tvasta-backend:local python --version         # Python 3.11.x
docker run --rm tvasta-backend:local python -m pip check      # No broken requirements found.
docker run --rm tvasta-backend:local id                       # uid=1000(tvasta)
docker run --rm -v "$PWD:/repo:ro" -w /repo/backend tvasta-backend:local \
  python -m pytest -q -p no:cacheprovider                     # the test suite, inside the image
```

## Troubleshooting

| Symptom | Cause |
|---|---|
| `backend` never starts, `migrate` shows `Exited (1)` | A migration failed: `docker compose logs migrate` |
| `DuplicateObject` / `DuplicateTable` from `migrate` | The database has tables but no `alembic_version` — see [data-model.md](data-model.md#migrations) |
| Port 8000 already in use | A hand-started `uvicorn` is running; stop it or start only `postgres minio` |
| `/static/models/*.glb` is served as `text/plain` | The slim image has no MIME table for `.glb`; the Three.js loader reads the bytes and does not use the header |
| Evidence images do not load in the browser | `MINIO_PUBLIC_ENDPOINT` is not an address the browser can reach |
| `pull access denied for minio/minio` | `docker-compose.override.yml` (selects `pgsty/minio`) is missing |
