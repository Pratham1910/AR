"""FastAPI application entrypoint (Project.md #10, #63)."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import assets, auth, evidence, inspection, metrology, models3d, procedures, video, vision
from app.core.config import get_settings

app = FastAPI(
    title="TVASTA Industrial AR/MR Procedure & QA Platform",
    version="0.1.0",
    description="Phase 1-5 — camera-based procedure execution, deterministic QA, "
    "3D viewer, marker-based physical<->3D registration, tracking, and video-to-procedure extraction.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(assets.router)
app.include_router(procedures.router)
app.include_router(inspection.router)
app.include_router(vision.router)
app.include_router(evidence.router)
app.include_router(models3d.router)
app.include_router(video.router)
app.include_router(metrology.router)

# Serves GLB/glTF files referenced by Model3D.storage_key (Project.md #20,
# Phase 4). A later phase can move this behind MinIO/S3 without changing the
# API contract — the frontend only ever sees a URL.
_settings = get_settings()
_models_dir = Path(_settings.models_3d_dir)
_models_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/models", StaticFiles(directory=str(_models_dir)), name="models_3d")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
